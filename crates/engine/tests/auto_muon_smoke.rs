use engine::full_model::{
    self, ModelBackwardWorkspace, ModelGrads, ModelOptState, ModelWeights, MuonConfig, TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::metal_muon::MetalMuon;
use engine::model::{FfnActivation, ModelConfig};

#[test]
fn muon_update_decreases_loss_on_fixed_sample() {
    let cfg = ModelConfig {
        dim: 128,
        hidden: 320,
        heads: 1,
        kv_heads: 1,
        hd: 128,
        seq: 64,
        nlayers: 1,
        vocab: 8192,
        q_dim: 128,
        kv_dim: 128,
        gqa_ratio: 1,
        ffn_activation: FfnActivation::SwiGlu,
    };
    let kernels = CompiledKernels::compile(&cfg);
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let metal_muon = MetalMuon::new().expect("Metal GPU required");
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let tc = TrainConfig {
        accum_steps: 1,
        warmup_steps: 1,
        total_steps: 8,
        max_lr: 1e-4,
        loss_scale: 1.0,
        softcap: 0.0,
        grad_clip: 1.0,
        ..Default::default()
    };
    let muon_cfg = MuonConfig {
        muon_lr: 0.005,
        muon_momentum: 0.95,
        newton_schulz_steps: 3,
    };

    let data: Vec<u32> = (0..cfg.seq + 1)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let input_tokens = data[..cfg.seq].to_vec();
    let target_tokens = data[1..cfg.seq + 1].to_vec();

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    for step in 0..4 {
        grads.zero_out();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &input_tokens, &target_tokens, 0.0);
        if step == 0 {
            first_loss = fwd.loss;
        }
        last_loss = fwd.loss;
        full_model::backward(
            &cfg,
            &kernels,
            &weights,
            &fwd,
            &input_tokens,
            0.0,
            1.0,
            &mut grads,
            &mut bwd_ws,
        );

        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights_muon(
            &cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 1,
            lr,
            &tc,
            &muon_cfg,
            &metal_adam,
            &metal_muon,
            1.0,
        );
    }

    assert!(
        last_loss < first_loss,
        "Muon loss should decrease on a fixed sample: first={first_loss:.4}, last={last_loss:.4}"
    );
}
