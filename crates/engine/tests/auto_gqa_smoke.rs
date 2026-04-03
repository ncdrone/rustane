use engine::full_model::{
    self, ModelBackwardWorkspace, ModelGrads, ModelOptState, ModelWeights, TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;

#[test]
fn gqa_six_layer_loss_decreases_on_fixed_sample() {
    let cfg = ModelConfig::gpt_karpathy_gqa();
    let kernels = CompiledKernels::compile(&cfg);
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let tc = TrainConfig {
        accum_steps: 1,
        warmup_steps: 0,
        total_steps: 8,
        max_lr: 1e-4,
        loss_scale: 1.0,
        softcap: 0.0,
        grad_clip: 1.0,
        ..Default::default()
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
        full_model::update_weights(
            &cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 1,
            lr,
            &tc,
            &metal_adam,
            1.0,
        );
    }

    assert!(
        last_loss < first_loss,
        "GQA loss should decrease on a fixed sample: first={first_loss:.4}, last={last_loss:.4}"
    );
}
