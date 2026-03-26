//! Phase 4 integration test: full 6-layer model training on ANE.
//!
//! Uses synthetic random token data to verify the full pipeline works.
//! Exit criterion: loss decreases over training steps.

use engine::full_model::{
    self, ModelBackwardWorkspace, ModelGrads, ModelOptState, ModelWeights, TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

#[test]
fn six_layer_loss_decreases() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("Compiling 10 kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    println!("Compiled in {:.1}s", t0.elapsed().as_secs_f32());

    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);

    // Overfit config: same data each step, no accumulation, lower LR
    let tc = TrainConfig {
        accum_steps: 1,
        warmup_steps: 0,
        total_steps: 20,
        max_lr: 1e-4,
        loss_scale: 1.0,
        softcap: 0.0,
        grad_clip: 1.0,
        ..Default::default()
    };

    // Fixed tokens at position 0 (same sample every step = overfit test)
    let data: Vec<u16> = (0..cfg.seq + 1)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u16)
        .collect();
    // Pad to minimum size for train_step
    let mut data_padded = vec![0u16; 10000];
    data_padded[..data.len()].copy_from_slice(&data);

    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let mut losses = Vec::new();
    let steps = 10;

    for step in 0..steps {
        let t0 = Instant::now();

        // Direct forward/backward on fixed data (bypass train_step's random sampling)
        let input_tokens: Vec<u32> = data[..cfg.seq].iter().map(|&t| t as u32).collect();
        let target_tokens: Vec<u32> = data[1..cfg.seq + 1].iter().map(|&t| t as u32).collect();

        grads.zero_out();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &input_tokens, &target_tokens, 0.0);
        let loss = fwd.loss;
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

        let elapsed = t0.elapsed().as_secs_f32();
        losses.push(loss);
        println!("step {step}: loss = {loss:.4}, lr = {lr:.6}, time = {elapsed:.2}s");
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    println!(
        "\nloss: {first:.4} → {last:.4} (delta = {:.4})",
        last - first
    );
    assert!(
        last < first,
        "loss should decrease when overfitting on one sample: first={first:.4}, last={last:.4}"
    );
}

#[test]
fn forward_produces_valid_loss() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);

    let tokens: Vec<u32> = (0..cfg.seq).map(|i| (i % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (1..=cfg.seq).map(|i| (i % cfg.vocab) as u32).collect();

    let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, 0.0);
    let loss = fwd.loss;

    println!("Forward pass loss: {loss:.4}");
    // Loss should be around -ln(1/VOCAB) = ln(8192) ≈ 9.01 for random weights
    assert!(loss > 0.0, "loss should be positive");
    assert!(
        loss < 20.0,
        "loss should be reasonable for random init: {loss}"
    );
}
