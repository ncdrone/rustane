//! Phase 3 integration test: single-layer training on ANE.
//!
//! Exit criteria: loss decreases over 10 training steps.

use engine::cpu::adam::AdamConfig;
use engine::layer::{CompiledKernels, LayerGrads, LayerWeights};
use engine::model::ModelConfig;
use engine::training::{self, LayerOptState};

#[test]
fn single_layer_loss_decreases() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("Compiling all 10 kernels...");
    let kernels = CompiledKernels::compile(&cfg);
    println!("All 10 kernels compiled ✓");

    let mut weights = LayerWeights::random(&cfg);
    let mut grads = LayerGrads::zeros(&cfg);
    let mut opt = LayerOptState::zeros(&cfg);
    let adam_cfg = AdamConfig {
        lr: 1e-4,
        weight_decay: 0.0,
        ..Default::default()
    };

    // Fixed random input — use larger magnitudes so MSE target isn't trivially close
    let n = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 13) % 200) as f32 * 0.01 - 1.0)
        .collect();

    let mut losses = Vec::new();
    let steps = 10;

    for t in 1..=steps {
        let loss = training::train_step(
            &cfg,
            &kernels,
            &mut weights,
            &mut grads,
            &mut opt,
            &x,
            t,
            &adam_cfg,
        );
        losses.push(loss);
        println!("step {t}: loss = {loss:.6}");
    }

    // Verify loss decreased
    let first = losses[0];
    let last = *losses.last().unwrap();
    println!("loss: {first:.6} → {last:.6} (delta = {:.6})", last - first);
    assert!(
        last < first,
        "loss should decrease: first={first:.6}, last={last:.6}"
    );
}

#[test]
fn all_10_kernels_compile() {
    let cfg = ModelConfig::gpt_karpathy();
    let _kernels = CompiledKernels::compile(&cfg);
    println!(
        "All 10 kernels compiled ✓ (sdpaFwd, woFwd, ffnFused, ffnBwdW2t, ffnBwdW13t, wotBwd, sdpaBwd1, sdpaBwd2, qBwd, kvBwd)"
    );
}
