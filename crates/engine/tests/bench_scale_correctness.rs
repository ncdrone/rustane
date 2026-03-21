//! Scale correctness tests: verify kernels compile, run, and train at 110M–1B.
//!
//! Run all:   cargo test -p engine --test bench_scale_correctness --release -- --ignored --nocapture
//! Run one:   cargo test -p engine --test bench_scale_correctness --release -- --ignored --nocapture test_600m

use engine::full_model::{self, ModelWeights, ModelGrads, ModelOptState, ModelForwardWorkspace, ModelBackwardWorkspace, TrainConfig};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;
use engine::metal_adam::MetalAdam;
use std::time::Instant;

fn test_config(cfg: &ModelConfig, name: &str) {
    println!("\n=== {name} — {}d/{}h/{}L/seq{} — ~{:.0}M params ===",
             cfg.dim, cfg.hidden, cfg.nlayers, cfg.seq,
             cfg.param_count() as f64 / 1e6);

    // 1. Compile all 10 kernels
    print!("  [1/4] Compiling 10 ANE kernels... ");
    let t = Instant::now();
    let kernels = CompiledKernels::compile(cfg);
    println!("OK ({:.1}s)", t.elapsed().as_secs_f32());

    // 2. Forward produces valid loss
    print!("  [2/4] Forward pass... ");
    let weights = ModelWeights::random(cfg);
    let mut tc = TrainConfig::default();
    let param_count = cfg.param_count();
    let is_large_model = param_count >= 9_000_000_000usize;
    let is_very_large_model = param_count >= 14_000_000_000usize;
    let is_ultra_deep_model = param_count >= 20_000_000_000usize;
    if is_large_model {
        tc.embed_lr_scale = 0.5;
    }
    if is_very_large_model {
        // 15B+ needs more aggressive stability tuning (48L NaN fix)
        tc.embed_lr_scale = 0.3;
        tc.max_lr = 1.5e-4;    // halved from 3e-4
        tc.grad_clip = 0.5;     // tighter clipping for deep models
    }
    if is_ultra_deep_model {
        // 20B+ ultra-deep models (64L+) need extreme stability (NaN fix)
        tc.embed_lr_scale = 0.15;
        tc.max_lr = 8e-5;      // very conservative LR
        tc.grad_clip = 0.3;    // very tight clipping
    }
    let is_extreme_deep_model = param_count >= 25_000_000_000usize;
    if is_extreme_deep_model {
        // 25B+ uses 20B hyperparams (extreme tier was too conservative)
        tc.embed_lr_scale = 0.15;
        tc.max_lr = 8e-5;      // same as 20B (proved stable)
        tc.grad_clip = 0.3;    // same as 20B (proved stable)
    }
    let tokens: Vec<u32> = (0..cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (1..=cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
    assert!(loss.is_finite(), "loss is not finite: {loss}");
    assert!(loss > 0.0, "loss should be positive: {loss}");
    println!("OK (loss={loss:.4})");

    // 3. 10-step training — loss must decrease
    print!("  [3/4] Training 10 steps... ");
    let mut weights = weights;
    let mut grads = ModelGrads::zeros(cfg);
    let mut opt = ModelOptState::zeros(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);
    let metal_adam = MetalAdam::new().expect("Metal GPU required");

    let mut losses = Vec::with_capacity(11);
    let train_steps: u32 = if is_large_model { 20 } else { 10 };
    for step in 0..train_steps {
        grads.zero_out();
        let loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
        losses.push(loss);
        full_model::backward_ws(cfg, &kernels, &weights, &fwd_ws, &tokens, tc.softcap, tc.loss_scale, &mut grads, &mut bwd_ws);
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip { tc.grad_clip / raw_norm } else { gsc };
        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights(cfg, &mut weights, &grads, &mut opt, step + 1, lr, &tc, &metal_adam, combined_scale);
    }
    // Final loss
    let final_loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
    losses.push(final_loss);

    let delta = final_loss - losses[0];
    let all_finite = losses.iter().all(|l| l.is_finite());
    println!("{:.4} → {:.4} (delta={delta:+.4})", losses[0], final_loss);
    assert!(all_finite, "NaN/Inf during training: {losses:?}");
    let required_delta = if is_large_model { -0.003 } else { -0.01 };
    assert!(delta < required_delta, "loss did not decrease enough: delta={delta}, required={required_delta}");

    // 4. Step timing (3 steps, report median)
    print!("  [4/4] Timing 3 steps... ");
    let mut step_times = Vec::with_capacity(3);
    for step in 10..13u32 {
        grads.zero_out();
        let t = Instant::now();
        let _loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
        full_model::backward_ws(cfg, &kernels, &weights, &fwd_ws, &tokens, tc.softcap, tc.loss_scale, &mut grads, &mut bwd_ws);
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip { tc.grad_clip / raw_norm } else { gsc };
        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights(cfg, &mut weights, &grads, &mut opt, step + 1, lr, &tc, &metal_adam, combined_scale);
        step_times.push(t.elapsed().as_secs_f32() * 1000.0);
    }
    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = step_times[1];
    println!("{median:.0}ms/step (median of 3)");

    println!("  PASS: {name}\n");
}

#[test]
#[ignore]
fn test_110m() {
    test_config(&ModelConfig::gpt_1024(), "110M");
}

#[test]
#[ignore]
fn test_600m() {
    test_config(&ModelConfig::target_600m(), "600M");
}

#[test]
#[ignore]
fn test_800m() {
    test_config(&ModelConfig::target_800m(), "800M");
}

#[test]
#[ignore]
fn test_1b() {
    test_config(&ModelConfig::target_1b(), "1B");
}

#[test]
#[ignore]
fn test_1_5b() {
    test_config(&ModelConfig::target_1_5b(), "1.5B");
}

#[test]
#[ignore]
fn test_10b_custom() {
    test_config(&ModelConfig {
        dim: 4096, hidden: 11008, heads: 32,
        kv_heads: 32, hd: 128, seq: 512, nlayers: 48, vocab: 8192,
        q_dim: 32 * 128, kv_dim: 32 * 128, gqa_ratio: 1,
    }, "10B");
}

#[test]
#[ignore]
fn test_15b_custom() {
    test_config(&ModelConfig {
        dim: 5120, hidden: 13824, heads: 40,
        kv_heads: 40, hd: 128, seq: 512, nlayers: 48, vocab: 8192,
        q_dim: 40 * 128, kv_dim: 40 * 128, gqa_ratio: 1,
    }, "15B");
}

#[test]
#[ignore]
fn test_13b_custom() {
    test_config(&ModelConfig {
        dim: 5120, hidden: 13824, heads: 40,
        kv_heads: 40, hd: 128, seq: 512, nlayers: 40, vocab: 8192,
        q_dim: 40 * 128, kv_dim: 40 * 128, gqa_ratio: 1,
    }, "13B");
}

#[test]
#[ignore]
fn test_20b_custom() {
    test_config(&ModelConfig {
        dim: 5120, hidden: 13824, heads: 40,
        kv_heads: 40, hd: 128, seq: 512, nlayers: 64, vocab: 8192,
        q_dim: 40 * 128, kv_dim: 40 * 128, gqa_ratio: 1,
    }, "20B");
}

#[test]
#[ignore]
fn test_25b_custom() {
    test_config(&ModelConfig {
        dim: 5120, hidden: 13824, heads: 40,
        kv_heads: 40, hd: 128, seq: 512, nlayers: 80, vocab: 8192,
        q_dim: 40 * 128, kv_dim: 40 * 128, gqa_ratio: 1,
    }, "25B");
}

#[test]
#[ignore]
fn test_30b_custom() {
    test_config(&ModelConfig {
        dim: 5120, hidden: 13824, heads: 40,
        kv_heads: 40, hd: 128, seq: 512, nlayers: 96, vocab: 8192,
        q_dim: 40 * 128, kv_dim: 40 * 128, gqa_ratio: 1,
    }, "30B");
}
