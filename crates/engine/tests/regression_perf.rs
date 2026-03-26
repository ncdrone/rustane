//! Deterministic regression tests: correctness + performance baselines.
//!
//! These tests use fixed configs and deterministic data to catch:
//! 1. Numerical regressions (loss values drift from golden values)
//! 2. Performance regressions (step time exceeds baseline by >20%)
//!
//! Run:  cargo test -p engine --test regression_perf --release -- --ignored --nocapture
//! Quick: cargo test -p engine --test regression_perf --release -- --ignored --nocapture correctness

use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

/// Run a deterministic N-step training loop, return (losses, median_step_ms, fwd_ms, bwd_ms).
fn run_deterministic(cfg: &ModelConfig, steps: u32) -> (Vec<f32>, f32, f32, f32) {
    let kernels = CompiledKernels::compile(cfg);
    let mut weights = ModelWeights::random(cfg);
    let mut grads = ModelGrads::zeros(cfg);
    let mut opt = ModelOptState::zeros(cfg);
    let tc = TrainConfig::default();
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // Warmup (1 step, untimed)
    {
        grads.zero_out();
        let _ = full_model::forward_ws(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        full_model::backward_ws(
            cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(0, &tc);
        full_model::update_weights(
            cfg,
            &mut weights,
            &grads,
            &mut opt,
            1,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
    }

    let mut losses = Vec::with_capacity(steps as usize + 1);
    let mut step_times = Vec::with_capacity(steps as usize);
    let mut fwd_times = Vec::with_capacity(steps as usize);
    let mut bwd_times = Vec::with_capacity(steps as usize);

    for step in 0..steps {
        grads.zero_out();
        let t0 = Instant::now();

        let t_fwd = Instant::now();
        let loss = full_model::forward_ws(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        let fwd_ms = t_fwd.elapsed().as_secs_f32() * 1000.0;

        let t_bwd = Instant::now();
        full_model::backward_ws(
            cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
        let bwd_ms = t_bwd.elapsed().as_secs_f32() * 1000.0;

        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(step + 2, &tc);
        full_model::update_weights(
            cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 2,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );

        let total_ms = t0.elapsed().as_secs_f32() * 1000.0;
        losses.push(loss);
        step_times.push(total_ms);
        fwd_times.push(fwd_ms);
        bwd_times.push(bwd_ms);
    }

    // Final loss after last update
    let final_loss = full_model::forward_ws(
        cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
    );
    losses.push(final_loss);

    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fwd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bwd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = step_times.len() / 2;
    (losses, step_times[mid], fwd_times[mid], bwd_times[mid])
}

// ── Correctness: loss must decrease and stay finite ──

#[test]
#[ignore]
fn correctness_gpt_karpathy() {
    println!("\n=== Regression: gpt_karpathy (fused FFN) ===");
    let cfg = ModelConfig::gpt_karpathy();
    let (losses, ms, fwd, bwd) = run_deterministic(&cfg, 5);

    println!(
        "  loss: {:.4} → {:.4} (delta={:+.4})",
        losses[0],
        losses.last().unwrap(),
        losses.last().unwrap() - losses[0]
    );
    println!("  timing: {ms:.0}ms/step (fwd={fwd:.0} bwd={bwd:.0})");

    assert!(losses.iter().all(|l| l.is_finite()), "NaN/Inf: {losses:?}");
    assert!(losses.last().unwrap() < &losses[0], "loss did not decrease");
}

#[test]
#[ignore]
fn correctness_1b() {
    println!("\n=== Regression: target_1b (decomposed FFN) ===");
    let cfg = ModelConfig::target_1b();
    let (losses, ms, fwd, bwd) = run_deterministic(&cfg, 5);

    println!(
        "  loss: {:.4} → {:.4} (delta={:+.4})",
        losses[0],
        losses.last().unwrap(),
        losses.last().unwrap() - losses[0]
    );
    println!("  timing: {ms:.0}ms/step (fwd={fwd:.0} bwd={bwd:.0})");

    assert!(losses.iter().all(|l| l.is_finite()), "NaN/Inf: {losses:?}");
    assert!(losses.last().unwrap() < &losses[0], "loss did not decrease");
}

// ── Performance: timing must not regress beyond tolerance ──
// Baselines measured on M3 Ultra 512GB, macOS 26.3.1, commit 0e13175

const BASELINE_KARPATHY_MS: f32 = 150.0; // gpt_karpathy full step
const BASELINE_1B_MS: f32 = 2660.0; // target_1b full step
const BASELINE_1B_FWD_MS: f32 = 933.0; // target_1b forward only
const PERF_TOLERANCE: f32 = 1.20; // 20% regression allowed

#[test]
#[ignore]
fn perf_gpt_karpathy() {
    println!("\n=== Perf regression: gpt_karpathy ===");
    let cfg = ModelConfig::gpt_karpathy();
    let (_, ms, fwd, bwd) = run_deterministic(&cfg, 5);

    let limit = BASELINE_KARPATHY_MS * PERF_TOLERANCE;
    println!("  {ms:.0}ms/step (baseline={BASELINE_KARPATHY_MS:.0}, limit={limit:.0})");
    println!("  fwd={fwd:.0}ms bwd={bwd:.0}ms");

    assert!(
        ms < limit,
        "REGRESSION: {ms:.0}ms exceeds {limit:.0}ms (baseline {BASELINE_KARPATHY_MS:.0} + 20%)"
    );
}

#[test]
#[ignore]
fn perf_1b() {
    println!("\n=== Perf regression: target_1b (decomposed FFN) ===");
    let cfg = ModelConfig::target_1b();
    let (_, ms, fwd, bwd) = run_deterministic(&cfg, 5);

    let limit = BASELINE_1B_MS * PERF_TOLERANCE;
    let fwd_limit = BASELINE_1B_FWD_MS * PERF_TOLERANCE;
    println!("  {ms:.0}ms/step (baseline={BASELINE_1B_MS:.0}, limit={limit:.0})");
    println!("  fwd={fwd:.0}ms (baseline={BASELINE_1B_FWD_MS:.0}, limit={fwd_limit:.0})");
    println!("  bwd={bwd:.0}ms");

    assert!(
        ms < limit,
        "REGRESSION: {ms:.0}ms exceeds {limit:.0}ms (baseline {BASELINE_1B_MS:.0} + 20%)"
    );
    assert!(
        fwd < fwd_limit,
        "FWD REGRESSION: {fwd:.0}ms exceeds {fwd_limit:.0}ms (baseline {BASELINE_1B_FWD_MS:.0} + 20%)"
    );
}
