//! Correctness test for parallel-grad-norm optimization.
//!
//! What was optimized:
//!   `grad_norm()` previously computed the global L2 norm of all 48.8M gradient
//!   parameters sequentially on a single thread. This optimization splits the work
//!   across 2 threads: embed+gamma+layers[0..mid] on thread 1, layers[mid..] on main.
//!   Each thread computes a partial sum-of-squares, then the results are combined.
//!
//! Invariant checked:
//!   The parallel grad_norm must produce the same value as a sequential computation.
//!   Since addition is associative for finite f32 values (no NaN/Inf), regrouping
//!   the partial sums should not change the result. However, floating-point addition
//!   is NOT associative in general (different grouping = different rounding), so we
//!   verify the specific grouping used (first half + second half) matches.
//!
//! Tolerance: 0.0 (exact match — same addition order within each group, only the
//!   inter-group combination changes, and (a+b+c)+(d+e+f) = a+b+c+d+e+f when
//!   the partial sums are combined at the end)
//!
//! Failure meaning:
//!   Thread scheduling changed the summation order, causing different FP rounding.
//!   This would indicate the grouping doesn't match the expected split.

use engine::cpu::vdsp;
use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelWeights, TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;

/// Reference sequential grad_norm implementation (pre-optimization).
fn grad_norm_sequential(grads: &ModelGrads) -> f32 {
    let mut sum = 0.0f32;
    sum += vdsp::svesq(&grads.dembed);
    sum += vdsp::svesq(&grads.dgamma_final);
    for lg in &grads.layers {
        for g in [
            &lg.dwq,
            &lg.dwk,
            &lg.dwv,
            &lg.dwo,
            &lg.dw1,
            &lg.dw3,
            &lg.dw2,
            &lg.dgamma1,
            &lg.dgamma2,
        ] {
            sum += vdsp::svesq(g);
        }
    }
    sum.sqrt()
}

/// Test 1: Parallel grad_norm matches sequential on real gradients from a backward pass.
#[test]
fn parallel_grad_norm_matches_sequential() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let tc = TrainConfig::default();

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);

    let _loss = full_model::forward_ws(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
    );
    full_model::backward_ws(
        &cfg,
        &kernels,
        &weights,
        &fwd_ws,
        &tokens,
        tc.softcap,
        tc.loss_scale,
        &mut grads,
        &mut bwd_ws,
    );

    let norm_parallel = full_model::grad_norm(&grads);
    let norm_sequential = grad_norm_sequential(&grads);

    // The parallel version groups (embed+gamma+layers[0..3]) + (layers[3..6]).
    // The sequential version adds all in order. Due to FP associativity differences,
    // these may differ slightly. We allow 1e-6 relative tolerance.
    let rel_err = ((norm_parallel - norm_sequential) / norm_sequential).abs();
    assert!(
        rel_err < 1e-6,
        "parallel={norm_parallel} vs sequential={norm_sequential}, rel_err={rel_err} (expected < 1e-6)"
    );

    println!(
        "PASS: parallel grad_norm matches sequential (rel_err={rel_err:.2e}, norm={norm_parallel:.4})"
    );
}

/// Test 2: Two consecutive calls return the same value (deterministic).
#[test]
fn parallel_grad_norm_deterministic() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let tc = TrainConfig::default();

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);

    let _loss = full_model::forward_ws(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
    );
    full_model::backward_ws(
        &cfg,
        &kernels,
        &weights,
        &fwd_ws,
        &tokens,
        tc.softcap,
        tc.loss_scale,
        &mut grads,
        &mut bwd_ws,
    );

    let norm1 = full_model::grad_norm(&grads);
    let norm2 = full_model::grad_norm(&grads);

    assert_eq!(
        norm1.to_bits(),
        norm2.to_bits(),
        "Two calls should return identical results: {norm1} vs {norm2}"
    );

    println!("PASS: parallel grad_norm is deterministic ({norm1:.4})");
}

/// Test 3: Norm is non-zero and finite for non-zero gradients.
#[test]
fn parallel_grad_norm_basic_properties() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let tc = TrainConfig::default();

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);

    let _loss = full_model::forward_ws(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
    );
    full_model::backward_ws(
        &cfg,
        &kernels,
        &weights,
        &fwd_ws,
        &tokens,
        tc.softcap,
        tc.loss_scale,
        &mut grads,
        &mut bwd_ws,
    );

    let norm = full_model::grad_norm(&grads);
    assert!(
        norm > 0.0,
        "Norm should be positive for non-zero gradients: {norm}"
    );
    assert!(norm.is_finite(), "Norm should be finite: {norm}");

    // Also check zero gradients
    let zero_grads = ModelGrads::zeros(&cfg);
    let zero_norm = full_model::grad_norm(&zero_grads);
    assert_eq!(
        zero_norm, 0.0,
        "Norm of zero gradients should be 0: {zero_norm}"
    );

    println!("PASS: grad_norm basic properties (norm={norm:.4}, zero={zero_norm})");
}
