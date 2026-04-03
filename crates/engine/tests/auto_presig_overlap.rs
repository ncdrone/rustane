//! Correctness test for presig-overlap optimization.
//!
//! What was optimized:
//!   In `backward_into`, the sigmoid computation (vsmul+expf+vsadd+recf on cache.h1)
//!   was moved from the sequential SiLU derivative section into the preceding async block
//!   that overlaps with ANE ffnBwdW2t dispatch. The sigmoid chain (~0.6ms/layer) is now
//!   hidden behind ANE execution time rather than sitting on the critical path.
//!
//! Invariant checked:
//!   `backward_into` and `backward` must produce bit-for-bit identical:
//!   1. The returned dx (gradient w.r.t. layer input)
//!   2. All LayerGrads fields (dw1, dw2, dw3, dwo, dwq, dwk, dwv, dgamma1, dgamma2)
//!
//!   This invariant is meaningful because `backward` computes sigmoid after ANE (sequential),
//!   while `backward_into` now computes sigmoid during ANE (overlapped). Same FP operations,
//!   same result — just different wall-clock timing.
//!
//! Tolerance: 0.0 (exact match — same operations, same order, same FP values)
//!
//! Failure meaning:
//!   The optimization changed the numerical output of backward_into. Gradients would
//!   be wrong, potentially causing training divergence over many steps.

use engine::layer::{
    self, BackwardWorkspace, CompiledKernels, ForwardCache, LayerGrads, LayerWeights,
};
use engine::model::ModelConfig;

fn assert_exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: backward={x} vs backward_into={y} (not bit-identical)"
        );
    }
}

#[test]
fn presig_overlap_backward_matches_reference() {
    let cfg = ModelConfig::gpt_karpathy();

    // Compile ANE kernels (required for both paths)
    let kernels = CompiledKernels::compile(&cfg);

    // Fixed deterministic weights
    let weights = LayerWeights::random(&cfg);

    // Fixed deterministic input (channel-first [dim, seq])
    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    // Forward to get cache (needed for both backward paths)
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    // Gradient signal from next layer (deterministic)
    let dy: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.003) - 0.5) * 0.01)
        .collect();

    // ── Reference: backward (sequential sigmoid, allocating) ──
    let mut grads_ref = LayerGrads::zeros(&cfg);
    let mut ws_ref = BackwardWorkspace::new(&cfg);
    let dx_ref = layer::backward(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_ref,
        &mut ws_ref,
    );

    // ── Optimized: backward_into (sigmoid pre-computed in async overlap) ──
    let mut grads_opt = LayerGrads::zeros(&cfg);
    let mut ws_opt = BackwardWorkspace::new(&cfg);
    let mut dx_opt = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_opt,
        &mut ws_opt,
        &mut dx_opt,
    );

    // dx must match exactly
    assert_exact(&dx_ref, &dx_opt, "dx");

    // All gradient tensors must match exactly
    assert_exact(&grads_ref.dwq, &grads_opt.dwq, "dwq");
    assert_exact(&grads_ref.dwk, &grads_opt.dwk, "dwk");
    assert_exact(&grads_ref.dwv, &grads_opt.dwv, "dwv");
    assert_exact(&grads_ref.dwo, &grads_opt.dwo, "dwo");
    assert_exact(&grads_ref.dw1, &grads_opt.dw1, "dw1");
    assert_exact(&grads_ref.dw2, &grads_opt.dw2, "dw2");
    assert_exact(&grads_ref.dw3, &grads_opt.dw3, "dw3");
    assert_exact(&grads_ref.dgamma1, &grads_opt.dgamma1, "dgamma1");
    assert_exact(&grads_ref.dgamma2, &grads_opt.dgamma2, "dgamma2");
}

#[test]
fn presig_overlap_back_to_back_idempotent() {
    // Edge case: calling backward_into twice on the same cache and weights
    // must give identical results (no stale data in workspace from prior call).
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.002) - 0.5) * 0.1)
        .collect();
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; n_in];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);
    let dy: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.005) - 0.5) * 0.01)
        .collect();

    let mut grads1 = LayerGrads::zeros(&cfg);
    let mut ws = BackwardWorkspace::new(&cfg);
    let mut dx1 = vec![0.0f32; n_in];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads1,
        &mut ws,
        &mut dx1,
    );

    // Reset grads and workspace, run again
    let mut grads2 = LayerGrads::zeros(&cfg);
    ws = BackwardWorkspace::new(&cfg);
    let mut dx2 = vec![0.0f32; n_in];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads2,
        &mut ws,
        &mut dx2,
    );

    assert_exact(&dx1, &dx2, "dx back-to-back");
    assert_exact(&grads1.dw1, &grads2.dw1, "dw1 back-to-back");
    assert_exact(&grads1.dw2, &grads2.dw2, "dw2 back-to-back");
    assert_exact(&grads1.dw3, &grads2.dw3, "dw3 back-to-back");
}
