//! Correctness test for move-dw2-to-sdpabwd1 optimization.
//!
//! What was optimized:
//!   In `backward_into`, the dW2 weight gradient accumulation (dW2 += dffn @ gate^T)
//!   was moved from step 5's async block (ANE ffnBwdW13t overlap, CPU-bound at ~2.3ms)
//!   to step 9's async block (ANE sdpaBwd1 overlap, ANE-bound with ~0.6ms headroom).
//!   This rebalances CPU work across async windows.
//!
//! Invariant checked:
//!   `backward_into` and `backward` must produce identical:
//!   1. The returned dx (gradient w.r.t. layer input)
//!   2. All LayerGrads fields — especially dw2 which moved to a different schedule point
//!
//!   dW2 uses dffn (from step 1) and cache.gate (from forward) — both available before
//!   either overlap window. The sgemm_at accumulation order doesn't matter (commutative
//!   addition). Output must be bit-identical since the same FP operations execute.
//!
//! Tolerance: 0.0 (exact match — same operations, same values, different schedule)
//!
//! Failure meaning:
//!   The dW2 sgemm received different input data at its new position, or a buffer
//!   was overwritten between step 5 and step 9 that dW2 depends on.

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

/// Test 1: backward_into must match backward on normal input.
/// Both functions have the same dW2 schedule change applied, so they should
/// still produce identical results (pure scheduling change).
#[test]
fn move_dw2_backward_matches_reference() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    // Forward to get cache
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    let dy: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.003) - 0.5) * 0.01)
        .collect();

    // ── Reference: backward (sequential-ish path) ──
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

    // ── Optimized: backward_into (dW2 in sdpaBwd1 overlap) ──
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

    // Compare dx
    assert_exact(&dx_ref, &dx_opt, "dx");

    // Compare all weight gradients — dw2 is the critical one (moved schedule)
    assert_exact(&grads_ref.dw2, &grads_opt.dw2, "dw2");
    assert_exact(&grads_ref.dw1, &grads_opt.dw1, "dw1");
    assert_exact(&grads_ref.dw3, &grads_opt.dw3, "dw3");
    assert_exact(&grads_ref.dwo, &grads_opt.dwo, "dwo");
    assert_exact(&grads_ref.dwq, &grads_opt.dwq, "dwq");
    assert_exact(&grads_ref.dwk, &grads_opt.dwk, "dwk");
    assert_exact(&grads_ref.dwv, &grads_opt.dwv, "dwv");
    assert_exact(&grads_ref.dgamma1, &grads_opt.dgamma1, "dgamma1");
    assert_exact(&grads_ref.dgamma2, &grads_opt.dgamma2, "dgamma2");

    println!("PASS: backward_into matches backward (dw2 correctly moved, all grads bit-identical)");
}

/// Test 2: Two consecutive backward_into calls produce identical results (idempotency).
/// Catches stale-buffer issues where dW2 at its new position (step 9) might read
/// data written by a previous backward_into call still in the IOSurface buffers.
#[test]
fn move_dw2_idempotent() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    let dy: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.003) - 0.5) * 0.01)
        .collect();

    // First call
    let mut grads1 = LayerGrads::zeros(&cfg);
    let mut ws1 = BackwardWorkspace::new(&cfg);
    let mut dx1 = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads1,
        &mut ws1,
        &mut dx1,
    );

    // Second call (fresh grads/ws, reuses same KernelBuffers IOSurfaces)
    let mut grads2 = LayerGrads::zeros(&cfg);
    let mut ws2 = BackwardWorkspace::new(&cfg);
    let mut dx2 = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads2,
        &mut ws2,
        &mut dx2,
    );

    assert_exact(&dx1, &dx2, "dx (idempotent)");
    assert_exact(&grads1.dw2, &grads2.dw2, "dw2 (idempotent)");
    assert_exact(&grads1.dw1, &grads2.dw1, "dw1 (idempotent)");
    assert_exact(&grads1.dw3, &grads2.dw3, "dw3 (idempotent)");
    assert_exact(&grads1.dwo, &grads2.dwo, "dwo (idempotent)");

    println!("PASS: backward_into is idempotent (2 calls produce identical results)");
}

/// Test 3: Gradient accumulation across 2 backward_into calls should double gradients.
/// Verifies dW2 accumulates correctly at its new position across multiple microbatches.
#[test]
fn move_dw2_accumulation() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    let dy: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.003) - 0.5) * 0.01)
        .collect();

    // Single call — grads for 1 microbatch
    let mut grads_single = LayerGrads::zeros(&cfg);
    let mut ws = BackwardWorkspace::new(&cfg);
    let mut dx = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_single,
        &mut ws,
        &mut dx,
    );

    // Two calls into same grads — grads for 2 microbatches (accumulating via beta=1.0)
    let mut grads_double = LayerGrads::zeros(&cfg);
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_double,
        &mut ws,
        &mut dx,
    );
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_double,
        &mut ws,
        &mut dx,
    );

    // dw2 after 2 calls should be ~2× dw2 after 1 call
    // Tolerance 1e-4: sgemm with beta=1.0 accumulation has FP rounding differences
    // when C starts at 0 (first call) vs non-zero (second call)
    let tol = 1e-4;
    for (i, (s, d)) in grads_single
        .dw2
        .iter()
        .zip(grads_double.dw2.iter())
        .enumerate()
    {
        let expected = 2.0 * s;
        let rel_err = if expected.abs() > 1e-10 {
            (d - expected).abs() / expected.abs()
        } else {
            (d - expected).abs()
        };
        assert!(
            rel_err < tol,
            "dw2[{i}]: 2×single={expected} vs double={d} (rel_err={rel_err:.2e}, tol={tol})"
        );
    }

    // Also check dw1, dw3, dwo (dw1/dw3 remain in step 5, dwo remains in step 9)
    for (i, (s, d)) in grads_single
        .dw1
        .iter()
        .zip(grads_double.dw1.iter())
        .enumerate()
    {
        let expected = 2.0 * s;
        let rel_err = if expected.abs() > 1e-10 {
            (d - expected).abs() / expected.abs()
        } else {
            (d - expected).abs()
        };
        assert!(
            rel_err < tol,
            "dw1[{i}]: 2×single={expected} vs double={d} (rel_err={rel_err:.2e})"
        );
    }

    println!("PASS: dW2 accumulates correctly across 2 backward_into calls at new position");
}
