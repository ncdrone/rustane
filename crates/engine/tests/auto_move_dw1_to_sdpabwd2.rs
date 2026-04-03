//! Correctness test for move-dw1-to-sdpabwd2 optimization.
//!
//! What was optimized:
//!   In `backward_into`, the dW1 weight gradient accumulation (dW1 += x2norm @ dh1^T)
//!   was moved from step 5's async block (ANE ffnBwdW13t overlap, CPU-bound at ~1.44ms)
//!   to step 10's async block (ANE sdpaBwd2 overlap, which had ~0.3ms CPU vs ~1ms+ ANE).
//!   This rebalances CPU work across async windows, saving ~0.76ms/layer.
//!
//! Invariant checked:
//!   `backward_into` and `backward` must produce identical:
//!   1. The returned dx (gradient w.r.t. layer input)
//!   2. All LayerGrads fields — especially dw1 which moved to a different schedule point
//!
//!   dW1 uses cache.x2norm (from forward) and ws.dh1 (from step 3 SiLU backward) —
//!   both available and unmodified between step 5 and step 10. The sgemm_at
//!   accumulation is position-independent. Output must be bit-identical.
//!
//! Tolerance: 0.0 (exact match — same operations, same values, different schedule)
//!
//! Failure meaning:
//!   The dW1 sgemm received different input data at its new position (step 10),
//!   or a buffer (x2norm or dh1) was overwritten between step 5 and step 10.

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

/// Test 1: backward_into must match backward (both have dW1 at step 10).
/// Pure scheduling change — both paths execute identical sgemm calls.
#[test]
fn move_dw1_backward_matches_reference() {
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

    // Reference: backward
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

    // Optimized: backward_into
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

    assert_exact(&dx_ref, &dx_opt, "dx");
    assert_exact(&grads_ref.dw1, &grads_opt.dw1, "dw1");
    assert_exact(&grads_ref.dw2, &grads_opt.dw2, "dw2");
    assert_exact(&grads_ref.dw3, &grads_opt.dw3, "dw3");
    assert_exact(&grads_ref.dwo, &grads_opt.dwo, "dwo");
    assert_exact(&grads_ref.dwq, &grads_opt.dwq, "dwq");
    assert_exact(&grads_ref.dwk, &grads_opt.dwk, "dwk");
    assert_exact(&grads_ref.dwv, &grads_opt.dwv, "dwv");
    assert_exact(&grads_ref.dgamma1, &grads_opt.dgamma1, "dgamma1");
    assert_exact(&grads_ref.dgamma2, &grads_opt.dgamma2, "dgamma2");

    println!(
        "PASS: backward_into matches backward (dw1 correctly moved to step 10, all grads bit-identical)"
    );
}

/// Test 2: Idempotency — two consecutive calls produce identical results.
/// Catches stale-buffer issues where dW1 at step 10 might read data from
/// a previous backward_into call still in IOSurface buffers.
#[test]
fn move_dw1_idempotent() {
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

    // Second call (fresh grads/ws)
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
    assert_exact(&grads1.dw1, &grads2.dw1, "dw1 (idempotent)");
    assert_exact(&grads1.dw2, &grads2.dw2, "dw2 (idempotent)");
    assert_exact(&grads1.dw3, &grads2.dw3, "dw3 (idempotent)");

    println!(
        "PASS: backward_into is idempotent (dw1 at step 10 produces identical results across calls)"
    );
}

/// Test 3: Gradient accumulation across 2 backward_into calls should double gradients.
/// Verifies dW1 accumulates correctly at its new position (step 10) across microbatches.
#[test]
fn move_dw1_accumulation() {
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

    // Single call
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

    // Double call into same grads (accumulating via beta=1.0)
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

    let tol = 1e-4;
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
            "dw1[{i}]: 2×single={expected} vs double={d} (rel_err={rel_err:.2e}, tol={tol})"
        );
    }

    // Also verify dw3 (remains in step 5) and dw2 (in step 9)
    for (i, (s, d)) in grads_single
        .dw3
        .iter()
        .zip(grads_double.dw3.iter())
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
            "dw3[{i}]: 2×single={expected} vs double={d} (rel_err={rel_err:.2e})"
        );
    }

    println!("PASS: dW1 accumulates correctly across 2 backward_into calls at step 10");
}
