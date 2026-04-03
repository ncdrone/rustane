//! Correctness test for fuse-qbwd-kvbwd-overlap optimization.
//!
//! What was optimized:
//!   In `backward_into`, the kvBwd ANE dispatch previously ran fully sequentially
//!   (stage → run → read) with no CPU overlap. This optimization:
//!   1. Moves wkt/wvt transposes from the qBwd scope to the sdpaBwd2 overlap
//!   2. Stages kvBwd IOSurface before the qBwd scope
//!   3. Combines qBwd + kvBwd ANE dispatches in one async scope, overlapping
//!      both ANE runs with the CPU dWq+dWk+dWv accumulations
//!
//! Invariant checked:
//!   `backward_into` and `backward` must produce identical:
//!   1. The returned dx (gradient w.r.t. layer input)
//!   2. All LayerGrads fields (dw1, dw2, dw3, dwo, dwq, dwk, dwv, dgamma1, dgamma2)
//!
//!   This is a pure scheduling change — same FP operations, same order, different
//!   wall-clock overlap pattern. Output must be bit-identical.
//!
//! Tolerance: 0.0 (exact match — same operations, same values)
//!
//! Failure meaning:
//!   The scheduling change altered numerical output. This would mean either:
//!   - A staging error (wrong data written to IOSurface)
//!   - A read order error (reading output before ANE finished)
//!   - A buffer reuse conflict (stale data in pre-staged buffer)

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

/// Test 1: backward_into must match backward on normal input
#[test]
fn fuse_qbwd_kvbwd_backward_matches_reference() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    // Fixed deterministic input (channel-first [dim, seq])
    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    // Forward to get cache
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    // Gradient signal from next layer
    let dy: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.003) - 0.5) * 0.01)
        .collect();

    // ── Reference: backward (sequential path) ──
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

    // ── Optimized: backward_into (fused qBwd+kvBwd overlap) ──
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

    // Compare all weight gradients
    assert_exact(&grads_ref.dwq, &grads_opt.dwq, "dwq");
    assert_exact(&grads_ref.dwk, &grads_opt.dwk, "dwk");
    assert_exact(&grads_ref.dwv, &grads_opt.dwv, "dwv");
    assert_exact(&grads_ref.dwo, &grads_opt.dwo, "dwo");
    assert_exact(&grads_ref.dw1, &grads_opt.dw1, "dw1");
    assert_exact(&grads_ref.dw2, &grads_opt.dw2, "dw2");
    assert_exact(&grads_ref.dw3, &grads_opt.dw3, "dw3");
    assert_exact(&grads_ref.dgamma1, &grads_opt.dgamma1, "dgamma1");
    assert_exact(&grads_ref.dgamma2, &grads_opt.dgamma2, "dgamma2");

    println!("PASS: backward_into matches backward (all grads bit-identical)");
}

/// Test 2: Two consecutive backward_into calls produce identical results (idempotency).
/// This catches stale-buffer issues where the first call leaves residual data in
/// IOSurface buffers that corrupts the second call.
#[test]
fn fuse_qbwd_kvbwd_idempotent() {
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

    // Call 1
    let mut grads1 = LayerGrads::zeros(&cfg);
    let mut ws = BackwardWorkspace::new(&cfg);
    let mut dx1 = vec![0.0f32; cfg.dim * cfg.seq];
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

    // Call 2 (reusing workspace — tests buffer reuse safety)
    let mut grads2 = LayerGrads::zeros(&cfg);
    let mut dx2 = vec![0.0f32; cfg.dim * cfg.seq];
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

    assert_exact(&dx1, &dx2, "dx_idempotent");
    assert_exact(&grads1.dwq, &grads2.dwq, "dwq_idempotent");
    assert_exact(&grads1.dwk, &grads2.dwk, "dwk_idempotent");
    assert_exact(&grads1.dwv, &grads2.dwv, "dwv_idempotent");

    println!("PASS: two consecutive backward_into calls produce identical output");
}

/// Test 3: Different gradient inputs produce different outputs.
/// Sanity check that the optimization isn't silently ignoring inputs.
#[test]
fn fuse_qbwd_kvbwd_different_inputs() {
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

    // Two different gradient signals
    let dy_a: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.003) - 0.5) * 0.01)
        .collect();
    let dy_b: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.007) + 0.3) * 0.01)
        .collect();

    let mut grads_a = LayerGrads::zeros(&cfg);
    let mut ws = BackwardWorkspace::new(&cfg);
    let mut dx_a = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy_a,
        &mut grads_a,
        &mut ws,
        &mut dx_a,
    );

    let mut grads_b = LayerGrads::zeros(&cfg);
    let mut dx_b = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy_b,
        &mut grads_b,
        &mut ws,
        &mut dx_b,
    );

    // dx values should differ
    let differ = dx_a
        .iter()
        .zip(dx_b.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(differ, "Different inputs should produce different dx");

    println!(
        "PASS: different inputs produce different outputs (optimization isn't ignoring inputs)"
    );
}
