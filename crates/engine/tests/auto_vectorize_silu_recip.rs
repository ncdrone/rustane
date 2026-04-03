//! Correctness test for vectorize-silu-recip optimization.
//!
//! What was optimized:
//!   In all 3 backward functions (backward_timed, backward, backward_into), the SiLU
//!   derivative loop previously computed `sig = 1.0 / (1.0 + exp_neg[i])` as a scalar
//!   fdiv per element. We now precompute sig via `vvrecf` (vectorized reciprocal) before
//!   the loop, then read sig from the precomputed array. The scalar loop no longer contains
//!   any division.
//!
//! Invariant checked:
//!   The output buffers dh1 and dh3 must be numerically identical (within f32 floating-point
//!   tolerance) between the old and new code paths. Both paths compute:
//!     sig = sigmoid(h1) = 1 / (1 + exp(-h1))
//!     dh3[i] = dsilu_raw[i] * h1[i] * sig[i]
//!     dh1[i] = dsilu_raw[i] * h3[i] * sig[i] * (1 + h1[i] * (1 - sig[i]))
//!
//! Tolerance: 1e-5 (reordered floating-point ops: vvrecf vs scalar fdiv have same precision
//! but different rounding; ulp difference expected to be at most 1-2).
//!
//! Failure meaning: The optimization changed the math, not just the execution order.
//!   Gradients would be wrong, potentially causing training divergence.

use engine::cpu::vdsp;

/// Reference implementation: scalar sigmoid + loop (old code path).
fn silu_bwd_scalar(
    h1: &[f32],
    h3: &[f32],
    dsilu_raw: &[f32],
    exp_neg: &mut [f32],
    neg_h1: &mut [f32],
    dh1: &mut [f32],
    dh3: &mut [f32],
) {
    let n = h1.len();
    vdsp::vsmul(h1, -1.0, neg_h1);
    vdsp::expf(neg_h1, exp_neg);
    for i in 0..n {
        let sig = 1.0 / (1.0 + exp_neg[i]);
        let silu_val = h1[i] * sig;
        let silu_deriv = sig * (1.0 + h1[i] * (1.0 - sig));
        dh3[i] = dsilu_raw[i] * silu_val;
        dh1[i] = dsilu_raw[i] * h3[i] * silu_deriv;
    }
}

/// Optimized implementation: vvrecf precomputes sig, then division-free loop (new code path).
fn silu_bwd_recip(
    h1: &[f32],
    h3: &[f32],
    dsilu_raw: &[f32],
    exp_neg: &mut [f32],
    neg_h1: &mut [f32],
    dh1: &mut [f32],
    dh3: &mut [f32],
) {
    let n = h1.len();
    vdsp::vsmul(h1, -1.0, neg_h1);
    vdsp::expf(neg_h1, exp_neg);
    vdsp::vsadd(exp_neg, 1.0, neg_h1); // neg_h1 = 1 + exp(-h1)
    vdsp::recf_inplace(neg_h1); // neg_h1 = sig
    for i in 0..n {
        let sig = neg_h1[i];
        let silu_val = h1[i] * sig;
        let silu_deriv = sig * (1.0 + h1[i] * (1.0 - sig));
        dh3[i] = dsilu_raw[i] * silu_val;
        dh1[i] = dsilu_raw[i] * h3[i] * silu_deriv;
    }
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: scalar={x} vs recip={y}, diff={diff} > tol={tol}"
        );
    }
}

/// Core test: scalar == recip on a given input.
fn run_test(n: usize, h1: &[f32], h3: &[f32], dsilu_raw: &[f32], label: &str) {
    let mut exp_neg_s = vec![0.0f32; n];
    let mut neg_h1_s = vec![0.0f32; n];
    let mut dh1_s = vec![0.0f32; n];
    let mut dh3_s = vec![0.0f32; n];

    let mut exp_neg_r = vec![0.0f32; n];
    let mut neg_h1_r = vec![0.0f32; n];
    let mut dh1_r = vec![0.0f32; n];
    let mut dh3_r = vec![0.0f32; n];

    silu_bwd_scalar(
        h1,
        h3,
        dsilu_raw,
        &mut exp_neg_s,
        &mut neg_h1_s,
        &mut dh1_s,
        &mut dh3_s,
    );
    silu_bwd_recip(
        h1,
        h3,
        dsilu_raw,
        &mut exp_neg_r,
        &mut neg_h1_r,
        &mut dh1_r,
        &mut dh3_r,
    );

    assert_close(&dh1_s, &dh1_r, 1e-5, &format!("{label}/dh1"));
    assert_close(&dh3_s, &dh3_r, 1e-5, &format!("{label}/dh3"));
}

#[test]
fn silu_recip_matches_scalar_typical() {
    // Typical activation range after a few training steps.
    let n = 64;
    let h1: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let h3: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
    let dsilu_raw: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.02).collect();
    run_test(n, &h1, &h3, &dsilu_raw, "typical");
}

#[test]
fn silu_recip_matches_scalar_zeros() {
    // Edge case: all-zero inputs. sig(0)=0.5, silu_val=0, silu_deriv=0.5.
    let n = 16;
    let h1 = vec![0.0f32; n];
    let h3 = vec![0.0f32; n];
    let dsilu_raw = vec![1.0f32; n];
    run_test(n, &h1, &h3, &dsilu_raw, "zeros");
}

#[test]
fn silu_recip_matches_scalar_large_values() {
    // Edge case: large positive/negative values (saturating sigmoid).
    let n = 8;
    let h1 = vec![-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0f32];
    let h3 = vec![1.0f32; n];
    let dsilu_raw = vec![1.0f32; n];
    run_test(n, &h1, &h3, &dsilu_raw, "large_values");
}

#[test]
fn silu_recip_matches_scalar_non_multiple_of_simd_width() {
    // Edge case: n=7 (not a multiple of 4 or 8 — tests SIMD tail handling).
    let n = 7;
    let h1: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let h3: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let dsilu_raw = vec![0.5f32; n];
    run_test(n, &h1, &h3, &dsilu_raw, "non_multiple_of_simd");
}

#[test]
fn silu_recip_matches_scalar_model_size() {
    // Full model size: hidden=2048, seq=512 → n=1,048,576
    // Tests that at production scale the two paths agree.
    let n = 2048 * 512;
    // Pseudo-random inputs using deterministic formula.
    let h1: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 6.28318 / 1000.0).sin()) * 2.0)
        .collect();
    let h3: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 3.14159 / 1000.0).cos()) * 1.5)
        .collect();
    let dsilu_raw: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 2.71828 / 500.0).sin()) * 0.1)
        .collect();
    run_test(n, &h1, &h3, &dsilu_raw, "model_size");
}
