//! Correctness test: sgemv_f16_par lm_head logits match sgemv_f32 reference.
//!
//! What was optimized: lm_head logit computation now uses sgemv_f16_par directly
//! on mmap'd f16 weights instead of pre-converting to a 4.69 GB f32 Vec.
//! This halves DRAM traffic (2.35 GB vs 4.69 GB for K2 vocab=163840) and
//! frees 4.69 GB RAM for expert page cache.
//!
//! Invariant: f16_par logits match f32 logits within tolerance 1e-2.
//! The f16 path does chunked convert+sgemv per 64-row tile, accumulating in f32.
//! Per-element error: ~sqrt(in_dim)*eps_f16 ≈ sqrt(7168)*5e-4 ≈ 0.042.
//! But after argmax (greedy sampling), the top token should match for
//! well-separated logit distributions. Tolerance 1e-2 catches gross errors.
//!
//! Failure means: sgemv_f16_par transpose/dimension bug, or f16→f32 conversion
//! introduces unacceptable error for logit computation.

use half::f16;

/// f32 reference: standard sgemv on pre-converted f32 weights.
fn logits_f32_ref(w_f32: &[f32], x: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab];
    moe_infer::blas::sgemv_f32(w_f32, x, &mut out, vocab, hidden);
    out
}

/// f16_par optimized: parallel chunked convert+sgemv on f16 weights.
fn logits_f16_par(w_f16: &[f16], x: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab];
    moe_infer::blas::sgemv_f16_par(w_f16, x, &mut out, vocab, hidden);
    out
}

#[test]
fn lm_head_f16_par_matches_f32() {
    // Use realistic dimensions: K2 has vocab=163840, hidden=7168.
    // Scale down for test speed: vocab=4096, hidden=512.
    let vocab = 4096;
    let hidden = 512;

    // Generate deterministic weights and input
    let w_f16: Vec<f16> = (0..vocab * hidden)
        .map(|i| f16::from_f32(((i as f64 * 0.00001).sin() * 0.1) as f32))
        .collect();
    let w_f32: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f64 * 0.01).cos() * 0.5) as f32)
        .collect();

    let ref_logits = logits_f32_ref(&w_f32, &x, vocab, hidden);
    let opt_logits = logits_f16_par(&w_f16, &x, vocab, hidden);

    // Check element-wise tolerance
    let max_diff = ref_logits.iter().zip(opt_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-2,
        "f16_par vs f32 max_diff={max_diff} (expected < 1e-2)");

    // Check argmax matches (greedy sampling must agree)
    let ref_argmax = ref_logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    let opt_argmax = opt_logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    assert_eq!(ref_argmax, opt_argmax,
        "argmax mismatch: f32={ref_argmax} f16_par={opt_argmax}");
}

#[test]
fn lm_head_f16_par_large_vocab() {
    // Larger test: vocab=16384, hidden=1024 (closer to production scale).
    let vocab = 16384;
    let hidden = 1024;

    let w_f16: Vec<f16> = (0..vocab * hidden)
        .map(|i| f16::from_f32(((i as f64 * 0.000003).sin() * 0.05) as f32))
        .collect();
    let w_f32: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f64 * 0.007).cos() * 0.3) as f32)
        .collect();

    let ref_logits = logits_f32_ref(&w_f32, &x, vocab, hidden);
    let opt_logits = logits_f16_par(&w_f16, &x, vocab, hidden);

    let max_diff = ref_logits.iter().zip(opt_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-2,
        "large vocab f16_par vs f32 max_diff={max_diff} (expected < 1e-2)");

    // Argmax must match
    let ref_argmax = ref_logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    let opt_argmax = opt_logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    assert_eq!(ref_argmax, opt_argmax,
        "large vocab argmax mismatch: f32={ref_argmax} f16_par={opt_argmax}");
}

#[test]
fn lm_head_f16_par_edge_single_row() {
    // Edge case: vocab=1, hidden=64 (single output row).
    let vocab = 1;
    let hidden = 64;

    let w_f16: Vec<f16> = (0..vocab * hidden)
        .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
        .collect();
    let w_f32: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.05).cos()).collect();

    let ref_logits = logits_f32_ref(&w_f32, &x, vocab, hidden);
    let opt_logits = logits_f16_par(&w_f16, &x, vocab, hidden);

    let diff = (ref_logits[0] - opt_logits[0]).abs();
    assert!(diff < 1e-2, "single row diff={diff}");
}

#[test]
fn lm_head_f16_par_edge_odd_dimensions() {
    // Edge case: non-power-of-2 dimensions (like K2's vocab=163840 which isn't 2^N).
    let vocab = 1000; // not a power of 2
    let hidden = 300;  // not a multiple of 64

    let w_f16: Vec<f16> = (0..vocab * hidden)
        .map(|i| f16::from_f32(((i as f64 * 0.00007).sin() * 0.2) as f32))
        .collect();
    let w_f32: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f64 * 0.013).cos() * 0.4) as f32)
        .collect();

    let ref_logits = logits_f32_ref(&w_f32, &x, vocab, hidden);
    let opt_logits = logits_f16_par(&w_f16, &x, vocab, hidden);

    let max_diff = ref_logits.iter().zip(opt_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-2,
        "odd dims f16_par vs f32 max_diff={max_diff} (expected < 1e-2)");
}
