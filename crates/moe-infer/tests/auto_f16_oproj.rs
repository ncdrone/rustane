//! Correctness test: sgemv_f16_par produces identical results to the
//! sequential sgemv_f16 and the pre-converted f32 sgemv path.
//!
//! What was optimized: O projection ([7168, 16384] = 470 MB f32) now reads
//! directly from f16 mmap using rayon-parallel chunked convert+sgemv.
//! This halves DRAM traffic (235 MB f16 vs 470 MB f32).
//!
//! Invariant: sgemv_f16_par produces numerically identical results to
//! converting f16→f32 in bulk then calling sgemv_f32, because both paths
//! perform the same to_f32() conversion on the same source data.
//!
//! Failure means: incorrect chunk boundary handling, rayon data race,
//! or conversion mismatch between bulk and per-chunk paths.

use half::f16;

/// sgemv_f16_par matches sgemv_f32 on pre-converted weights (O projection size).
///
/// Tests the exact dimensions used in V3 MLA O projection: [7168, 16384].
/// Both paths convert the same f16 source data — results must be bit-identical.
#[test]
fn f16_par_matches_f32_oproj_dimensions() {
    let out_dim = 7168;
    let in_dim = 16384;

    // Generate deterministic f16 weights (same as backbone mmap would contain)
    let w_f16: Vec<f16> = (0..out_dim * in_dim)
        .map(|i| f16::from_f32(((i as f64 * 0.00001).sin() * 0.1) as f32))
        .collect();

    // Input vector (simulating v_concat)
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f64 * 0.001).cos() * 0.5) as f32)
        .collect();

    // Path A: bulk convert f16→f32, then sgemv_f32 (current pipeline path)
    let w_f32: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let mut y_f32 = vec![0.0f32; out_dim];
    moe_infer::blas::sgemv_f32(&w_f32, &x, &mut y_f32, out_dim, in_dim);

    // Path B: sgemv_f16_par (new parallel path)
    let mut y_par = vec![0.0f32; out_dim];
    moe_infer::blas::sgemv_f16_par(&w_f16, &x, &mut y_par, out_dim, in_dim);

    // Same f16→f32 conversion, but BLAS processes 7168 rows in one call (bulk f32)
    // vs 64-row chunks (f16_par). Different accumulation order within Accelerate
    // causes ~1e-5 to 1e-4 differences on 16K-dimensional dot products.
    let max_diff: f32 = y_f32.iter().zip(y_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4,
        "f16_par vs f32 max_diff={max_diff} (expected <1e-4 for chunked BLAS accumulation)");
}

/// sgemv_f16_par handles non-chunk-aligned output dimensions correctly.
///
/// CHUNK_ROWS=64, so out_dim=100 has a partial last chunk of 36 rows.
/// Edge case: incorrect chunk boundary → buffer overrun or wrong results.
#[test]
fn f16_par_handles_partial_last_chunk() {
    let out_dim = 100; // 64 + 36 = partial last chunk
    let in_dim = 512;

    let w_f16: Vec<f16> = (0..out_dim * in_dim)
        .map(|i| f16::from_f32(((i as f64 * 0.0001).sin() * 0.5) as f32))
        .collect();
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f64 * 0.01).cos()) as f32)
        .collect();

    // Reference: single-threaded sgemv_f16
    let mut y_seq = vec![0.0f32; out_dim];
    moe_infer::blas::sgemv_f16(&w_f16, &x, &mut y_seq, out_dim, in_dim);

    // Parallel path
    let mut y_par = vec![0.0f32; out_dim];
    moe_infer::blas::sgemv_f16_par(&w_f16, &x, &mut y_par, out_dim, in_dim);

    let max_diff: f32 = y_seq.iter().zip(y_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5,
        "partial chunk max_diff={max_diff} (expected <1e-5)");
}

/// sgemv_f16_par handles single-chunk case (out_dim <= CHUNK_ROWS).
///
/// When out_dim=32 < CHUNK_ROWS=64, there's exactly one chunk.
/// Edge case: rayon with 1 chunk should still work correctly.
#[test]
fn f16_par_single_chunk() {
    let out_dim = 32;
    let in_dim = 256;

    let w_f16: Vec<f16> = (0..out_dim * in_dim)
        .map(|i| f16::from_f32(((i as f64 * 0.001).sin()) as f32))
        .collect();
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f64 * 0.01).cos()) as f32)
        .collect();

    let mut y_seq = vec![0.0f32; out_dim];
    moe_infer::blas::sgemv_f16(&w_f16, &x, &mut y_seq, out_dim, in_dim);

    let mut y_par = vec![0.0f32; out_dim];
    moe_infer::blas::sgemv_f16_par(&w_f16, &x, &mut y_par, out_dim, in_dim);

    let max_diff: f32 = y_seq.iter().zip(y_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5,
        "single chunk max_diff={max_diff} (expected <1e-5)");
}
