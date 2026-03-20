//! Unit tests for 4-bit weight packing.
//!
//! Invariants:
//! 1. pack -> unpack -> repack produces identical bit patterns (exact)
//! 2. Quantization error is bounded by group_size and value range
//! 3. Edge cases: constant weights, zero weights, negative weights

use quantize::PackedWeights4Bit;

fn make_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Pack -> unpack -> repack must produce identical bit patterns.
#[test]
fn roundtrip_exact_bit_match() {
    let (rows, cols, gs) = (64, 256, 128);
    let weights = make_weights(rows, cols, 42);

    let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
    let nibbles = packed.unpack_raw();
    let repacked = PackedWeights4Bit::repack_raw(
        &nibbles, rows, cols, gs, &packed.scales, &packed.zeros,
    );

    // Exact bit match on packed data
    assert_eq!(packed.data.len(), repacked.data.len());
    for (i, (&a, &b)) in packed.data.iter().zip(repacked.data.iter()).enumerate() {
        assert_eq!(a, b, "packed data mismatch at index {i}: {a:#010x} != {b:#010x}");
    }
}

/// Roundtrip at various sizes.
#[test]
fn roundtrip_various_sizes() {
    for &(rows, cols, gs) in &[
        (1, 128, 128),
        (8, 128, 128),
        (32, 256, 128),
        (128, 1024, 128),
        (256, 4096, 128),
        (16, 256, 32),   // smaller group size
    ] {
        let weights = make_weights(rows, cols, rows as u64 * 1000 + cols as u64);
        let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
        let nibbles = packed.unpack_raw();
        let repacked = PackedWeights4Bit::repack_raw(
            &nibbles, rows, cols, gs, &packed.scales, &packed.zeros,
        );
        assert_eq!(packed.data, repacked.data, "roundtrip failed for {rows}x{cols} gs={gs}");
    }
}

/// All nibble values must be in [0, 15].
#[test]
fn nibble_values_in_range() {
    let weights = make_weights(32, 256, 99);
    let packed = PackedWeights4Bit::pack(&weights, 32, 256, 128);
    let nibbles = packed.unpack_raw();

    for (i, &n) in nibbles.iter().enumerate() {
        assert!(n <= 15, "nibble out of range at index {i}: {n}");
    }
}

/// Quantization error: max abs error should be bounded.
/// For group_size=128 with uniform [-1, 1] weights:
///   scale = 2.0 / 15.0 ≈ 0.133
///   max quantization error = scale / 2 ≈ 0.067
///   plus f16 rounding on scale/zero ≈ small additional error
/// Allow max_error < 0.08 (generous bound).
#[test]
fn quantization_error_bounded() {
    let (rows, cols, gs) = (64, 1024, 128);
    let weights = make_weights(rows, cols, 7);

    let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
    let dequant = packed.dequant_to_f32();

    assert_eq!(dequant.len(), weights.len());

    let max_err = weights
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // For 4-bit with range [-1, 1]: max error ≈ scale/2 ≈ 0.067
    // With f16 scale/zero rounding, allow 0.08
    assert!(
        max_err < 0.08,
        "max quantization error too large: {max_err} (expected < 0.08)"
    );
    // Also verify it's not zero (sanity check that weights actually differ)
    assert!(max_err > 0.0, "quantization error is exactly zero — suspicious");
}

/// Constant weights: all same value should dequant back exactly (scale=1.0, all nibbles=0).
#[test]
fn constant_weights() {
    let (rows, cols, gs) = (8, 128, 128);
    let weights = vec![0.5f32; rows * cols];

    let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
    let dequant = packed.dequant_to_f32();

    // All dequantized values should be 0.5 (or very close due to f16)
    let max_err = dequant.iter().map(|v| (v - 0.5).abs()).fold(0.0f32, f32::max);
    assert!(max_err < 1e-3, "constant weight error: {max_err}");
}

/// Zero weights should roundtrip cleanly.
#[test]
fn zero_weights() {
    let (rows, cols, gs) = (8, 128, 128);
    let weights = vec![0.0f32; rows * cols];

    let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
    let dequant = packed.dequant_to_f32();

    let max_err = dequant.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_err < 1e-3, "zero weight error: {max_err}");
}

/// Negative weights: ensure min/max quantization handles negatives.
#[test]
fn negative_weights() {
    let (rows, cols, gs) = (8, 128, 128);
    let weights: Vec<f32> = (0..rows * cols)
        .map(|i| -2.0 + (i as f32 / (rows * cols) as f32) * 4.0) // range [-2, 2]
        .collect();

    let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
    let dequant = packed.dequant_to_f32();

    let max_err = weights
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Range is 4.0, scale = 4.0/15 ≈ 0.267, max error ≈ 0.133
    assert!(max_err < 0.15, "negative weight error: {max_err}");
}

/// CPU GEMV: dequant + matmul should produce correct results.
#[test]
fn gemv_correctness() {
    let (rows, cols, gs) = (32, 256, 128);
    let weights = make_weights(rows, cols, 123);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 / cols as f32) * 2.0 - 1.0).collect();

    // Reference: f64 matmul on original weights
    let mut y_ref = vec![0.0f64; rows];
    for r in 0..rows {
        for c in 0..cols {
            y_ref[r] += weights[r * cols + c] as f64 * x[c] as f64;
        }
    }

    let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);
    let y_quant = packed.gemv(&x);

    // Tolerance: quantization error * sqrt(cols) roughly
    // Each element has ~0.067 error, 256 elements sum -> ~0.067 * sqrt(256) ≈ 1.07
    // Be generous: allow 2.0
    let max_err = y_ref
        .iter()
        .zip(y_quant.iter())
        .map(|(a, b)| (*a - *b as f64).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_err < 2.0,
        "GEMV error too large: {max_err} (expected < 2.0)"
    );
}
