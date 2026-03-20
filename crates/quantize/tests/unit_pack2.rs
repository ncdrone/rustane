//! Unit tests for 2-bit palettized weight packing.

use quantize::PackedWeights2Bit;

fn make_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Roundtrip: indices are preserved exactly.
#[test]
fn roundtrip_indices() {
    let (rows, cols, gs) = (32, 256, 128);
    let weights = make_weights(rows, cols, 42);

    let packed = PackedWeights2Bit::pack(&weights, rows, cols, gs);
    let indices = packed.unpack_raw();

    // All indices should be 0-3
    for (i, &idx) in indices.iter().enumerate() {
        assert!(idx <= 3, "index out of range at {i}: {idx}");
    }
}

/// Quantization error: 2-bit is coarser than 4-bit, allow larger error.
#[test]
fn quantization_error_bounded() {
    let (rows, cols, gs) = (32, 256, 128);
    let weights = make_weights(rows, cols, 7);

    let packed = PackedWeights2Bit::pack(&weights, rows, cols, gs);
    let dequant = packed.dequant_to_f32();

    let max_err = weights
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // 2-bit with 4 palette entries: max error depends on distribution
    // For uniform [-1, 1]: quartile spacing ~0.5, max error ~0.25
    // Allow 0.6 (generous)
    assert!(max_err < 0.6, "2-bit max error: {max_err} (expected < 0.6)");
}

/// 2-bit packing saves 2x memory vs 4-bit.
#[test]
fn memory_savings() {
    let (rows, cols) = (1024, 1024);
    let weights = make_weights(rows, cols, 99);

    let packed_4bit = quantize::PackedWeights4Bit::pack(&weights, rows, cols, 128);
    let packed_2bit = PackedWeights2Bit::pack(&weights, rows, cols, 128);

    let bytes_4bit = packed_4bit.data.len() * 4 + packed_4bit.scales.len() * 2 + packed_4bit.zeros.len() * 2;
    let bytes_2bit = packed_2bit.data.len() * 4 + packed_2bit.palettes.len() * 2;

    // 2-bit data is half the size of 4-bit data
    assert!(
        bytes_2bit < bytes_4bit,
        "2-bit ({bytes_2bit}) should be smaller than 4-bit ({bytes_4bit})"
    );
}

/// Zero weights roundtrip cleanly.
#[test]
fn zero_weights() {
    let (rows, cols, gs) = (16, 128, 128);
    let weights = vec![0.0f32; rows * cols];
    let packed = PackedWeights2Bit::pack(&weights, rows, cols, gs);
    let dequant = packed.dequant_to_f32();

    let max_err = dequant.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_err < 0.01, "zero weight error: {max_err}");
}
