//! Verify Rust pack4.rs nibble packing convention.
//! Fatal trap: if Python converter disagrees, every weight is silently wrong.
//!
//! Rust convention (from pack4.rs):
//! - 8 nibbles per u32, LSB-first
//! - nibble 0 = bits [0:3], nibble 1 = bits [4:7], ..., nibble 7 = bits [28:31]
//! - Asymmetric quantization: scale = (max - min) / 15, zero = min
//! - Dequant: val = q * scale + zero
//!
//! Python converter MUST output this exact format.

use quantize::PackedWeights4Bit;

#[test]
fn rust_nibble_order_lsb_first() {
    // Pack known values that produce a known nibble pattern.
    // 8 values in one u32: values [0, 1, 2, ..., 7] map to nibbles [0, 2, 4, 6, 9, 11, 13, 15]
    // with min=0, max=7, scale=7/15≈0.4667, zero=0
    let weights: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let packed = PackedWeights4Bit::pack(&weights, 1, 8, 8);

    // Verify the u32 packing: nibble_i at bits [i*4 : i*4+3]
    let word = packed.data[0];
    for i in 0..8u32 {
        let nibble = (word >> (i * 4)) & 0xF;
        // Values 0..7 with scale=(7/15), zero=0 → q = round(val / scale)
        let expected_q = (weights[i as usize] / (7.0 / 15.0)).round() as u32;
        assert_eq!(
            nibble,
            expected_q.min(15),
            "nibble at position {i}: expected {expected_q}, got {nibble}"
        );
    }
}

#[test]
fn pack_unpack_roundtrip_monotonic() {
    // 128 monotonically increasing values in one group
    let weights: Vec<f32> = (0..128).map(|i| i as f32 / 127.0).collect();
    let packed = PackedWeights4Bit::pack(&weights, 1, 128, 128);

    // Unpack raw nibbles — should be monotonically non-decreasing
    let raw = packed.unpack_raw();
    assert_eq!(raw.len(), 128);
    for i in 1..raw.len() {
        assert!(
            raw[i] >= raw[i - 1],
            "nibble order wrong at index {i}: raw[{i}]={} < raw[{}]={}",
            raw[i],
            i - 1,
            raw[i - 1]
        );
    }
}

#[test]
fn dequant_roundtrip_max_error() {
    // Verify dequantization error is bounded
    let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let packed = PackedWeights4Bit::pack(&weights, 1, 128, 128);
    let dequant = packed.dequant_to_f32();

    let max_err = weights
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // 4-bit with 128 values in range [-0.64, 0.63]: scale ≈ 0.085
    // Max error should be < scale/2 ≈ 0.043
    assert!(
        max_err < 0.05,
        "dequant max error {max_err} exceeds threshold 0.05"
    );
}

#[test]
fn python_convention_documentation() {
    // Document the exact convention Python converter must follow:
    //
    // For a row of in_features values:
    //   1. Compute per-group min/max
    //   2. scale = (max - min) / 15.0  (asymmetric, NOT symmetric)
    //   3. zero = min  (NOT centered at 0)
    //   4. q = clamp(round((val - zero) / scale), 0, 15)
    //   5. Pack 8 consecutive q values into one u32, LSB-first:
    //      word = q[0] | (q[1] << 4) | (q[2] << 8) | ... | (q[7] << 28)
    //   6. Store scales as f16, zeros as f16
    //
    // Python equivalent:
    //   for i in range(0, group_size, 8):
    //       word = 0
    //       for j in range(8):
    //           word |= int(q[i+j]) << (j * 4)
    //       data.append(word)  # as uint32

    // Verify with a concrete example
    let vals = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let packed = PackedWeights4Bit::pack(&vals, 1, 8, 8);
    let word = packed.data[0];

    // min=0, max=7, scale=7/15, zero=0
    // q = [0, 2, 4, 6, 9, 11, 13, 15]  (round(val * 15/7))
    let expected_q: Vec<u32> = vals
        .iter()
        .map(|&v| (v * 15.0 / 7.0).round() as u32)
        .collect();

    let mut expected_word: u32 = 0;
    for (j, &q) in expected_q.iter().enumerate() {
        expected_word |= q << (j as u32 * 4);
    }

    assert_eq!(
        word, expected_word,
        "packed word {word:#010x} != expected {expected_word:#010x}"
    );
}
