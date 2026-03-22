//! FP8 e4m3fn dequantization: LUT + block-wise scale lookup.
//!
//! DeepSeek-V3 stores weights in float8_e4m3fn format with block-wise scales:
//! - weight: [M, K] as raw bytes (1 byte per element)
//! - scale_inv: [ceil(M/128), ceil(K/128)] as f32
//! - dequant: lut[byte] * scale_inv[row/128][col/128]

use std::sync::LazyLock;

/// Block size for FP8 scale lookup (matches DeepSeek-V3 config).
pub const FP8_BLOCK_SIZE: usize = 128;

/// Precomputed LUT: fp8_e4m3fn byte → f32 value (256 entries).
static FP8_LUT: LazyLock<[f32; 256]> = LazyLock::new(build_fp8_e4m3fn_lut);

/// Build the FP8 e4m3fn lookup table.
///
/// Format: 1 sign + 4 exponent + 3 mantissa, bias=7
/// - Exponent 0b1111 with mantissa 0b111 = NaN (maps to 0.0)
/// - Exponent 0b0000 = subnormal: value = (-1)^sign * 2^(-6) * (mantissa/8)
/// - Otherwise: value = (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
fn build_fp8_e4m3fn_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for byte in 0u16..256 {
        let sign_bit = (byte >> 7) & 1;
        let exp_bits = (byte >> 3) & 0xF;
        let mant_bits = byte & 0x7;

        let sign = if sign_bit == 1 { -1.0f32 } else { 1.0f32 };

        let val = if exp_bits == 0xF && mant_bits == 0x7 {
            // NaN → 0.0
            0.0
        } else if exp_bits == 0 {
            // Subnormal: 2^(-6) * (mant/8)
            sign * (2.0f32).powi(-6) * (mant_bits as f32 / 8.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8)
            sign * (2.0f32).powi(exp_bits as i32 - 7) * (1.0 + mant_bits as f32 / 8.0)
        };

        lut[byte as usize] = val;
    }
    lut
}

/// Get the FP8 LUT (lazily initialized).
pub fn fp8_lut() -> &'static [f32; 256] {
    &FP8_LUT
}

/// Dequantize FP8 tensor with block-wise scales.
///
/// - `weight`: [M, K] as raw bytes (1 byte per element, row-major)
/// - `scale_inv`: [ceil(M/block), ceil(K/block)] as f32 (row-major)
/// - Returns: [M, K] as f32
pub fn dequant_fp8_block(
    weight: &[u8],
    scale_inv: &[f32],
    m: usize,
    k: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(weight.len(), m * k);
    let num_col_blocks = (k + block_size - 1) / block_size;
    let lut = fp8_lut();
    let mut out = vec![0.0f32; m * k];

    for i in 0..m {
        let row_block = i / block_size;
        let row_offset = i * k;
        for j in 0..k {
            let col_block = j / block_size;
            let scale = scale_inv[row_block * num_col_blocks + col_block];
            out[row_offset + j] = lut[weight[row_offset + j] as usize] * scale;
        }
    }
    out
}

/// Convert f32 value to f16 bytes (IEEE 754 half-precision).
pub fn f32_to_f16(val: f32) -> half::f16 {
    half::f16::from_f32(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_known_values() {
        let lut = fp8_lut();
        // 0x00 = +0.0 (exp=0, mant=0, subnormal → 0)
        assert_eq!(lut[0x00], 0.0);
        // 0x38 = 0_0111_000 → exp=7, mant=0 → 2^(7-7) * 1.0 = 1.0
        assert_eq!(lut[0x38], 1.0);
        // 0xB8 = 1_0111_000 → -1.0
        assert_eq!(lut[0xB8], -1.0);
        // 0x39 = 0_0111_001 → 2^0 * (1 + 1/8) = 1.125
        assert_eq!(lut[0x39], 1.125);
        // 0x40 = 0_1000_000 → 2^(8-7) * 1.0 = 2.0
        assert_eq!(lut[0x40], 2.0);
        // 0x30 = 0_0110_000 → 2^(6-7) * 1.0 = 0.5
        assert_eq!(lut[0x30], 0.5);
        // 0x7F = 0_1111_111 → NaN → 0.0
        assert_eq!(lut[0x7F], 0.0);
        // 0xFF = 1_1111_111 → -NaN → 0.0
        assert_eq!(lut[0xFF], 0.0);
    }

    #[test]
    fn lut_max_value() {
        let lut = fp8_lut();
        // 0x7E = 0_1111_110 → 2^(15-7) * (1 + 6/8) = 256 * 1.75 = 448.0
        assert_eq!(lut[0x7E], 448.0);
        // 0xFE = 1_1111_110 → -448.0
        assert_eq!(lut[0xFE], -448.0);
    }

    #[test]
    fn lut_subnormals() {
        let lut = fp8_lut();
        // 0x01 = 0_0000_001 → 2^(-6) * (1/8) = 0.001953125
        assert!((lut[0x01] - 0.001953125).abs() < 1e-9);
        // 0x07 = 0_0000_111 → 2^(-6) * (7/8) = 0.013671875
        assert!((lut[0x07] - 0.013671875).abs() < 1e-9);
    }

    #[test]
    fn lut_all_256_no_nan_inf() {
        let lut = fp8_lut();
        for byte in 0..256 {
            assert!(!lut[byte].is_nan(), "LUT[{byte:#04x}] is NaN");
            assert!(!lut[byte].is_infinite(), "LUT[{byte:#04x}] is Inf");
        }
    }

    #[test]
    fn lut_symmetry() {
        let lut = fp8_lut();
        // Positive/negative symmetry: byte XOR 0x80 flips sign
        for byte in 0u8..128 {
            let neg_byte = byte | 0x80;
            if lut[byte as usize] == 0.0 && lut[neg_byte as usize] == 0.0 {
                continue; // both NaN→0 or both zero
            }
            assert!(
                (lut[byte as usize] + lut[neg_byte as usize]).abs() < 1e-9,
                "Asymmetry at {byte:#04x}: +{} vs -{}",
                lut[byte as usize], lut[neg_byte as usize]
            );
        }
    }

    #[test]
    fn block_dequant_basic() {
        // 2x2 matrix, block_size=2 (one block)
        let weight = vec![0x38u8, 0x40, 0x30, 0x38]; // [1.0, 2.0; 0.5, 1.0]
        let scale_inv = vec![3.0f32]; // single block
        let out = dequant_fp8_block(&weight, &scale_inv, 2, 2, 2);
        assert_eq!(out, vec![3.0, 6.0, 1.5, 3.0]);
    }

    #[test]
    fn block_dequant_multi_block() {
        // 2x2 matrix, block_size=1 (4 blocks, each element its own block)
        let weight = vec![0x38u8, 0x38, 0x38, 0x38]; // all 1.0
        let scale_inv = vec![1.0f32, 2.0, 3.0, 4.0]; // [2x2] scales
        let out = dequant_fp8_block(&weight, &scale_inv, 2, 2, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn block_dequant_partial_block() {
        // 3x3 matrix, block_size=2 → 2x2 blocks (partial at edges)
        let weight = vec![0x38u8; 9]; // all 1.0
        let scale_inv = vec![1.0f32, 2.0, 3.0, 4.0]; // [2x2] block scales
        let out = dequant_fp8_block(&weight, &scale_inv, 3, 3, 2);
        // Row 0: cols 0-1 → block (0,0)=1.0, col 2 → block (0,1)=2.0
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 2.0);
        // Row 1: cols 0-1 → block (0,0)=1.0, col 2 → block (0,1)=2.0
        assert_eq!(out[3], 1.0);
        assert_eq!(out[5], 2.0);
        // Row 2: cols 0-1 → block (1,0)=3.0, col 2 → block (1,1)=4.0
        assert_eq!(out[6], 3.0);
        assert_eq!(out[8], 4.0);
    }
}
