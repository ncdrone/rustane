//! 4-bit weight packing with per-group scale and zero point.
//!
//! Layout:
//! - `data`: packed nibbles, 8 per u32, LSB-first
//!   nibble 0 = bits [0:3], nibble 1 = bits [4:7], ..., nibble 7 = bits [28:31]
//! - `scales`/`zeros`: per-group f16 values
//!   group index = row * num_groups_per_row + col / group_size
//!
//! Quantization: asymmetric, min/max per group
//!   scale = (max - min) / 15.0
//!   zero = min
//!   q = clamp(round((val - zero) / scale), 0, 15)
//!   dequant = q * scale + zero

use half::f16;

/// 4-bit quantized weight matrix.
pub struct PackedWeights4Bit {
    /// Packed nibbles: 8 per u32, row-major. Length = out_features * in_features / 8.
    pub data: Vec<u32>,
    /// Per-group scale in f16. Length = out_features * (in_features / group_size).
    pub scales: Vec<f16>,
    /// Per-group zero point in f16. Length = out_features * (in_features / group_size).
    pub zeros: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

impl PackedWeights4Bit {
    /// Number of groups per row.
    #[inline]
    pub fn num_groups_per_row(&self) -> usize {
        self.in_features / self.group_size
    }

    /// Number of packed u32s per row.
    #[inline]
    pub fn packed_per_row(&self) -> usize {
        self.in_features / 8
    }

    /// Pack f32 weights into 4-bit representation.
    /// `weights` is row-major [out_features, in_features].
    pub fn pack(weights: &[f32], out_features: usize, in_features: usize, group_size: usize) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        assert!(in_features % 8 == 0, "in_features must be multiple of 8");
        assert!(in_features % group_size == 0, "in_features must be multiple of group_size");
        assert!(group_size % 8 == 0, "group_size must be multiple of 8");

        let num_groups_per_row = in_features / group_size;
        let packed_per_row = in_features / 8;

        let mut data = vec![0u32; out_features * packed_per_row];
        let mut scales = Vec::with_capacity(out_features * num_groups_per_row);
        let mut zeros = Vec::with_capacity(out_features * num_groups_per_row);

        for row in 0..out_features {
            let row_offset = row * in_features;

            for g in 0..num_groups_per_row {
                let group_start = row_offset + g * group_size;
                let group = &weights[group_start..group_start + group_size];

                let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let range = max - min;
                let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
                let zero = min;

                scales.push(f16::from_f32(scale));
                zeros.push(f16::from_f32(zero));

                for i in 0..group_size {
                    let val = group[i];
                    let q = ((val - zero) / scale).round().clamp(0.0, 15.0) as u32;

                    let col = g * group_size + i;
                    let packed_idx = row * packed_per_row + col / 8;
                    let nibble_pos = col % 8;
                    data[packed_idx] |= q << (nibble_pos * 4);
                }
            }
        }

        Self { data, scales, zeros, out_features, in_features, group_size }
    }

    /// Unpack to raw 4-bit integer values (0-15).
    pub fn unpack_raw(&self) -> Vec<u8> {
        let packed_per_row = self.packed_per_row();
        let mut result = vec![0u8; self.out_features * self.in_features];

        for row in 0..self.out_features {
            for col in 0..self.in_features {
                let packed_idx = row * packed_per_row + col / 8;
                let nibble_pos = col % 8;
                result[row * self.in_features + col] =
                    ((self.data[packed_idx] >> (nibble_pos * 4)) & 0xF) as u8;
            }
        }
        result
    }

    /// Repack from raw nibble values. For roundtrip testing.
    pub fn repack_raw(
        nibbles: &[u8],
        out_features: usize,
        in_features: usize,
        group_size: usize,
        scales: &[f16],
        zeros: &[f16],
    ) -> Self {
        let packed_per_row = in_features / 8;
        let mut data = vec![0u32; out_features * packed_per_row];

        for row in 0..out_features {
            for col in 0..in_features {
                let nibble = nibbles[row * in_features + col] as u32;
                assert!(nibble <= 15, "nibble out of range: {nibble}");
                let packed_idx = row * packed_per_row + col / 8;
                let nibble_pos = col % 8;
                data[packed_idx] |= nibble << (nibble_pos * 4);
            }
        }

        Self {
            data,
            scales: scales.to_vec(),
            zeros: zeros.to_vec(),
            out_features,
            in_features,
            group_size,
        }
    }

    /// CPU reference dequantization to f32.
    /// Returns row-major [out_features, in_features].
    pub fn dequant_to_f32(&self) -> Vec<f32> {
        let packed_per_row = self.packed_per_row();
        let num_groups = self.num_groups_per_row();
        let mut result = vec![0.0f32; self.out_features * self.in_features];

        for row in 0..self.out_features {
            for col in 0..self.in_features {
                let packed_idx = row * packed_per_row + col / 8;
                let nibble_pos = col % 8;
                let q = ((self.data[packed_idx] >> (nibble_pos * 4)) & 0xF) as f32;

                let group_idx = row * num_groups + col / self.group_size;
                let scale = self.scales[group_idx].to_f32();
                let zero = self.zeros[group_idx].to_f32();

                result[row * self.in_features + col] = q * scale + zero;
            }
        }
        result
    }

    /// CPU reference: dequantize + GEMV. y = W @ x.
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features);
        let packed_per_row = self.packed_per_row();
        let num_groups = self.num_groups_per_row();
        let mut y = vec![0.0f32; self.out_features];

        for row in 0..self.out_features {
            let mut sum = 0.0f64; // f64 accumulation for reference precision
            for col in 0..self.in_features {
                let packed_idx = row * packed_per_row + col / 8;
                let nibble_pos = col % 8;
                let q = ((self.data[packed_idx] >> (nibble_pos * 4)) & 0xF) as f32;

                let group_idx = row * num_groups + col / self.group_size;
                let scale = self.scales[group_idx].to_f32();
                let zero = self.zeros[group_idx].to_f32();

                let w = q * scale + zero;
                sum += (w as f64) * (x[col] as f64);
            }
            y[row] = sum as f32;
        }
        y
    }
}
