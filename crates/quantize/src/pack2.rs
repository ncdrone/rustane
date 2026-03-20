//! 2-bit weight packing with per-group palettized quantization.
//!
//! For cold experts that aren't accessed frequently, 2-bit packing
//! saves 2x memory vs 4-bit with moderate quality loss.
//!
//! Layout: 16 crumbs (2-bit values) per u32, LSB-first.
//! Each group of `group_size` weights shares a 4-entry palette (f16).

use half::f16;

/// 2-bit palettized weight matrix.
pub struct PackedWeights2Bit {
    /// Packed crumbs: 16 per u32, LSB-first. Length = out_features * in_features / 16.
    pub data: Vec<u32>,
    /// Per-group palette (4 entries each, f16). Length = groups * 4.
    pub palettes: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

impl PackedWeights2Bit {
    /// Pack f32 weights into 2-bit palettized representation.
    pub fn pack(weights: &[f32], out_features: usize, in_features: usize, group_size: usize) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        assert!(in_features % 16 == 0, "in_features must be multiple of 16");
        assert!(in_features % group_size == 0);
        assert!(group_size % 16 == 0);

        let num_groups_per_row = in_features / group_size;
        let packed_per_row = in_features / 16;
        let total_groups = out_features * num_groups_per_row;

        let mut data = vec![0u32; out_features * packed_per_row];
        let mut palettes = Vec::with_capacity(total_groups * 4);

        for row in 0..out_features {
            for g in 0..num_groups_per_row {
                let group_start = row * in_features + g * group_size;
                let group = &weights[group_start..group_start + group_size];

                // Find 4-entry palette via min-max split
                let palette = compute_palette_4(group);
                for &p in &palette {
                    palettes.push(f16::from_f32(p));
                }

                // Quantize each weight to nearest palette entry
                for i in 0..group_size {
                    let val = group[i];
                    let idx = nearest_palette_idx(&palette, val);

                    let col = g * group_size + i;
                    let packed_idx = row * packed_per_row + col / 16;
                    let crumb_pos = col % 16;
                    data[packed_idx] |= (idx as u32) << (crumb_pos * 2);
                }
            }
        }

        Self { data, palettes, out_features, in_features, group_size }
    }

    /// CPU reference dequantization to f32.
    pub fn dequant_to_f32(&self) -> Vec<f32> {
        let packed_per_row = self.in_features / 16;
        let num_groups = self.in_features / self.group_size;
        let mut result = vec![0.0f32; self.out_features * self.in_features];

        for row in 0..self.out_features {
            for col in 0..self.in_features {
                let packed_idx = row * packed_per_row + col / 16;
                let crumb_pos = col % 16;
                let idx = ((self.data[packed_idx] >> (crumb_pos * 2)) & 0x3) as usize;

                let group_idx = row * num_groups + col / self.group_size;
                let palette_base = group_idx * 4;
                result[row * self.in_features + col] = self.palettes[palette_base + idx].to_f32();
            }
        }
        result
    }

    /// Unpack to raw 2-bit indices (0-3).
    pub fn unpack_raw(&self) -> Vec<u8> {
        let packed_per_row = self.in_features / 16;
        let mut result = vec![0u8; self.out_features * self.in_features];

        for row in 0..self.out_features {
            for col in 0..self.in_features {
                let packed_idx = row * packed_per_row + col / 16;
                let crumb_pos = col % 16;
                result[row * self.in_features + col] =
                    ((self.data[packed_idx] >> (crumb_pos * 2)) & 0x3) as u8;
            }
        }
        result
    }
}

/// Compute a 4-entry palette for a group of weights using min-max quantiles.
fn compute_palette_4(group: &[f32]) -> [f32; 4] {
    let mut sorted: Vec<f32> = group.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n < 4 {
        return [sorted[0], sorted[0], sorted[0], sorted[0]];
    }

    // Place palette entries at quartile boundaries
    [
        sorted[n / 8],
        sorted[3 * n / 8],
        sorted[5 * n / 8],
        sorted[7 * n / 8],
    ]
}

fn nearest_palette_idx(palette: &[f32; 4], val: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f32::INFINITY;
    for (i, &p) in palette.iter().enumerate() {
        let d = (val - p).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}
