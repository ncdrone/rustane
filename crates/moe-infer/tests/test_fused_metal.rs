//! Metal kernel correctness tests (v2 shader, fused kernels).

use moe_kernels::{MetalDequantGemv, ExpertGemvOp};
use half::f16;
use quantize::PackedWeights4Bit;

/// Pack f32 weights into 4-bit quantized format for testing.
fn pack_test_weights(w: &[f32], out_features: usize, in_features: usize, group_size: usize) -> Vec<u8> {
    let num_groups = out_features * (in_features / group_size);
    let packed_u32s = out_features * in_features / 8;

    // Quantize: find per-group min/max, map to 0..15
    let mut scales = vec![f16::ZERO; num_groups];
    let mut zeros = vec![f16::ZERO; num_groups];
    let mut nibbles = vec![0u8; out_features * in_features];

    for row in 0..out_features {
        let groups_per_row = in_features / group_size;
        for g in 0..groups_per_row {
            let start = row * in_features + g * group_size;
            let group_vals = &w[start..start + group_size];
            let min_v = group_vals.iter().cloned().fold(f32::MAX, f32::min);
            let max_v = group_vals.iter().cloned().fold(f32::MIN, f32::max);

            let range = max_v - min_v;
            let scale = if range > 1e-10 { range / 15.0 } else { 1.0 };
            let zero = min_v;

            let gidx = row * groups_per_row + g;
            scales[gidx] = f16::from_f32(scale);
            zeros[gidx] = f16::from_f32(zero);

            for i in 0..group_size {
                let q = ((group_vals[i] - zero) / scale).round().max(0.0).min(15.0) as u8;
                nibbles[start + i] = q;
            }
        }
    }

    // Pack nibbles into uint32
    let mut packed = vec![0u32; packed_u32s];
    for row in 0..out_features {
        let packed_per_row = in_features / 8;
        for pi in 0..packed_per_row {
            let base = row * in_features + pi * 8;
            let mut val = 0u32;
            for n in 0..8 {
                val |= (nibbles[base + n] as u32) << (n * 4);
            }
            packed[row * packed_per_row + pi] = val;
        }
    }

    // Build bytes: packed_data || scales_f16 || zeros_f16
    let mut bytes = Vec::new();
    for &p in &packed {
        bytes.extend_from_slice(&p.to_ne_bytes());
    }
    for &s in &scales {
        bytes.extend_from_slice(&s.to_ne_bytes());
    }
    for &z in &zeros {
        bytes.extend_from_slice(&z.to_ne_bytes());
    }
    bytes
}

/// CPU reference GEMV using the same quantized format.
fn cpu_dequant_gemv(bytes: &[u8], x: &[f32], out_features: usize, in_features: usize, group_size: usize) -> Vec<f32> {
    let packed_u32s = out_features * in_features / 8;
    let packed_bytes = packed_u32s * 4;
    let num_groups = out_features * (in_features / group_size);

    let packed = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const u32, packed_u32s)
    };
    let scales = unsafe {
        std::slice::from_raw_parts(bytes[packed_bytes..].as_ptr() as *const f16, num_groups)
    };
    let zeros = unsafe {
        std::slice::from_raw_parts(bytes[packed_bytes + num_groups * 2..].as_ptr() as *const f16, num_groups)
    };

    let groups_per_row = in_features / group_size;
    let packed_per_row = in_features / 8;
    let groups_per_8 = group_size / 8;

    let mut y = vec![0.0f32; out_features];
    for row in 0..out_features {
        let mut sum = 0.0f64;
        for pi in 0..packed_per_row {
            let col = pi * 8;
            let group_idx = row * groups_per_row + pi / groups_per_8;
            let scale = scales[group_idx].to_f32();
            let zero = zeros[group_idx].to_f32();
            let pack = packed[row * packed_per_row + pi];

            for n in 0..8u32 {
                let nibble = (pack >> (n * 4)) & 0xF;
                let w = nibble as f32 * scale + zero;
                sum += w as f64 * x[(col + n as usize)] as f64;
            }
        }
        y[row] = sum as f32;
    }
    y
}

#[test]
fn v2_shader_matches_cpu_reference() {
    let metal = MetalDequantGemv::new().expect("Metal GPU required");

    // Test at inference dimensions: [768, 2048] (moe_inter × hidden)
    let out_features = 768;
    let in_features = 2048;
    let group_size = 32;

    // Generate test weights
    let w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32 * 0.001).sin() * 0.5)
        .collect();
    let x: Vec<f32> = (0..in_features)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let bytes = pack_test_weights(&w, out_features, in_features, group_size);

    // CPU reference
    let cpu_y = cpu_dequant_gemv(&bytes, &x, out_features, in_features, group_size);

    // Metal v2 (via batch_gemv_mmap which now uses v2 pipeline)
    let mmap_buf = metal.wrap_mmap(&bytes);
    let packed_u32s = out_features * in_features / 8;
    let num_groups = out_features * (in_features / group_size);

    let op = ExpertGemvOp {
        packed_offset: 0,
        scales_offset: packed_u32s * 4,
        zeros_offset: packed_u32s * 4 + num_groups * 2,
        out_features,
        in_features,
        group_size,
    };

    let metal_y = metal.batch_gemv_mmap(&mmap_buf, &[op], &[&x]);
    let gpu_y = &metal_y[0];

    let max_diff = cpu_y.iter().zip(gpu_y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    eprintln!("V2 shader vs CPU: max_diff={max_diff:.6} (out={out_features}, in={in_features})");
    assert!(max_diff < 1e-3, "V2 shader diverges from CPU reference: max_diff={max_diff}");
}

#[test]
fn v2_shader_multiple_sizes() {
    let metal = MetalDequantGemv::new().expect("Metal GPU required");

    // Test various sizes including non-multiple-of-8 out_features
    let cases = [
        (768, 2048, 32),   // gate/up: [moe_inter, hidden]
        (2048, 768, 32),   // down: [hidden, moe_inter]
        (64, 2048, 32),    // small (router-like)
        (100, 2048, 32),   // non-multiple-of-8 out_features (bounds check)
    ];

    for (out_features, in_features, group_size) in cases {
        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();
        let x: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();

        let bytes = pack_test_weights(&w, out_features, in_features, group_size);
        let cpu_y = cpu_dequant_gemv(&bytes, &x, out_features, in_features, group_size);

        let mmap_buf = metal.wrap_mmap(&bytes);
        let packed_u32s = out_features * in_features / 8;
        let num_groups = out_features * (in_features / group_size);

        let op = ExpertGemvOp {
            packed_offset: 0,
            scales_offset: packed_u32s * 4,
            zeros_offset: packed_u32s * 4 + num_groups * 2,
            out_features, in_features, group_size,
        };

        let metal_y = metal.batch_gemv_mmap(&mmap_buf, &[op], &[&x]);
        let gpu_y = &metal_y[0];

        let max_diff = cpu_y.iter().zip(gpu_y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        eprintln!("  [{out_features},{in_features}]: max_diff={max_diff:.6}");
        assert!(max_diff < 1e-3,
            "V2 diverges at [{out_features},{in_features}]: max_diff={max_diff}");
    }
}
