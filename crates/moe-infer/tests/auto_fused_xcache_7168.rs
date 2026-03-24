//! Correctness test: fused gate+up+SiLU shader with in_features=7168 (K2 hidden_size).
//!
//! What was fixed: x_cache[4096] in FUSED_GATE_UP_SILU_SHADER was undersized for
//! in_features=7168 — writes to x_cache[4096..7167] were OOB threadgroup memory.
//! Fixed to x_cache[7168] (28KB, fits in 32KB threadgroup memory limit).
//!
//! Invariant: fused GPU kernel must match CPU reference for ALL columns, not just 0..4095.
//!
//! Failure means: x_cache is still too small, or threadgroup memory allocation is wrong.

use moe_kernels::{MetalDequantGemv, FusedGateUpSiluOp};
use half::f16;

/// Pack f32 weights into 4-bit quantized format.
fn pack_test_weights(w: &[f32], out_features: usize, in_features: usize, group_size: usize) -> Vec<u8> {
    let num_groups = out_features * (in_features / group_size);
    let packed_u32s = out_features * in_features / 8;

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

/// CPU reference: dequant GEMV.
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

/// Core test: fused gate+up+SiLU at K2 dimensions (in_features=7168).
/// This exercises x_cache[4096..7167] which was OOB before the fix.
#[test]
fn fused_k2_dims_7168_matches_cpu() {
    let mut metal = MetalDequantGemv::new().expect("Metal GPU required");

    // K2 dimensions: expert gate/up are [2048, 7168] (out=moe_inter, in=hidden)
    let out_features = 2048;
    let in_features = 7168;
    let group_size = 128;

    // Init scratch to pre-cache constant buffers for these dimensions
    metal.init_scratch(in_features, out_features, 8, group_size);

    // Use distinct non-zero patterns that make columns 4096+ matter
    let gate_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32 * 0.0001).sin() * 0.3)
        .collect();
    let up_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32 * 0.00013 + 0.5).cos() * 0.3)
        .collect();

    // x with large values in columns 4096+ to amplify any OOB x_cache reads
    let x: Vec<f32> = (0..in_features)
        .map(|i| if i >= 4096 { 2.0 * (i as f32 * 0.01).cos() } else { (i as f32 * 0.01).cos() })
        .collect();

    let gate_bytes = pack_test_weights(&gate_w, out_features, in_features, group_size);
    let up_bytes = pack_test_weights(&up_w, out_features, in_features, group_size);

    // CPU reference: gate → SiLU → × up
    let gate_out = cpu_dequant_gemv(&gate_bytes, &x, out_features, in_features, group_size);
    let up_out = cpu_dequant_gemv(&up_bytes, &x, out_features, in_features, group_size);
    let cpu_fused: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    // GPU fused
    let packed_u32s = out_features * in_features / 8;
    let num_groups = out_features * (in_features / group_size);
    let matrix_bytes = packed_u32s * 4 + num_groups * 2 + num_groups * 2;

    let mut combined_mmap = Vec::new();
    combined_mmap.extend_from_slice(&gate_bytes);
    combined_mmap.extend_from_slice(&up_bytes);
    let mmap_buf = metal.wrap_mmap(&combined_mmap);

    let fused_op = FusedGateUpSiluOp {
        gate_packed_offset: 0,
        gate_scales_offset: packed_u32s * 4,
        gate_zeros_offset: packed_u32s * 4 + num_groups * 2,
        up_packed_offset: matrix_bytes,
        up_scales_offset: matrix_bytes + packed_u32s * 4,
        up_zeros_offset: matrix_bytes + packed_u32s * 4 + num_groups * 2,
        out_features, in_features, group_size,
    };

    let fused_result = metal.batch_fused_gate_up_silu_mmap(&mmap_buf, &[fused_op], &[&x]);
    let gpu_fused = &fused_result[0];

    let max_diff = cpu_fused.iter().zip(gpu_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    let mean_diff: f32 = cpu_fused.iter().zip(gpu_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / out_features as f32;

    eprintln!("Fused K2 [2048×7168]: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}");
    assert!(max_diff < 0.5, "Fused diverges at K2 dims: max_diff={max_diff} (threshold 0.5 for INT4)");
}

/// Verify columns 4096+ specifically contribute to the output.
/// Uses x=0 for cols 0..4095 and x=1.0 for cols 4096..7167.
/// If x_cache OOB reads returned 0, the output would be wrong.
#[test]
fn fused_k2_high_columns_contribute() {
    let mut metal = MetalDequantGemv::new().expect("Metal GPU required");

    let out_features = 64;  // small for speed
    let in_features = 7168;
    let group_size = 128;

    metal.init_scratch(in_features, out_features, 8, group_size);

    // Weights: all 1.0 (after quantization noise)
    let gate_w: Vec<f32> = vec![0.5; out_features * in_features];
    let up_w: Vec<f32> = vec![0.5; out_features * in_features];

    // x = 0 for low cols, 1.0 for high cols
    let mut x = vec![0.0f32; in_features];
    for i in 4096..7168 {
        x[i] = 1.0;
    }

    let gate_bytes = pack_test_weights(&gate_w, out_features, in_features, group_size);
    let up_bytes = pack_test_weights(&up_w, out_features, in_features, group_size);

    // CPU reference
    let gate_out = cpu_dequant_gemv(&gate_bytes, &x, out_features, in_features, group_size);
    let up_out = cpu_dequant_gemv(&up_bytes, &x, out_features, in_features, group_size);
    let cpu_fused: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    // GPU fused
    let packed_u32s = out_features * in_features / 8;
    let num_groups = out_features * (in_features / group_size);
    let matrix_bytes = packed_u32s * 4 + num_groups * 2 + num_groups * 2;

    let mut combined_mmap = Vec::new();
    combined_mmap.extend_from_slice(&gate_bytes);
    combined_mmap.extend_from_slice(&up_bytes);
    let mmap_buf = metal.wrap_mmap(&combined_mmap);

    let fused_op = FusedGateUpSiluOp {
        gate_packed_offset: 0,
        gate_scales_offset: packed_u32s * 4,
        gate_zeros_offset: packed_u32s * 4 + num_groups * 2,
        up_packed_offset: matrix_bytes,
        up_scales_offset: matrix_bytes + packed_u32s * 4,
        up_zeros_offset: matrix_bytes + packed_u32s * 4 + num_groups * 2,
        out_features, in_features, group_size,
    };

    let fused_result = metal.batch_fused_gate_up_silu_mmap(&mmap_buf, &[fused_op], &[&x]);
    let gpu_fused = &fused_result[0];

    // All CPU outputs should be non-zero (columns 4096+ contribute)
    let cpu_nonzero = cpu_fused.iter().all(|&v| v.abs() > 0.01);
    assert!(cpu_nonzero, "CPU reference should be non-zero when high columns have x=1.0");

    // GPU should also be non-zero (proves x_cache correctly loaded cols 4096+)
    let gpu_nonzero = gpu_fused.iter().all(|&v| v.abs() > 0.01);
    assert!(gpu_nonzero, "GPU fused output is zero — x_cache OOB for cols 4096+!");

    let max_diff = cpu_fused.iter().zip(gpu_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    eprintln!("Fused high-cols test: max_diff={max_diff:.6}, cpu[0]={:.4}, gpu[0]={:.4}",
        cpu_fused[0], gpu_fused[0]);
    assert!(max_diff < 0.5, "Fused diverges for high columns: max_diff={max_diff}");
}
