//! Real-weight validation: load actual Qwen3-MoE-30B weights from safetensors
//! and validate our quantization + Metal dequant pipeline on real data.
//!
//! This is the most significant integration test — proves our infrastructure
//! works on trained weights, not just synthetic data.
//!
//! Run: cargo test -p moe-infer --test real_weights_validation --release -- --ignored --nocapture

use std::path::Path;

fn shard_path() -> std::path::PathBuf {
    // Try relative to workspace root (cargo test runs from workspace root)
    let relative = Path::new("weights/qwen3-moe-30b-ref/model-00001-of-00016.safetensors");
    if relative.exists() { return relative.to_path_buf(); }
    // Try absolute path
    let abs = Path::new("/Users/dan/Dev/rustane/weights/qwen3-moe-30b-ref/model-00001-of-00016.safetensors");
    abs.to_path_buf()
}

fn shard_exists() -> bool {
    shard_path().exists()
}

/// Discover what tensors are in the shard and print summary.
#[test]
#[ignore = "requires downloaded weights"]
fn inspect_shard_contents() {
    if !shard_exists() {
        println!("SKIP: shard not found. Run: hf download Qwen/Qwen3-30B-A3B model-00001-of-00016.safetensors --local-dir weights/qwen3-moe-30b-ref");
        return;
    }

    let data = std::fs::read(shard_path()).expect("read shard");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse safetensors");

    let mut expert_tensors = Vec::new();
    let mut attn_tensors = Vec::new();
    let mut other_tensors = Vec::new();
    let mut total_params: u64 = 0;

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let params: u64 = shape.iter().product::<usize>() as u64;
        total_params += params;

        if name.contains("experts") {
            expert_tensors.push((name.clone(), shape, params));
        } else if name.contains("self_attn") {
            attn_tensors.push((name.clone(), shape, params));
        } else {
            other_tensors.push((name.clone(), shape, params));
        }
    }

    println!("\n=== Qwen3-MoE-30B Shard 1 Contents ===");
    println!("Total tensors: {}", tensors.len());
    println!("Total params: {total_params} ({:.1}M)", total_params as f64 / 1e6);
    println!("\nExpert tensors: {}", expert_tensors.len());
    for (name, shape, params) in expert_tensors.iter().take(10) {
        println!("  {name}: {shape:?} ({:.1}M)", *params as f64 / 1e6);
    }
    if expert_tensors.len() > 10 {
        println!("  ... and {} more", expert_tensors.len() - 10);
    }

    println!("\nAttention tensors: {}", attn_tensors.len());
    for (name, shape, params) in attn_tensors.iter().take(5) {
        println!("  {name}: {shape:?} ({:.1}M)", *params as f64 / 1e6);
    }

    println!("\nOther tensors: {}", other_tensors.len());
    for (name, shape, params) in other_tensors.iter().take(5) {
        println!("  {name}: {shape:?} ({:.1}M)", *params as f64 / 1e6);
    }
}

/// Load real expert weights, quantize to 4-bit, verify dequant error.
#[test]
#[ignore = "requires downloaded weights"]
fn quantize_real_expert_weights_4bit() {
    if !shard_exists() {
        println!("SKIP: weights not found");
        return;
    }

    let data = std::fs::read(shard_path()).expect("read shard");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse");

    // Find expert gate/up/down tensors
    let mut expert_names: Vec<String> = Vec::new();
    for (name, _) in tensors.tensors() {
        if name.contains("experts") && (name.contains("gate_proj") || name.contains("up_proj") || name.contains("down_proj")) {
            expert_names.push(name.clone());
        }
    }
    expert_names.sort();

    if expert_names.is_empty() {
        println!("No expert tensors found in shard 1 (may be in later shards)");
        // Try attention or other weights instead
        validate_any_weight_tensor(&tensors);
        return;
    }

    println!("\nFound {} expert weight tensors", expert_names.len());

    // Test first few experts
    let mut total_max_err = 0.0f32;
    let mut total_mean_err = 0.0f64;
    let mut count = 0;

    for name in expert_names.iter().take(6) {
        let view = tensors.tensor(name).unwrap();
        let shape = view.shape();
        let weights = tensor_to_f32(&view);

        let (rows, cols) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            continue;
        };

        // Need cols to be multiple of 8 for 4-bit packing
        if cols % 8 != 0 || rows == 0 { continue; }
        // Need cols to be multiple of 128 for group_size
        let group_size = if cols % 128 == 0 { 128 } else if cols % 64 == 0 { 64 } else if cols % 32 == 0 { 32 } else { continue };

        let packed = quantize::PackedWeights4Bit::pack(&weights, rows, cols, group_size);
        let dequant = packed.dequant_to_f32();

        let max_err = weights.iter().zip(dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_err = weights.iter().zip(dequant.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum::<f64>() / weights.len() as f64;

        // Weight statistics
        let w_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let w_mean = weights.iter().sum::<f32>() / weights.len() as f32;

        println!("  {name}: [{rows}x{cols}] gs={group_size}");
        println!("    range: [{w_min:.4}, {w_max:.4}], mean: {w_mean:.6}");
        println!("    4-bit error: max={max_err:.6}, mean={mean_err:.8}");

        total_max_err = total_max_err.max(max_err);
        total_mean_err += mean_err;
        count += 1;
    }

    if count > 0 {
        println!("\n  Overall: max_err={total_max_err:.6}, avg_mean_err={:.8}",
            total_mean_err / count as f64);

        // Real weights should quantize well — max error typically < 0.1 for 4-bit gs=128
        assert!(total_max_err < 0.2,
            "4-bit quantization error too large: {total_max_err} (expected < 0.2)");
    }
}

/// Load real expert weights, quantize to 4-bit, run Metal dequant GEMV.
#[test]
#[ignore = "requires downloaded weights + Metal GPU"]
fn metal_dequant_real_weights() {
    if !shard_exists() {
        println!("SKIP: weights not found");
        return;
    }

    let metal = moe_kernels::MetalDequantGemv::new().expect("Metal device required");

    let data = std::fs::read(shard_path()).expect("read shard");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse");

    // Find a suitable weight matrix
    let (name, weights, rows, cols) = find_suitable_tensor(&tensors);
    println!("\nMetal dequant GEMV on real weights: {name} [{rows}x{cols}]");

    let group_size = if cols % 128 == 0 { 128 } else { 64 };
    let packed = quantize::PackedWeights4Bit::pack(&weights, rows, cols, group_size);

    // Create random input
    let x: Vec<f32> = (0..cols).map(|i| {
        let s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.1
    }).collect();

    // CPU reference
    let y_cpu = packed.gemv(&x);

    // Metal GPU
    let y_gpu = metal.gemv(&packed, &x);

    let max_diff = y_cpu.iter().zip(y_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("  CPU vs Metal max diff: {max_diff:.6}");
    println!("  CPU output sample: {:?}", &y_cpu[..5.min(y_cpu.len())]);
    println!("  Metal output sample: {:?}", &y_gpu[..5.min(y_gpu.len())]);

    // For large matrices with real weights, allow more tolerance due to f32 vs f32 accumulation differences
    assert!(max_diff < 0.1,
        "Metal dequant GEMV error too large on real weights: {max_diff}");
}

/// Quantize to 2-bit and compare error against 4-bit.
#[test]
#[ignore = "requires downloaded weights"]
fn quantize_real_weights_2bit_vs_4bit() {
    if !shard_exists() {
        println!("SKIP: weights not found");
        return;
    }

    let data = std::fs::read(shard_path()).expect("read shard");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse");

    let (name, weights, rows, cols) = find_suitable_tensor(&tensors);
    println!("\n2-bit vs 4-bit comparison on: {name} [{rows}x{cols}]");

    let group_size = if cols % 128 == 0 { 128 } else { 64 };

    // 4-bit
    let packed4 = quantize::PackedWeights4Bit::pack(&weights, rows, cols, group_size);
    let dequant4 = packed4.dequant_to_f32();
    let err4_max = weights.iter().zip(dequant4.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let err4_mean = weights.iter().zip(dequant4.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / weights.len() as f64;

    // 2-bit (need cols % 16 == 0)
    if cols % 16 != 0 {
        println!("  Skipping 2-bit: cols={cols} not multiple of 16");
        return;
    }
    let packed2 = quantize::PackedWeights2Bit::pack(&weights, rows, cols, group_size);
    let dequant2 = packed2.dequant_to_f32();
    let err2_max = weights.iter().zip(dequant2.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let err2_mean = weights.iter().zip(dequant2.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / weights.len() as f64;

    // Memory savings
    let bytes_4bit = packed4.data.len() * 4 + packed4.scales.len() * 2 + packed4.zeros.len() * 2;
    let bytes_2bit = packed2.data.len() * 4 + packed2.palettes.len() * 2;
    let bytes_fp32 = weights.len() * 4;

    println!("  4-bit: max_err={err4_max:.6}, mean_err={err4_mean:.8}, size={:.1} MB",
        bytes_4bit as f64 / 1e6);
    println!("  2-bit: max_err={err2_max:.6}, mean_err={err2_mean:.8}, size={:.1} MB",
        bytes_2bit as f64 / 1e6);
    println!("  fp32:  size={:.1} MB", bytes_fp32 as f64 / 1e6);
    println!("  4-bit compression: {:.1}x", bytes_fp32 as f64 / bytes_4bit as f64);
    println!("  2-bit compression: {:.1}x", bytes_fp32 as f64 / bytes_2bit as f64);

    assert!(err4_max < err2_max, "4-bit should have less error than 2-bit");
}

// --- Helpers ---

fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    let dtype = view.dtype();
    let data = view.data();
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            data.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        other => panic!("Unsupported dtype: {other:?}"),
    }
}

fn find_suitable_tensor(tensors: &safetensors::SafeTensors<'_>) -> (String, Vec<f32>, usize, usize) {
    // Prefer expert weights, fallback to any 2D matrix
    for (name, view) in tensors.tensors() {
        let shape = view.shape();
        if shape.len() == 2 && shape[0] >= 64 && shape[1] >= 64 && shape[1] % 8 == 0 {
            let weights = tensor_to_f32(&view);
            return (name.clone(), weights, shape[0], shape[1]);
        }
    }
    panic!("No suitable 2D tensor found in shard");
}

fn validate_any_weight_tensor(tensors: &safetensors::SafeTensors<'_>) {
    let (name, weights, rows, cols) = find_suitable_tensor(tensors);
    println!("\nValidating on: {name} [{rows}x{cols}]");

    let group_size = if cols % 128 == 0 { 128 } else if cols % 64 == 0 { 64 } else { 32 };
    let packed = quantize::PackedWeights4Bit::pack(&weights, rows, cols, group_size);
    let dequant = packed.dequant_to_f32();

    let max_err = weights.iter().zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_err = weights.iter().zip(dequant.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>() / weights.len() as f64;

    println!("  4-bit error: max={max_err:.6}, mean={mean_err:.8}");
    assert!(max_err < 0.2, "error too large: {max_err}");
}
