//! Convert DeepSeek-V2-Lite bf16 safetensors → rustane format.
//!
//! Produces:
//!   backbone.bin         -- embed, per-layer MLA + dense/shared weights, final norm, lm_head
//!   backbone_index.json  -- maps tensor names to {offset, shape, dtype}
//!   layer_XX_experts.bin -- per-layer routed expert weights (4-bit quantized)
//!
//! Key differences from Qwen3 converter:
//!   - MLA tensors: q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj → split W_UK/W_UV
//!   - Layer 0: dense FFN (gate/up/down_proj)
//!   - Layers 1-26: MoE with shared experts (shared_experts.{gate,up,down}_proj)
//!   - Router gate: mlp.gate.weight
//!   - bf16 input (V2-Lite)
//!
//! Usage:
//!   cargo run -p moe-infer --release --bin convert_deepseek -- \
//!     --model-dir weights/deepseek-v2-lite \
//!     --output-dir weights/rustane-deepseek-v2-lite

use anyhow::{Context, Result, bail};
use half::f16;
use safetensors::SafeTensors;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

// V2-Lite config (from config.json)
const HIDDEN: usize = 2048;
const VOCAB: usize = 102400;
const NUM_LAYERS: usize = 27;
const NUM_HEADS: usize = 16;
const NOPE_DIM: usize = 128;
const ROPE_DIM: usize = 64;
const V_HEAD_DIM: usize = 128;
const KV_LORA_RANK: usize = 512;
const DENSE_INTER: usize = 10944;
const MOE_INTER: usize = 1408;
const NUM_EXPERTS: usize = 64;
const GROUP_SIZE: usize = 128;
const FIRST_K_DENSE: usize = 1;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = String::new();
    let mut output_dir = String::new();
    let mut max_layers: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = args[i].clone(); }
            "--output-dir" => { i += 1; output_dir = args[i].clone(); }
            "--max-layers" => { i += 1; max_layers = Some(args[i].parse()?); }
            _ => bail!("unknown arg: {}", args[i]),
        }
        i += 1;
    }

    if model_dir.is_empty() || output_dir.is_empty() {
        eprintln!("Usage: convert_deepseek --model-dir <path> --output-dir <path> [--max-layers N]");
        std::process::exit(1);
    }

    let model_path = Path::new(&model_dir);
    let out_path = Path::new(&output_dir);
    fs::create_dir_all(out_path)?;

    let num_layers = max_layers.unwrap_or(NUM_LAYERS);
    eprintln!("Converting DeepSeek-V2-Lite: {num_layers} layers");
    eprintln!("  model_dir: {model_dir}");
    eprintln!("  output_dir: {output_dir}");

    // Load safetensors shards
    let shards = load_shards(model_path)?;
    eprintln!("Loaded {} shards, {} total tensors",
        shards.len(), shards.values().map(|s| s.len()).sum::<usize>());

    // Build backbone.bin + index
    let mut backbone = Vec::new();
    let mut index: HashMap<String, serde_json::Value> = HashMap::new();

    // 1. Embedding
    write_tensor_f16(&shards, "model.embed_tokens.weight", &mut backbone, &mut index)?;
    eprintln!("  embed_tokens: {} bytes", index["model.embed_tokens.weight"]["offset"]);

    // 2. Per-layer weights
    for layer in 0..num_layers {
        let prefix = format!("model.layers.{layer}");
        eprintln!("  Layer {layer}...");

        // Norms (f32)
        write_tensor_f32(&shards, &format!("{prefix}.input_layernorm.weight"), &mut backbone, &mut index)?;
        write_tensor_f32(&shards, &format!("{prefix}.post_attention_layernorm.weight"), &mut backbone, &mut index)?;

        // MLA attention
        write_tensor_f16(&shards, &format!("{prefix}.self_attn.q_proj.weight"), &mut backbone, &mut index)?;
        write_tensor_f16(&shards, &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"), &mut backbone, &mut index)?;
        write_tensor_f32(&shards, &format!("{prefix}.self_attn.kv_a_layernorm.weight"), &mut backbone, &mut index)?;
        write_tensor_f16(&shards, &format!("{prefix}.self_attn.o_proj.weight"), &mut backbone, &mut index)?;

        // Split kv_b_proj → W_UK + W_UV at conversion time
        split_kv_b_proj(&shards, &prefix, &mut backbone, &mut index)?;

        if layer < FIRST_K_DENSE {
            // Dense FFN (layer 0)
            write_tensor_f16(&shards, &format!("{prefix}.mlp.gate_proj.weight"), &mut backbone, &mut index)?;
            write_tensor_f16(&shards, &format!("{prefix}.mlp.up_proj.weight"), &mut backbone, &mut index)?;
            write_tensor_f16(&shards, &format!("{prefix}.mlp.down_proj.weight"), &mut backbone, &mut index)?;
        } else {
            // MoE: router gate
            write_tensor_f16(&shards, &format!("{prefix}.mlp.gate.weight"), &mut backbone, &mut index)?;

            // Shared experts
            write_tensor_f16(&shards, &format!("{prefix}.mlp.shared_experts.gate_proj.weight"), &mut backbone, &mut index)?;
            write_tensor_f16(&shards, &format!("{prefix}.mlp.shared_experts.up_proj.weight"), &mut backbone, &mut index)?;
            write_tensor_f16(&shards, &format!("{prefix}.mlp.shared_experts.down_proj.weight"), &mut backbone, &mut index)?;

            // Routed experts → quantize to 4-bit and write separate file
            convert_experts(&shards, layer, &prefix, out_path)?;
        }
    }

    // 3. Final norm + LM head
    write_tensor_f32(&shards, "model.norm.weight", &mut backbone, &mut index)?;
    write_tensor_f16(&shards, "lm_head.weight", &mut backbone, &mut index)?;

    // Write backbone
    let backbone_path = out_path.join("backbone.bin");
    fs::write(&backbone_path, &backbone)?;
    eprintln!("Wrote backbone.bin: {:.1} MB", backbone.len() as f64 / 1e6);

    let index_path = out_path.join("backbone_index.json");
    let index_json = serde_json::to_string_pretty(&index)?;
    fs::write(&index_path, &index_json)?;
    eprintln!("Wrote backbone_index.json: {} tensors", index.len());

    eprintln!("Done!");
    Ok(())
}

/// Load all safetensor shards from model directory.
fn load_shards(model_dir: &Path) -> Result<HashMap<String, Vec<u8>>> {
    let mut shards = HashMap::new();
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            let data = fs::read(entry.path())
                .with_context(|| format!("reading {}", entry.path().display()))?;
            shards.insert(name, data);
        }
    }
    Ok(shards)
}

/// Find a tensor across all shards.
fn find_tensor<'a>(shards: &'a HashMap<String, Vec<u8>>, name: &str) -> Result<(safetensors::tensor::TensorView<'a>, &'a str)> {
    for (shard_name, data) in shards {
        let st = SafeTensors::deserialize(data)
            .with_context(|| format!("parsing shard {shard_name}"))?;
        if let Ok(tensor) = st.tensor(name) {
            // We need to return owned data, but SafeTensors borrows from data
            // Instead, return a reference with the shard name
            return Ok((tensor, shard_name.as_str()));
        }
    }
    bail!("tensor not found: {name}")
}

/// Get tensor data as f32 values.
fn tensor_to_f32(shards: &HashMap<String, Vec<u8>>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    for (_shard_name, data) in shards {
        let st = SafeTensors::deserialize(data)?;
        if let Ok(tensor) = st.tensor(name) {
            let shape: Vec<usize> = tensor.shape().to_vec();
            let bytes = tensor.data();
            let values: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::BF16 => {
                    bytes.chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    bytes.chunks_exact(2)
                        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect()
                }
                safetensors::Dtype::F32 => {
                    bytes.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
                dt => bail!("unsupported dtype {dt:?} for {name}"),
            };
            return Ok((values, shape));
        }
    }
    bail!("tensor not found: {name}")
}

/// Write a tensor as f16 to backbone, recording in index.
fn write_tensor_f16(
    shards: &HashMap<String, Vec<u8>>,
    name: &str,
    backbone: &mut Vec<u8>,
    index: &mut HashMap<String, serde_json::Value>,
) -> Result<()> {
    let (values, shape) = tensor_to_f32(shards, name)?;
    let offset = backbone.len();
    // Convert to f16 and write
    for &v in &values {
        let h = f16::from_f32(v);
        backbone.extend_from_slice(&h.to_le_bytes());
    }
    index.insert(name.to_string(), json!({
        "offset": offset,
        "dtype": "f16",
        "shape": shape,
    }));
    Ok(())
}

/// Write a tensor as f32 to backbone, recording in index.
fn write_tensor_f32(
    shards: &HashMap<String, Vec<u8>>,
    name: &str,
    backbone: &mut Vec<u8>,
    index: &mut HashMap<String, serde_json::Value>,
) -> Result<()> {
    let (values, shape) = tensor_to_f32(shards, name)?;
    let offset = backbone.len();
    for &v in &values {
        backbone.extend_from_slice(&v.to_le_bytes());
    }
    index.insert(name.to_string(), json!({
        "offset": offset,
        "dtype": "f32",
        "shape": shape,
    }));
    Ok(())
}

/// Split kv_b_proj into W_UK and W_UV and write to backbone.
fn split_kv_b_proj(
    shards: &HashMap<String, Vec<u8>>,
    prefix: &str,
    backbone: &mut Vec<u8>,
    index: &mut HashMap<String, serde_json::Value>,
) -> Result<()> {
    let name = format!("{prefix}.self_attn.kv_b_proj.weight");
    let (values, shape) = tensor_to_f32(shards, &name)?;

    // kv_b_proj: [num_heads * (nope + v), kv_lora_rank] row-major
    assert_eq!(shape.len(), 2);
    let out_dim = shape[0];
    let in_dim = shape[1];
    assert_eq!(out_dim, NUM_HEADS * (NOPE_DIM + V_HEAD_DIM));
    assert_eq!(in_dim, KV_LORA_RANK);

    // Reshape to [num_heads, nope + v, kv_lora_rank]
    // Split into W_UK [num_heads, nope, kv_lora_rank] and W_UV [num_heads, v, kv_lora_rank]
    let mut w_uk = Vec::with_capacity(NUM_HEADS * NOPE_DIM * KV_LORA_RANK);
    let mut w_uv = Vec::with_capacity(NUM_HEADS * V_HEAD_DIM * KV_LORA_RANK);

    for h in 0..NUM_HEADS {
        let head_base = h * (NOPE_DIM + V_HEAD_DIM) * KV_LORA_RANK;
        // W_UK: rows 0..NOPE_DIM
        for row in 0..NOPE_DIM {
            let src = head_base + row * KV_LORA_RANK;
            for col in 0..KV_LORA_RANK {
                w_uk.push(f16::from_f32(values[src + col]));
            }
        }
        // W_UV: rows NOPE_DIM..NOPE_DIM+V_HEAD_DIM
        for row in 0..V_HEAD_DIM {
            let src = head_base + (NOPE_DIM + row) * KV_LORA_RANK;
            for col in 0..KV_LORA_RANK {
                w_uv.push(f16::from_f32(values[src + col]));
            }
        }
    }

    // Write W_UK
    let uk_name = format!("{prefix}.self_attn.w_uk");
    let offset = backbone.len();
    for &v in &w_uk {
        backbone.extend_from_slice(&v.to_le_bytes());
    }
    index.insert(uk_name, json!({
        "offset": offset,
        "dtype": "f16",
        "shape": [NUM_HEADS, NOPE_DIM, KV_LORA_RANK],
    }));

    // Write W_UV
    let uv_name = format!("{prefix}.self_attn.w_uv");
    let offset = backbone.len();
    for &v in &w_uv {
        backbone.extend_from_slice(&v.to_le_bytes());
    }
    index.insert(uv_name, json!({
        "offset": offset,
        "dtype": "f16",
        "shape": [NUM_HEADS, V_HEAD_DIM, KV_LORA_RANK],
    }));

    Ok(())
}

/// Convert and quantize routed experts for one MoE layer.
fn convert_experts(
    shards: &HashMap<String, Vec<u8>>,
    layer: usize,
    prefix: &str,
    out_dir: &Path,
) -> Result<()> {
    let expert_path = out_dir.join(format!("layer_{layer:02}_experts.bin"));
    let mut file = fs::File::create(&expert_path)?;

    for expert in 0..NUM_EXPERTS {
        let ep = format!("{prefix}.mlp.experts.{expert}");
        let gate = tensor_to_f32(shards, &format!("{ep}.gate_proj.weight"))?.0;
        let up = tensor_to_f32(shards, &format!("{ep}.up_proj.weight"))?.0;
        let down = tensor_to_f32(shards, &format!("{ep}.down_proj.weight"))?.0;

        // Quantize each matrix to 4-bit
        let gate_q = quantize_4bit(&gate, MOE_INTER, HIDDEN, GROUP_SIZE);
        let up_q = quantize_4bit(&up, MOE_INTER, HIDDEN, GROUP_SIZE);
        let down_q = quantize_4bit(&down, HIDDEN, MOE_INTER, GROUP_SIZE);

        file.write_all(&gate_q)?;
        file.write_all(&up_q)?;
        file.write_all(&down_q)?;
    }

    let size = file.metadata()?.len();
    eprintln!("    layer_{layer:02}_experts.bin: {:.1} MB", size as f64 / 1e6);
    Ok(())
}

/// Quantize a matrix to 4-bit with group_size groups.
/// Returns packed bytes: [packed_u32s | scales_f16 | zeros_f16]
fn quantize_4bit(values: &[f32], out_dim: usize, in_dim: usize, group_size: usize) -> Vec<u8> {
    assert_eq!(values.len(), out_dim * in_dim);
    assert_eq!(in_dim % group_size, 0);

    let num_groups_per_row = in_dim / group_size;
    let total_groups = out_dim * num_groups_per_row;
    let packed_u32s = out_dim * in_dim / 8;

    let mut packed = vec![0u32; packed_u32s];
    let mut scales = vec![f16::ZERO; total_groups];
    let mut zeros = vec![f16::ZERO; total_groups];

    for row in 0..out_dim {
        for g in 0..num_groups_per_row {
            let start = row * in_dim + g * group_size;
            let group = &values[start..start + group_size];

            let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };

            let group_idx = row * num_groups_per_row + g;
            scales[group_idx] = f16::from_f32(scale);
            zeros[group_idx] = f16::from_f32(min);

            for k in 0..group_size {
                let col = g * group_size + k;
                let val = values[row * in_dim + col];
                let q = ((val - min) / scale).round().clamp(0.0, 15.0) as u32;
                let flat_idx = row * in_dim + col;
                let word_idx = flat_idx / 8;
                let nibble_pos = flat_idx % 8;
                packed[word_idx] |= q << (nibble_pos * 4);
            }
        }
    }

    // Serialize: packed + scales + zeros
    let mut out = Vec::with_capacity(packed_u32s * 4 + total_groups * 4);
    for &w in &packed {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for &s in &scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &z in &zeros {
        out.extend_from_slice(&z.to_le_bytes());
    }
    out
}
