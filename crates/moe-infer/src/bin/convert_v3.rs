//! Convert DeepSeek-V3 FP8 safetensors → rustane format.
//!
//! Produces:
//!   backbone.bin         -- embed, per-layer MLA + dense/shared weights, norms, lm_head
//!   backbone_index.json  -- maps tensor names to {offset, shape, dtype}
//!   layer_XX_experts.bin -- per-layer routed expert weights (4-bit quantized)
//!
//! Uses model.safetensors.index.json for O(1) shard lookup (not scanning 163 files).
//!
//! Usage:
//!   cargo run -p moe-infer --release --bin convert_v3 -- \
//!     --model-dir weights/deepseek-v3 \
//!     --output-dir weights/rustane-v3 \
//!     [--max-layers N]

use anyhow::{Context, Result, bail};
use half::f16;
use rayon::prelude::*;
use safetensors::SafeTensors;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use moe_infer::fp8::{fp8_lut, FP8_BLOCK_SIZE};

// V3 config
const HIDDEN: usize = 7168;
const NUM_LAYERS: usize = 61;
const NUM_HEADS: usize = 128;
const NOPE_DIM: usize = 128;
const V_HEAD_DIM: usize = 128;
const KV_LORA_RANK: usize = 512;
const MOE_INTER: usize = 2048;
const NUM_EXPERTS: usize = 256;
const GROUP_SIZE: usize = 128;
const FIRST_K_DENSE: usize = 3;

/// Maps tensor names to their shard file path for O(1) lookup.
struct ShardIndex {
    /// tensor_name → shard file path
    tensor_to_shard: HashMap<String, PathBuf>,
    /// Cached shard data: shard_path → bytes (keeps last N shards in memory)
    shard_cache: HashMap<PathBuf, Vec<u8>>,
    /// Max shards to keep cached
    max_cached: usize,
}

impl ShardIndex {
    /// Build index from model.safetensors.index.json
    fn build(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let mut tensor_to_shard = HashMap::new();

        if index_path.exists() {
            let content = fs::read_to_string(&index_path)
                .with_context(|| format!("reading {}", index_path.display()))?;
            let parsed: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(weight_map) = parsed.get("weight_map").and_then(|v| v.as_object()) {
                for (tensor_name, shard_name) in weight_map {
                    if let Some(shard) = shard_name.as_str() {
                        tensor_to_shard.insert(
                            tensor_name.clone(),
                            model_dir.join(shard),
                        );
                    }
                }
            }
            eprintln!("Shard index: {} tensor→shard mappings", tensor_to_shard.len());
        } else {
            // Fallback: scan all shards for tensor names
            eprintln!("WARNING: no model.safetensors.index.json — falling back to shard scanning");
            for entry in fs::read_dir(model_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "safetensors") {
                    let data = fs::read(&path)?;
                    let st = SafeTensors::deserialize(&data)?;
                    for name in st.names() {
                        tensor_to_shard.insert(name.to_string(), path.clone());
                    }
                }
            }
        }

        Ok(Self {
            tensor_to_shard,
            shard_cache: HashMap::new(),
            max_cached: 4,
        })
    }

    /// Load a tensor's raw bytes. Uses cached shard data when available.
    fn load_tensor_raw(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, safetensors::Dtype)> {
        let shard_path = self.tensor_to_shard.get(name)
            .with_context(|| format!("tensor not in index: {name}"))?
            .clone();

        // Load shard if not cached
        if !self.shard_cache.contains_key(&shard_path) {
            // Evict oldest if at capacity
            if self.shard_cache.len() >= self.max_cached {
                // Remove a shard that isn't the one we're about to load
                let to_remove = self.shard_cache.keys()
                    .find(|k| *k != &shard_path)
                    .cloned();
                if let Some(key) = to_remove {
                    self.shard_cache.remove(&key);
                }
            }
            let data = fs::read(&shard_path)
                .with_context(|| format!("reading shard {}", shard_path.display()))?;
            self.shard_cache.insert(shard_path.clone(), data);
        }

        let data = &self.shard_cache[&shard_path];
        let st = SafeTensors::deserialize(data)
            .with_context(|| format!("parsing shard {}", shard_path.display()))?;
        let tensor = st.tensor(name)
            .with_context(|| format!("tensor {name} not in shard {}", shard_path.display()))?;

        Ok((tensor.data().to_vec(), tensor.shape().to_vec(), tensor.dtype()))
    }

    /// Load tensor and convert to f32.
    fn tensor_to_f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (bytes, shape, dtype) = self.load_tensor_raw(name)?;
        let values: Vec<f32> = match dtype {
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
        Ok((values, shape))
    }

    /// Load FP8 tensor with block-wise scale, dequantize to f32.
    fn fp8_tensor_to_f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (bytes, shape, dtype) = self.load_tensor_raw(name)?;

        match dtype {
            safetensors::Dtype::F8_E4M3 | safetensors::Dtype::U8 => {}
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32 => {
                return self.tensor_to_f32(name);
            }
            dt => bail!("unexpected dtype {dt:?} for FP8 tensor {name}"),
        }

        let scale_name = format!("{name}_scale_inv");
        let (scale_values, _) = self.tensor_to_f32(&scale_name)
            .with_context(|| format!("loading scale for {name}"))?;

        assert_eq!(shape.len(), 2, "FP8 tensor must be 2D: {name}");
        let m = shape[0];
        let k = shape[1];
        assert_eq!(bytes.len(), m * k);

        let lut = fp8_lut();
        let num_col_blocks = (k + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        let mut out = vec![0.0f32; m * k];

        for i in 0..m {
            let row_block = i / FP8_BLOCK_SIZE;
            let row_offset = i * k;
            for j in 0..k {
                let col_block = j / FP8_BLOCK_SIZE;
                let scale = scale_values[row_block * num_col_blocks + col_block];
                out[row_offset + j] = lut[bytes[row_offset + j] as usize] * scale;
            }
        }

        Ok((out, shape))
    }

    /// Get the shard path for a tensor name.
    fn shard_for(&self, name: &str) -> Option<&PathBuf> {
        self.tensor_to_shard.get(name)
    }
}

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
        eprintln!("Usage: convert_v3 --model-dir <path> --output-dir <path> [--max-layers N]");
        std::process::exit(1);
    }

    let model_path = Path::new(&model_dir);
    let out_path = Path::new(&output_dir);
    fs::create_dir_all(out_path)?;

    let num_layers = max_layers.unwrap_or(NUM_LAYERS);
    eprintln!("Converting DeepSeek-V3: {num_layers} layers");
    eprintln!("  model_dir: {model_dir}");
    eprintln!("  output_dir: {output_dir}");

    // Build shard index for O(1) tensor→shard lookup
    let mut si = ShardIndex::build(model_path)?;

    // Build backbone.bin + index
    let mut backbone = Vec::new();
    let mut index: HashMap<String, serde_json::Value> = HashMap::new();

    // 1. Embedding
    write_tensor_f16(&mut si, "model.embed_tokens.weight", &mut backbone, &mut index)?;
    eprintln!("  embed_tokens: {:.1} MB", backbone.len() as f64 / 1e6);

    // 2. Per-layer weights
    for layer in 0..num_layers {
        let prefix = format!("model.layers.{layer}");
        let t = std::time::Instant::now();

        // Norms
        write_tensor_f32(&mut si, &format!("{prefix}.input_layernorm.weight"), &mut backbone, &mut index)?;
        write_tensor_f32(&mut si, &format!("{prefix}.post_attention_layernorm.weight"), &mut backbone, &mut index)?;

        // MLA attention (FP8 → f16)
        write_fp8_f16(&mut si, &format!("{prefix}.self_attn.q_a_proj.weight"), &mut backbone, &mut index)?;
        write_tensor_f32(&mut si, &format!("{prefix}.self_attn.q_a_layernorm.weight"), &mut backbone, &mut index)?;
        write_fp8_f16(&mut si, &format!("{prefix}.self_attn.q_b_proj.weight"), &mut backbone, &mut index)?;
        write_fp8_f16(&mut si, &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"), &mut backbone, &mut index)?;
        write_tensor_f32(&mut si, &format!("{prefix}.self_attn.kv_a_layernorm.weight"), &mut backbone, &mut index)?;
        write_fp8_f16(&mut si, &format!("{prefix}.self_attn.o_proj.weight"), &mut backbone, &mut index)?;

        // Split kv_b_proj → W_UK + W_UV
        split_kv_b_proj(&mut si, &prefix, &mut backbone, &mut index)?;

        if layer < FIRST_K_DENSE {
            // Dense FFN layers 0-2
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.gate_proj.weight"), &mut backbone, &mut index)?;
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.up_proj.weight"), &mut backbone, &mut index)?;
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.down_proj.weight"), &mut backbone, &mut index)?;
        } else {
            // MoE layer
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.gate.weight"), &mut backbone, &mut index)?;
            write_tensor_f32(&mut si, &format!("{prefix}.mlp.gate.e_score_correction_bias"), &mut backbone, &mut index)?;
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.shared_experts.gate_proj.weight"), &mut backbone, &mut index)?;
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.shared_experts.up_proj.weight"), &mut backbone, &mut index)?;
            write_fp8_f16(&mut si, &format!("{prefix}.mlp.shared_experts.down_proj.weight"), &mut backbone, &mut index)?;

            // Routed experts — group by shard for efficient I/O
            convert_experts_indexed(&mut si, layer, &prefix, out_path)?;
        }

        eprintln!("  Layer {layer} done ({:.1}s)", t.elapsed().as_secs_f64());
    }

    // 3. Final norm + LM head
    write_tensor_f32(&mut si, "model.norm.weight", &mut backbone, &mut index)?;
    write_tensor_f16(&mut si, "lm_head.weight", &mut backbone, &mut index)?;

    // Write backbone
    let backbone_path = out_path.join("backbone.bin");
    fs::write(&backbone_path, &backbone)?;
    eprintln!("Wrote backbone.bin: {:.1} GB", backbone.len() as f64 / 1e9);

    let index_path = out_path.join("backbone_index.json");
    let index_json = serde_json::to_string_pretty(&index)?;
    fs::write(&index_path, &index_json)?;
    eprintln!("Wrote backbone_index.json: {} tensors", index.len());

    eprintln!("Done!");
    Ok(())
}

fn write_tensor_f16(si: &mut ShardIndex, name: &str, bb: &mut Vec<u8>, idx: &mut HashMap<String, serde_json::Value>) -> Result<()> {
    let (values, shape) = si.tensor_to_f32(name)?;
    let offset = bb.len();
    for &v in &values { bb.extend_from_slice(&f16::from_f32(v).to_le_bytes()); }
    idx.insert(name.to_string(), json!({"offset": offset, "dtype": "f16", "shape": shape}));
    Ok(())
}

fn write_tensor_f32(si: &mut ShardIndex, name: &str, bb: &mut Vec<u8>, idx: &mut HashMap<String, serde_json::Value>) -> Result<()> {
    let (values, shape) = si.tensor_to_f32(name)?;
    let offset = bb.len();
    for &v in &values { bb.extend_from_slice(&v.to_le_bytes()); }
    idx.insert(name.to_string(), json!({"offset": offset, "dtype": "f32", "shape": shape}));
    Ok(())
}

fn write_fp8_f16(si: &mut ShardIndex, name: &str, bb: &mut Vec<u8>, idx: &mut HashMap<String, serde_json::Value>) -> Result<()> {
    let (values, shape) = si.fp8_tensor_to_f32(name)?;
    let offset = bb.len();
    for &v in &values { bb.extend_from_slice(&f16::from_f32(v).to_le_bytes()); }
    idx.insert(name.to_string(), json!({"offset": offset, "dtype": "f16", "shape": shape}));
    Ok(())
}

fn split_kv_b_proj(si: &mut ShardIndex, prefix: &str, bb: &mut Vec<u8>, idx: &mut HashMap<String, serde_json::Value>) -> Result<()> {
    let name = format!("{prefix}.self_attn.kv_b_proj.weight");
    let (values, shape) = si.fp8_tensor_to_f32(&name)?;

    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], NUM_HEADS * (NOPE_DIM + V_HEAD_DIM));
    assert_eq!(shape[1], KV_LORA_RANK);

    let mut w_uk = Vec::with_capacity(NUM_HEADS * NOPE_DIM * KV_LORA_RANK);
    let mut w_uv = Vec::with_capacity(NUM_HEADS * V_HEAD_DIM * KV_LORA_RANK);

    for h in 0..NUM_HEADS {
        let head_base = h * (NOPE_DIM + V_HEAD_DIM) * KV_LORA_RANK;
        for row in 0..NOPE_DIM {
            let src = head_base + row * KV_LORA_RANK;
            for col in 0..KV_LORA_RANK { w_uk.push(f16::from_f32(values[src + col])); }
        }
        for row in 0..V_HEAD_DIM {
            let src = head_base + (NOPE_DIM + row) * KV_LORA_RANK;
            for col in 0..KV_LORA_RANK { w_uv.push(f16::from_f32(values[src + col])); }
        }
    }

    let offset = bb.len();
    for &v in &w_uk { bb.extend_from_slice(&v.to_le_bytes()); }
    idx.insert(format!("{prefix}.self_attn.w_uk"), json!({"offset": offset, "dtype": "f16", "shape": [NUM_HEADS, NOPE_DIM, KV_LORA_RANK]}));

    let offset = bb.len();
    for &v in &w_uv { bb.extend_from_slice(&v.to_le_bytes()); }
    idx.insert(format!("{prefix}.self_attn.w_uv"), json!({"offset": offset, "dtype": "f16", "shape": [NUM_HEADS, V_HEAD_DIM, KV_LORA_RANK]}));

    Ok(())
}

/// Convert experts with indexed shard lookup.
/// Groups all 6 tensor names (gate/up/down × weight/scale) by shard,
/// loads each shard once, extracts all tensors from it, then quantizes.
fn convert_experts_indexed(si: &mut ShardIndex, layer: usize, prefix: &str, out_dir: &Path) -> Result<()> {
    let t = std::time::Instant::now();

    // Collect ALL tensor names for all experts (6 per expert: weight + scale for gate/up/down)
    let mut shard_to_tensors: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for e in 0..NUM_EXPERTS {
        let ep = format!("{prefix}.mlp.experts.{e}");
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            let w_name = format!("{ep}.{proj}.weight");
            let s_name = format!("{ep}.{proj}.weight_scale_inv");
            for name in [&w_name, &s_name] {
                if let Some(shard) = si.shard_for(name) {
                    shard_to_tensors.entry(shard.clone()).or_default().push(name.clone());
                }
            }
        }
    }

    let num_shards = shard_to_tensors.len();
    eprintln!("    layer {layer}: experts spread across {num_shards} shards");

    // Load each shard once, extract all tensors we need from it
    // Store raw tensor data keyed by tensor name
    let mut tensor_data: HashMap<String, (Vec<u8>, Vec<usize>, safetensors::Dtype)> = HashMap::new();

    for (shard_path, tensor_names) in &shard_to_tensors {
        let shard_bytes = fs::read(shard_path)
            .with_context(|| format!("reading {}", shard_path.display()))?;
        let st = SafeTensors::deserialize(&shard_bytes)
            .with_context(|| format!("parsing {}", shard_path.display()))?;

        for name in tensor_names {
            if let Ok(tensor) = st.tensor(name) {
                tensor_data.insert(
                    name.clone(),
                    (tensor.data().to_vec(), tensor.shape().to_vec(), tensor.dtype()),
                );
            }
        }
    }

    eprintln!("    loaded {} tensors from {num_shards} shards", tensor_data.len());

    // Now convert each expert in parallel — all data is in memory
    let lut = fp8_lut();
    let results: Vec<Result<Vec<u8>>> = (0..NUM_EXPERTS).into_par_iter().map(|e| {
        let ep = format!("{prefix}.mlp.experts.{e}");

        let gate = dequant_from_cache(&tensor_data, &format!("{ep}.gate_proj.weight"), lut)?;
        let up = dequant_from_cache(&tensor_data, &format!("{ep}.up_proj.weight"), lut)?;
        let down = dequant_from_cache(&tensor_data, &format!("{ep}.down_proj.weight"), lut)?;

        let gate_q = quantize_4bit(&gate, MOE_INTER, HIDDEN, GROUP_SIZE);
        let up_q = quantize_4bit(&up, MOE_INTER, HIDDEN, GROUP_SIZE);
        let down_q = quantize_4bit(&down, HIDDEN, MOE_INTER, GROUP_SIZE);

        let mut buf = Vec::with_capacity(gate_q.len() + up_q.len() + down_q.len());
        buf.extend_from_slice(&gate_q);
        buf.extend_from_slice(&up_q);
        buf.extend_from_slice(&down_q);
        Ok(buf)
    }).collect();

    // Write in order
    let expert_path = out_dir.join(format!("layer_{layer:02}_experts.bin"));
    let mut file = fs::File::create(&expert_path)?;
    for (e, result) in results.into_iter().enumerate() {
        let buf = result.with_context(|| format!("expert {e} layer {layer}"))?;
        file.write_all(&buf)?;
    }

    // Drop tensor_data to free RAM before next layer
    drop(tensor_data);

    let size = file.metadata()?.len();
    eprintln!("    layer_{layer:02}_experts.bin: {:.1} MB ({:.1}s, {} experts)",
        size as f64 / 1e6, t.elapsed().as_secs_f64(), NUM_EXPERTS);
    Ok(())
}

/// Dequantize an FP8 tensor from pre-loaded cache.
fn dequant_from_cache(
    cache: &HashMap<String, (Vec<u8>, Vec<usize>, safetensors::Dtype)>,
    name: &str,
    lut: &[f32; 256],
) -> Result<Vec<f32>> {
    let (bytes, shape, dtype) = cache.get(name)
        .with_context(|| format!("tensor {name} not in cache"))?;

    // Handle non-FP8 dtypes
    match dtype {
        safetensors::Dtype::F8_E4M3 | safetensors::Dtype::U8 => {}
        safetensors::Dtype::BF16 => {
            return Ok(bytes.chunks_exact(2)
                .map(|c| { let bits = u16::from_le_bytes([c[0], c[1]]); f32::from_bits((bits as u32) << 16) })
                .collect());
        }
        safetensors::Dtype::F32 => {
            return Ok(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect());
        }
        dt => bail!("unexpected dtype {dt:?} for {name}"),
    }

    assert_eq!(shape.len(), 2);
    let m = shape[0];
    let k = shape[1];

    // Load scale
    let scale_name = format!("{name}_scale_inv");
    let (scale_bytes, _, scale_dtype) = cache.get(&scale_name)
        .with_context(|| format!("scale {scale_name} not in cache"))?;

    let scale_values: Vec<f32> = match scale_dtype {
        safetensors::Dtype::F32 => {
            scale_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        safetensors::Dtype::BF16 => {
            scale_bytes.chunks_exact(2).map(|c| { let bits = u16::from_le_bytes([c[0], c[1]]); f32::from_bits((bits as u32) << 16) }).collect()
        }
        dt => bail!("unexpected scale dtype {dt:?}"),
    };

    let num_col_blocks = (k + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
    let mut out = vec![0.0f32; m * k];

    for i in 0..m {
        let row_block = i / FP8_BLOCK_SIZE;
        let row_offset = i * k;
        for j in 0..k {
            let col_block = j / FP8_BLOCK_SIZE;
            let scale = scale_values[row_block * num_col_blocks + col_block];
            out[row_offset + j] = lut[bytes[row_offset + j] as usize] * scale;
        }
    }

    Ok(out)
}

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

    let mut out = Vec::with_capacity(packed_u32s * 4 + total_groups * 4);
    for &w in &packed { out.extend_from_slice(&w.to_le_bytes()); }
    for &s in &scales { out.extend_from_slice(&s.to_le_bytes()); }
    for &z in &zeros { out.extend_from_slice(&z.to_le_bytes()); }
    out
}
