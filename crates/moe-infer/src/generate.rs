//! Generation loop: embed → 48 layers → LM head → sample → repeat.
//!
//! Layer structure:
//! - Layer 0 (dense): RMSNorm → Attention → Residual → RMSNorm → Dense FFN → Residual
//! - Layers 1-47 (MoE): RMSNorm → Attention → Residual → RMSNorm → MoE FFN → Residual

use half::f16;
use anyhow::{Result, bail};

/// Compare accelerated output against CPU reference.
/// Only active with `--features dual-path`.
#[cfg(feature = "dual-path")]
#[allow(dead_code)]
fn validate_output(accel: &[f32], cpu: &[f32], layer: usize, op: &str) -> Vec<f32> {
    let max_diff = accel.iter().zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    if max_diff > 1e-2 {
        eprintln!("WARN: L{layer} [{op}] max_diff={max_diff:.6} — using CPU fallback");
        cpu.to_vec()
    } else {
        accel.to_vec()
    }
}

use crate::attention::{GqaConfig, RopeTables, gqa_forward};
use crate::config::InferConfig;
use crate::kv_cache::KvCache;
use crate::rmsnorm::rmsnorm;
use crate::sampler;
use crate::weights::{BackboneWeights, LayerWeights};

use moe_router::{MoeRouter, RouterConfig};
use moe_kernels::MetalDequantGemv;
use quantize::PackedWeights4Bit;

/// Loaded model ready for generation.
pub struct Model {
    pub weights: BackboneWeights,
    pub config: InferConfig,
    pub rope: RopeTables,
    pub gqa_config: GqaConfig,
    pub metal: Option<MetalDequantGemv>,
}

/// Sampling configuration.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub greedy: bool,
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 1, greedy: true }
    }
}

/// Generation output with metadata.
pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub tokens_generated: usize,
}

impl Model {
    /// Load model from weights directory + config TOML.
    pub fn load(weights_dir: &std::path::Path, config_path: &std::path::Path) -> Result<Self> {
        let config = InferConfig::from_toml(config_path)
            .map_err(|e| anyhow::anyhow!(e))?;
        let weights = BackboneWeights::load(weights_dir)?;

        let max_seq = config.model.max_position_embeddings.min(4096); // Cap for memory
        let rope = RopeTables::build(max_seq, config.head_dim(), config.rope_theta());
        let gqa_config = GqaConfig {
            num_q_heads: config.num_q_heads(),
            num_kv_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            max_seq,
        };

        let metal = MetalDequantGemv::new();
        if metal.is_some() {
            eprintln!("Metal GPU: enabled");
        } else {
            eprintln!("Metal GPU: not available, using CPU");
        }

        Ok(Self { weights, config, rope, gqa_config, metal })
    }
}

/// Parse raw expert bytes into a PackedWeights4Bit for Metal GEMV.
fn parse_packed_weights(
    data: &[u8], out_features: usize, in_features: usize, group_size: usize,
) -> PackedWeights4Bit {
    let packed_u32s = out_features * in_features / 8;
    let packed_bytes = packed_u32s * 4;
    let num_groups = out_features * (in_features / group_size);

    let data_u32 = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u32, packed_u32s)
    }.to_vec();
    let scales = unsafe {
        std::slice::from_raw_parts(data[packed_bytes..].as_ptr() as *const f16, num_groups)
    }.to_vec();
    let zeros = unsafe {
        std::slice::from_raw_parts(
            data[packed_bytes + num_groups * 2..].as_ptr() as *const f16, num_groups
        )
    }.to_vec();

    PackedWeights4Bit { data: data_u32, scales, zeros, out_features, in_features, group_size }
}

/// Run one expert's FFN on Metal GPU: gate_proj → SiLU → up_proj → down_proj.
fn metal_gemv_expert(
    metal: &MetalDequantGemv,
    data: &[u8], x: &[f32], hidden: usize, inter: usize, group_size: usize,
) -> Vec<f32> {
    let matrix_packed = inter * hidden / 2;
    let num_groups = inter * (hidden / group_size);
    let scales_size = num_groups * 2;
    let matrix_total = matrix_packed + scales_size * 2;

    let gate = parse_packed_weights(&data[0..matrix_total], inter, hidden, group_size);
    let up = parse_packed_weights(&data[matrix_total..2 * matrix_total], inter, hidden, group_size);

    // down_proj: [hidden, inter]
    let down_packed = inter * hidden / 2;
    let down_groups = hidden * (inter / group_size);
    let down_scales = down_groups * 2;
    let down_total = down_packed + down_scales * 2;
    let down = parse_packed_weights(&data[2 * matrix_total..2 * matrix_total + down_total], hidden, inter, group_size);

    let gate_out = metal.gemv(&gate, x);
    let up_out = metal.gemv(&up, x);

    // SiLU(gate) * up
    let h: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();

    metal.gemv(&down, &h)
}

/// MoE FFN: route + dispatch top-k experts.
/// Uses Metal GPU when available, falls back to CPU scalar dequant.
fn moe_ffn(
    x: &[f32],
    lw: &LayerWeights<'_>,
    expert_mmap: Option<&memmap2::Mmap>,
    router: &mut MoeRouter,
    metal: Option<&MetalDequantGemv>,
    hidden: usize,
    moe_inter: usize,
    num_experts: usize,
    group_size: usize,
) -> Vec<f32> {
    // 1. Router gate logits
    let gate_logits = matvec_f16(lw.router, x, num_experts, hidden);

    // 2. Route (softmax for Qwen3)
    let route = router.route_softmax(&gate_logits);

    // 3. Dispatch routed experts (4-bit quantized from expert file)
    let mut combined = vec![0.0f32; hidden];

    if let Some(expert_data) = expert_mmap {
        // Expert stride: each expert has gate_proj + up_proj + down_proj in 4-bit
        // Size per matrix: moe_inter * hidden / 2 (4-bit = 0.5 bytes)
        // Plus scales/zeros: (moe_inter * hidden / group_size) * 2 bytes each
        let matrix_packed_bytes = moe_inter * hidden / 2; // 4-bit packed
        let num_groups = moe_inter * (hidden / group_size);
        let scales_bytes = num_groups * 2; // f16
        let matrix_total = matrix_packed_bytes + scales_bytes * 2; // packed + scales + zeros
        let expert_stride = matrix_total * 3; // gate + up + down

        for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
            let base = eid * expert_stride;
            if base + expert_stride > expert_data.len() {
                continue; // Skip if out of bounds (safety)
            }

            let expert_out = if let Some(m) = metal {
                metal_gemv_expert(m, &expert_data[base..base + expert_stride], x, hidden, moe_inter, group_size)
            } else {
                dequant_expert_ffn(&expert_data[base..base + expert_stride], x, hidden, moe_inter, group_size)
            };

            for d in 0..hidden {
                combined[d] += weight * expert_out[d];
            }
        }
    }

    combined
}

/// Dequantize and run one expert's FFN from packed 4-bit data.
fn dequant_expert_ffn(
    data: &[u8],
    x: &[f32],
    hidden: usize,
    inter: usize,
    group_size: usize,
) -> Vec<f32> {
    let matrix_packed = inter * hidden / 2;
    let num_groups = inter * (hidden / group_size);
    let scales_size = num_groups * 2;
    let matrix_total = matrix_packed + scales_size * 2;

    // Parse three matrices: gate, up, down
    let gate_data = &data[0..matrix_total];
    let up_data = &data[matrix_total..2 * matrix_total];
    let down_data = &data[2 * matrix_total..3 * matrix_total];

    let gate_out = dequant_gemv(gate_data, x, inter, hidden, group_size);
    let up_out = dequant_gemv(up_data, x, inter, hidden, group_size);

    // SiLU(gate) * up
    let mut activated = vec![0.0f32; inter];
    for i in 0..inter {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        activated[i] = silu * up_out[i];
    }

    // down_proj: [hidden, inter] — note transposed dims
    dequant_gemv(down_data, &activated, hidden, inter, group_size)
}

/// Dequantize 4-bit packed data and perform GEMV.
/// Data layout: [packed_u32s | scales_f16 | zeros_f16]
fn dequant_gemv(
    data: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<f32> {
    let packed_u32s = out_dim * in_dim / 8;
    let packed_bytes = packed_u32s * 4;
    let num_groups_per_row = in_dim / group_size;
    let total_groups = out_dim * num_groups_per_row;

    let packed = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u32, packed_u32s)
    };
    let scales = unsafe {
        std::slice::from_raw_parts(
            data[packed_bytes..].as_ptr() as *const f16,
            total_groups,
        )
    };
    let zeros = unsafe {
        std::slice::from_raw_parts(
            data[packed_bytes + total_groups * 2..].as_ptr() as *const f16,
            total_groups,
        )
    };

    let packed_per_row = in_dim / 8;
    let mut y = vec![0.0f32; out_dim];

    for row in 0..out_dim {
        let mut sum = 0.0f64;
        for col in 0..in_dim {
            let packed_idx = row * packed_per_row + col / 8;
            let nibble_pos = col % 8;
            let q = ((packed[packed_idx] >> (nibble_pos as u32 * 4)) & 0xF) as f32;

            let group_idx = row * num_groups_per_row + col / group_size;
            let scale = scales[group_idx].to_f32();
            let zero = zeros[group_idx].to_f32();

            let w = q * scale + zero;
            sum += w as f64 * x[col] as f64;
        }
        y[row] = sum as f32;
    }
    y
}

/// Matrix-vector multiply: y = W @ x using Accelerate BLAS.
/// W is [out, in] row-major as f16, x is [in] as f32.
fn matvec_f16(w: &[f16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    crate::blas::sgemv_f16(w, x, &mut y, out_dim, in_dim);
    y
}

/// Run a single transformer layer.
pub fn run_layer(
    model: &Model,
    cache: &mut KvCache,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    pos: usize,
) -> Result<Vec<f32>> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();
    let lw = model.weights.layer_weights(layer)?;

    // Convert f32 norm weights to owned vec for rmsnorm
    let input_norm_gamma = lw.input_norm.to_vec();
    let post_norm_gamma = lw.post_attn_norm.to_vec();

    // 1. RMSNorm → Attention → Residual
    let normed = rmsnorm(x, &input_norm_gamma, eps);
    let attn_out = gqa_forward(
        &normed, &lw, cache, layer, pos, &model.rope, &model.gqa_config, eps,
    );
    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    // 2. RMSNorm → FFN → Residual
    let normed2 = rmsnorm(&residual, &post_norm_gamma, eps);

    // All layers are MoE (decoder_sparse_step=1)
    let expert_mmap = model.weights.expert_mmap(layer);
    let ffn_out = moe_ffn(
        &normed2,
        &lw,
        expert_mmap,
        router,
        model.metal.as_ref(),
        hidden,
        model.config.moe_inter_size(),
        model.config.num_experts(),
        model.config.quantization.group_size,
    );

    for d in 0..hidden {
        residual[d] += ffn_out[d];
    }

    Ok(residual)
}

/// Generate tokens from a prompt.
pub fn generate(
    model: &Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &SamplingConfig,
) -> Result<GenerateOutput> {
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    if input_ids.is_empty() {
        bail!("empty prompt after tokenization");
    }

    let hidden = model.config.hidden_size();
    let vocab = model.config.vocab_size();
    let num_layers = model.config.num_layers();
    let max_seq = model.gqa_config.max_seq;

    // Create KV cache
    let mut cache = KvCache::new(
        num_layers,
        model.config.num_kv_heads(),
        model.config.head_dim(),
        max_seq,
    );

    // Create router (one shared instance)
    let router_config = RouterConfig {
        num_experts: model.config.num_experts(),
        top_k: model.config.num_experts_per_tok(),
        norm_topk_prob: model.config.ffn.norm_topk_prob,
        bias_lr: 0.0, // No load balancing during inference
    };
    let mut router = MoeRouter::new(router_config);

    let embed_table = model.weights.embed_table()?;
    let final_norm = model.weights.final_norm()?;
    let lm_head = model.weights.lm_head()?;

    let mut all_ids = input_ids.clone();

    // Prefill: process all input tokens
    for (i, &token_id) in input_ids.iter().enumerate() {
        let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_layer(model, &mut cache, &mut router, layer, &x, i)?;
        }

        // Only sample from last prefill token
        if i == input_ids.len() - 1 {
            let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
            let logits = matvec_f16(lm_head, &normed, vocab, hidden);
            let next_token = sample(&logits, sampling, i as u64);
            all_ids.push(next_token);
        }
    }

    // Decode: generate new tokens one at a time
    let mut pos = input_ids.len();
    for step in 0..max_new_tokens.saturating_sub(1) {
        if pos >= max_seq {
            break;
        }
        let token_id = *all_ids.last().unwrap();
        if token_id == model.config.model.eos_token_id {
            break;
        }

        let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_layer(model, &mut cache, &mut router, layer, &x, pos)?;
        }

        let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
        let logits = matvec_f16(lm_head, &normed, vocab, hidden);
        let next_token = sample(&logits, sampling, (pos + step) as u64);
        all_ids.push(next_token);
        pos += 1;
    }

    let generated_ids = all_ids[input_ids.len()..].to_vec();
    let text = tokenizer.decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

    Ok(GenerateOutput {
        token_ids: generated_ids.clone(),
        text,
        tokens_generated: generated_ids.len(),
    })
}

/// Embed a token: extract row from f16 embed table, convert to f32.
fn embed_f16_to_f32(table: &[f16], token_id: usize, hidden: usize) -> Vec<f32> {
    let start = token_id * hidden;
    table[start..start + hidden].iter().map(|v| v.to_f32()).collect()
}

/// Sample next token from logits.
fn sample(logits: &[f32], config: &SamplingConfig, seed: u64) -> u32 {
    if config.greedy {
        sampler::sample_greedy(logits) as u32
    } else {
        sampler::sample_top_k(logits, config.top_k, config.temperature, seed) as u32
    }
}
