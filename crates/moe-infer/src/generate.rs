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

use crate::attention::{GqaConfig, RopeTables, gqa_forward_f32};
use crate::config::InferConfig;
use crate::kv_cache::KvCache;
use crate::rmsnorm::rmsnorm;
use crate::sampler;
use crate::weights::{BackboneWeights, LayerWeights};

use moe_router::{MoeRouter, RouterConfig};
use moe_kernels::{MetalDequantGemv, ExpertGemvOp};
use quantize::PackedWeights4Bit;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

/// Pre-converted f32 attention weights for one layer (eliminates per-token f16→f32 conversion).
pub struct LayerAttnF32 {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub router: Vec<f32>,
}

/// Loaded model ready for generation.
pub struct Model {
    pub weights: BackboneWeights,
    pub config: InferConfig,
    pub rope: RopeTables,
    pub gqa_config: GqaConfig,
    pub metal: Option<MetalDequantGemv>,
    /// Pre-wrapped Metal buffers for each layer's expert mmap (zero-copy).
    pub expert_metal_bufs: std::collections::HashMap<usize, Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Pre-converted f32 attention weights (eliminates ~5ms/layer f16→f32 conversion).
    pub attn_f32: Vec<LayerAttnF32>,
    /// LM head pre-converted to f32.
    pub lm_head_f32: Vec<f32>,
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

/// Generation output with metadata and timing breakdown.
pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub tokens_generated: usize,
    /// Prefill time in seconds (processing input tokens).
    pub prefill_secs: f64,
    /// Decode time in seconds (generating new tokens).
    pub decode_secs: f64,
    /// Number of input tokens (prompt length).
    pub prompt_tokens: usize,
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

        // Pre-convert attention weights from f16 to f32 (eliminates ~5ms/layer per-token conversion)
        let t_preconv = std::time::Instant::now();
        let num_layers = config.num_layers();
        let mut attn_f32 = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let lw = weights.layer_weights(layer)?;
            attn_f32.push(LayerAttnF32 {
                q_proj: lw.q_proj.iter().map(|v| v.to_f32()).collect(),
                k_proj: lw.k_proj.iter().map(|v| v.to_f32()).collect(),
                v_proj: lw.v_proj.iter().map(|v| v.to_f32()).collect(),
                o_proj: lw.o_proj.iter().map(|v| v.to_f32()).collect(),
                router: lw.router.iter().map(|v| v.to_f32()).collect(),
            });
        }
        let lm_head_f16 = weights.lm_head()?;
        let lm_head_f32: Vec<f32> = lm_head_f16.iter().map(|v| v.to_f32()).collect();
        eprintln!("Pre-converted attention weights to f32: {:.1}s", t_preconv.elapsed().as_secs_f64());

        let metal = MetalDequantGemv::new();
        let mut expert_metal_bufs = std::collections::HashMap::new();
        if let Some(ref m) = metal {
            eprintln!("Metal GPU: enabled");
            for layer in 0..num_layers {
                if let Some(mmap) = weights.expert_mmap(layer) {
                    let buf = m.wrap_mmap(mmap);
                    expert_metal_bufs.insert(layer, buf);
                }
            }
            eprintln!("Metal GPU: {} layers' expert weights wrapped (zero-copy)", expert_metal_bufs.len());
        } else {
            eprintln!("Metal GPU: not available, using CPU");
        }

        Ok(Self { weights, config, rope, gqa_config, metal, expert_metal_bufs, attn_f32, lm_head_f32 })
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

/// MoE FFN: route + dispatch top-k experts.
/// Uses zero-copy batched Metal GPU when mmap_buf is available (2 cmd_buf commits per layer),
/// falls back to CPU scalar dequant.
fn moe_ffn(
    x: &[f32],
    router_f32: &[f32],
    expert_mmap: Option<&memmap2::Mmap>,
    mmap_metal_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    router: &mut MoeRouter,
    metal: Option<&MetalDequantGemv>,
    hidden: usize,
    moe_inter: usize,
    num_experts: usize,
    group_size: usize,
) -> Vec<f32> {
    // 1. Router gate logits (pre-converted f32)
    let mut gate_logits = vec![0.0f32; num_experts];
    crate::blas::sgemv_f32(router_f32, x, &mut gate_logits, num_experts, hidden);

    // 2. Route (softmax for Qwen3)
    let route = router.route_softmax(&gate_logits);

    // 3. Dispatch routed experts
    let mut combined = vec![0.0f32; hidden];

    if let Some(expert_data) = expert_mmap {
        // Compute data layout constants
        let gu_packed = moe_inter * hidden / 2;
        let gu_groups = moe_inter * (hidden / group_size);
        let gu_scales = gu_groups * 2;
        let gu_total = gu_packed + gu_scales * 2; // gate_proj or up_proj total bytes

        let dn_packed = hidden * moe_inter / 2;
        let dn_groups = hidden * (moe_inter / group_size);
        let dn_scales = dn_groups * 2;
        let dn_total = dn_packed + dn_scales * 2;

        let expert_stride = gu_total * 2 + dn_total; // gate + up + down

        if let (Some(m), Some(mmap_buf)) = (metal, mmap_metal_buf) {
            // ===== ZERO-COPY BATCHED METAL PATH =====
            // No weight copies — use byte offsets into pre-wrapped mmap buffer.
            let mut gate_up_ops = Vec::with_capacity(route.expert_ids.len() * 2);
            let mut routing_weights: Vec<f32> = Vec::new();

            for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
                let base = eid * expert_stride;
                if base + expert_stride > expert_data.len() { continue; }

                // gate_proj: [moe_inter, hidden]
                gate_up_ops.push(ExpertGemvOp {
                    packed_offset: base,
                    scales_offset: base + gu_packed,
                    zeros_offset: base + gu_packed + gu_scales,
                    out_features: moe_inter, in_features: hidden, group_size,
                });

                // up_proj: [moe_inter, hidden]
                gate_up_ops.push(ExpertGemvOp {
                    packed_offset: base + gu_total,
                    scales_offset: base + gu_total + gu_packed,
                    zeros_offset: base + gu_total + gu_packed + gu_scales,
                    out_features: moe_inter, in_features: hidden, group_size,
                });

                routing_weights.push(weight);
            }

            let n = routing_weights.len();
            if n == 0 { return combined; }

            // Batch 1: all gate + up GEMVs (2*n dispatches, 1 commit, zero-copy weights)
            let x_slices: Vec<&[f32]> = vec![x; 2 * n];
            let gate_up_results = m.batch_gemv_mmap(mmap_buf, &gate_up_ops, &x_slices);

            // SiLU(gate) * up on CPU (tiny: 8 × 768 floats)
            let mut activated: Vec<Vec<f32>> = Vec::with_capacity(n);
            for i in 0..n {
                let gate_out = &gate_up_results[2 * i];
                let up_out = &gate_up_results[2 * i + 1];
                let h: Vec<f32> = gate_out.iter().zip(up_out.iter())
                    .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();
                activated.push(h);
            }

            // Build down ops
            let mut down_ops = Vec::with_capacity(n);
            let mut act_slices: Vec<&[f32]> = Vec::with_capacity(n);
            let mut eid_idx = 0;
            for (&eid, _) in route.expert_ids.iter().zip(route.weights.iter()) {
                let base = eid * expert_stride;
                if base + expert_stride > expert_data.len() { continue; }

                // down_proj: [hidden, moe_inter]
                down_ops.push(ExpertGemvOp {
                    packed_offset: base + 2 * gu_total,
                    scales_offset: base + 2 * gu_total + dn_packed,
                    zeros_offset: base + 2 * gu_total + dn_packed + dn_scales,
                    out_features: hidden, in_features: moe_inter, group_size,
                });
                act_slices.push(&activated[eid_idx]);
                eid_idx += 1;
            }

            // Batch 2: all down GEMVs (n dispatches, 1 commit, zero-copy weights)
            let down_results = m.batch_gemv_mmap(mmap_buf, &down_ops, &act_slices);

            // Combine with routing weights
            for (i, weight) in routing_weights.iter().enumerate() {
                for d in 0..hidden {
                    combined[d] += weight * down_results[i][d];
                }
            }
        } else {
            // ===== CPU FALLBACK =====
            for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
                let base = eid * expert_stride;
                if base + expert_stride > expert_data.len() { continue; }

                let expert_out = dequant_expert_ffn(
                    &expert_data[base..base + expert_stride], x, hidden, moe_inter, group_size,
                );
                for d in 0..hidden {
                    combined[d] += weight * expert_out[d];
                }
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

/// Matrix-vector multiply: y = W @ x using Accelerate BLAS (f32 weights, no conversion).
fn matvec_f32_direct(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    crate::blas::sgemv_f32(w, x, &mut y, out_dim, in_dim);
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

    // 1. RMSNorm → Attention → Residual (using pre-converted f32 weights)
    let normed = rmsnorm(x, &input_norm_gamma, eps);
    let af = &model.attn_f32[layer];
    let attn_out = gqa_forward_f32(
        &normed,
        &af.q_proj, &af.k_proj, &af.v_proj, &af.o_proj,
        lw.q_norm, lw.k_norm,
        cache, layer, pos, &model.rope, &model.gqa_config, eps,
    );
    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    // 2. RMSNorm → FFN → Residual
    let normed2 = rmsnorm(&residual, &post_norm_gamma, eps);

    // All layers are MoE (decoder_sparse_step=1)
    let expert_mmap = model.weights.expert_mmap(layer);
    let mmap_metal_buf = model.expert_metal_bufs.get(&layer)
        .map(|b| b.as_ref());
    let ffn_out = moe_ffn(
        &normed2,
        &af.router,
        expert_mmap,
        mmap_metal_buf,
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

    let mut all_ids = input_ids.clone();
    let prompt_tokens = input_ids.len();

    // Prefill: process all input tokens
    let t_prefill = std::time::Instant::now();
    for (i, &token_id) in input_ids.iter().enumerate() {
        let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
        let mut x = emb;
        for layer in 0..num_layers {
            x = run_layer(model, &mut cache, &mut router, layer, &x, i)?;
        }

        // Only sample from last prefill token
        if i == input_ids.len() - 1 {
            let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
            let logits = matvec_f32_direct(&model.lm_head_f32, &normed, vocab, hidden);
            let next_token = sample(&logits, sampling, i as u64);
            all_ids.push(next_token);
        }
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    // Decode: generate new tokens one at a time
    let t_decode = std::time::Instant::now();
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
        let logits = matvec_f32_direct(&model.lm_head_f32, &normed, vocab, hidden);
        let next_token = sample(&logits, sampling, (pos + step) as u64);
        all_ids.push(next_token);
        pos += 1;
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    let generated_ids = all_ids[input_ids.len()..].to_vec();
    let text = tokenizer.decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

    Ok(GenerateOutput {
        token_ids: generated_ids.clone(),
        text,
        tokens_generated: generated_ids.len(),
        prefill_secs,
        decode_secs,
        prompt_tokens,
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
