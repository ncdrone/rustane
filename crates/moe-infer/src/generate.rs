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

use ane_bridge::ane::TensorData;
use moe_router::{MoeRouter, RouterConfig};
use moe_kernels::{MetalDequantGemv, ExpertGemvOp, FusedGateUpSiluOp, gqa_prefill};
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

/// Pre-compiled ANE graphs for batched prefill attention.
pub struct AnePrefillCache {
    /// One compiled graph per layer (q_norm_w/k_norm_w baked as constants).
    exes: Vec<ane_bridge::ane::Executable>,
    /// Shared activation IOSurface [1, hidden, 1, seq] — re-staged per layer (tiny).
    acts_td: TensorData,
    /// Per-layer weight IOSurfaces [1, hidden, 1, wts_sp] — pre-staged at load time.
    wts_tds: Vec<TensorData>,
    /// Shared output IOSurface [1, out_ch, 1, seq].
    output_td: TensorData,
    /// Compiled sequence length (64, 128, or 256).
    pub seq: usize,
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
    /// Pre-compiled ANE prefill graphs (None if ANE unavailable).
    pub ane_prefill: Option<AnePrefillCache>,
}

/// Per-operation timing for one layer (microseconds).
#[derive(Default, Clone, Debug)]
pub struct LayerTimings {
    pub rmsnorm1_us: f64,
    pub attn_proj_us: f64,      // Q+K+V+O sgemv total
    pub attn_scores_us: f64,    // RoPE + QK^T + softmax + weighted V
    pub rmsnorm2_us: f64,
    pub router_us: f64,         // gate sgemv + softmax + topk
    pub metal_dispatch_us: f64, // Metal gate+up+down (all dispatches + waits)
    pub cpu_silu_us: f64,       // CPU SiLU between batches
    pub combine_us: f64,        // weighted expert combine + residuals
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

        let mut metal = MetalDequantGemv::new();
        let mut expert_metal_bufs = std::collections::HashMap::new();
        if let Some(ref mut m) = metal {
            eprintln!("Metal GPU: enabled");
            // Pre-allocate scratch buffers for zero-allocation decode
            m.init_scratch(
                config.hidden_size(),
                config.moe_inter_size(),
                config.num_experts_per_tok(),
            );
            for layer in 0..num_layers {
                if let Some(mmap) = weights.expert_mmap(layer) {
                    let buf = m.wrap_mmap(mmap);
                    expert_metal_bufs.insert(layer, buf);
                }
            }
            eprintln!("Metal GPU: {} layers' expert weights wrapped (zero-copy, scratch pre-allocated)", expert_metal_bufs.len());
        } else {
            eprintln!("Metal GPU: not available, using CPU");
        }

        // Pre-compile ANE prefill graphs (one per layer, seq=64)
        let ane_prefill = {
            let t = std::time::Instant::now();
            let cache = compile_ane_prefill(&config, &weights, &attn_f32, 64);
            if cache.is_some() {
                eprintln!("ANE prefill: compiled {} graphs for seq=64 in {:.1}s",
                    num_layers, t.elapsed().as_secs_f64());
            } else {
                eprintln!("ANE prefill: compilation failed, using sequential CPU");
            }
            cache
        };

        Ok(Self { weights, config, rope, gqa_config, metal, expert_metal_bufs, attn_f32, lm_head_f32, ane_prefill })
    }
}

fn compile_ane_prefill(
    config: &InferConfig,
    weights: &BackboneWeights,
    attn_f32: &[LayerAttnF32],
    seq: usize,
) -> Option<AnePrefillCache> {
    use objc2_foundation::NSQualityOfService;

    let hidden = config.hidden_size();
    let num_q_heads = config.num_q_heads();
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let num_layers = config.num_layers();
    let wts_sp = gqa_prefill::weight_spatial_width(q_dim, kv_dim);

    let mut exes = Vec::with_capacity(num_layers);
    let mut wts_tds = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let lw = weights.layer_weights(layer).ok()?;
        let graph = gqa_prefill::build_gqa_prefill(
            hidden, num_q_heads, num_kv_heads, head_dim,
            seq, config.rope_theta(), config.rms_norm_eps(),
            lw.q_norm, lw.k_norm,
        );
        let exe = graph.compile(NSQualityOfService::UserInteractive).ok()?;
        exes.push(exe);

        // Pre-stage weights (transposed, done once at load time)
        let af = &attn_f32[layer];
        let wts_td = TensorData::new(ane_bridge::ane::Shape {
            batch: 1, channels: hidden, height: 1, width: wts_sp,
        });
        {
            let mut locked = wts_td.as_f32_slice_mut();
            let buf = &mut *locked;
            for c in 0..hidden {
                for i in 0..q_dim {
                    buf[c * wts_sp + i] = af.q_proj[i * hidden + c];
                }
                for i in 0..kv_dim {
                    buf[c * wts_sp + q_dim + i] = af.k_proj[i * hidden + c];
                }
                for i in 0..kv_dim {
                    buf[c * wts_sp + q_dim + kv_dim + i] = af.v_proj[i * hidden + c];
                }
            }
        }
        wts_tds.push(wts_td);
    }

    let out_ch = gqa_prefill::output_channels(q_dim, kv_dim);
    let acts_td = TensorData::new(ane_bridge::ane::Shape {
        batch: 1, channels: hidden, height: 1, width: seq,
    });
    let output_td = TensorData::new(ane_bridge::ane::Shape {
        batch: 1, channels: out_ch, height: 1, width: seq,
    });

    Some(AnePrefillCache { exes, acts_td, wts_tds, output_td, seq })
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
            // ===== SINGLE CMD_BUF METAL PATH =====
            // Fused gate+up+SiLU + down in ONE command buffer commit.
            let mut fused_ops = Vec::with_capacity(route.expert_ids.len());
            let mut down_ops = Vec::with_capacity(route.expert_ids.len());
            let mut routing_weights: Vec<f32> = Vec::new();

            for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
                let base = eid * expert_stride;
                if base + expert_stride > expert_data.len() { continue; }

                fused_ops.push(FusedGateUpSiluOp {
                    gate_packed_offset: base,
                    gate_scales_offset: base + gu_packed,
                    gate_zeros_offset: base + gu_packed + gu_scales,
                    up_packed_offset: base + gu_total,
                    up_scales_offset: base + gu_total + gu_packed,
                    up_zeros_offset: base + gu_total + gu_packed + gu_scales,
                    out_features: moe_inter, in_features: hidden, group_size,
                });

                down_ops.push(ExpertGemvOp {
                    packed_offset: base + 2 * gu_total,
                    scales_offset: base + 2 * gu_total + dn_packed,
                    zeros_offset: base + 2 * gu_total + dn_packed + dn_scales,
                    out_features: hidden, in_features: moe_inter, group_size,
                });

                routing_weights.push(weight);
            }

            let n = routing_weights.len();
            if n == 0 { return combined; }

            // Single cmd_buf: fused gate+up+SiLU → down (scratch buffers pipeline data)
            let down_results = m.fused_and_down_single_cmdbuf(mmap_buf, &fused_ops, &down_ops, x);

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

/// MoE FFN with timing instrumentation. Returns (output, router_us, metal_us, silu_us, combine_us).
fn moe_ffn_timed(
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
) -> (Vec<f32>, f64, f64, f64, f64) {
    use std::time::Instant;

    // 1. Router gate logits
    let t = Instant::now();
    let mut gate_logits = vec![0.0f32; num_experts];
    crate::blas::sgemv_f32(router_f32, x, &mut gate_logits, num_experts, hidden);
    let route = router.route_softmax(&gate_logits);
    let router_us = t.elapsed().as_secs_f64() * 1e6;

    // 2. Dispatch routed experts
    let mut combined = vec![0.0f32; hidden];
    let mut metal_us = 0.0;
    let mut silu_us = 0.0;

    if let Some(expert_data) = expert_mmap {
        let gu_packed = moe_inter * hidden / 2;
        let gu_groups = moe_inter * (hidden / group_size);
        let gu_scales = gu_groups * 2;
        let gu_total = gu_packed + gu_scales * 2;
        let dn_packed = hidden * moe_inter / 2;
        let dn_groups = hidden * (moe_inter / group_size);
        let dn_scales = dn_groups * 2;
        let dn_total = dn_packed + dn_scales * 2;
        let expert_stride = gu_total * 2 + dn_total;

        if let (Some(m), Some(mmap_buf)) = (metal, mmap_metal_buf) {
            let mut fused_ops = Vec::with_capacity(route.expert_ids.len());
            let mut down_ops = Vec::with_capacity(route.expert_ids.len());
            let mut routing_weights: Vec<f32> = Vec::new();

            for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
                let base = eid * expert_stride;
                if base + expert_stride > expert_data.len() { continue; }
                fused_ops.push(FusedGateUpSiluOp {
                    gate_packed_offset: base,
                    gate_scales_offset: base + gu_packed,
                    gate_zeros_offset: base + gu_packed + gu_scales,
                    up_packed_offset: base + gu_total,
                    up_scales_offset: base + gu_total + gu_packed,
                    up_zeros_offset: base + gu_total + gu_packed + gu_scales,
                    out_features: moe_inter, in_features: hidden, group_size,
                });
                down_ops.push(ExpertGemvOp {
                    packed_offset: base + 2 * gu_total, scales_offset: base + 2 * gu_total + dn_packed,
                    zeros_offset: base + 2 * gu_total + dn_packed + dn_scales,
                    out_features: hidden, in_features: moe_inter, group_size,
                });
                routing_weights.push(weight);
            }

            let n = routing_weights.len();
            if n == 0 { return (combined, router_us, 0.0, 0.0, 0.0); }

            // Single cmd_buf: fused + down
            let t = Instant::now();
            let down_results = m.fused_and_down_single_cmdbuf(mmap_buf, &fused_ops, &down_ops, x);
            metal_us = t.elapsed().as_secs_f64() * 1e6;

            let t = Instant::now();
            for (i, weight) in routing_weights.iter().enumerate() {
                for d in 0..hidden {
                    combined[d] += weight * down_results[i][d];
                }
            }
            let combine_us = t.elapsed().as_secs_f64() * 1e6;
            return (combined, router_us, metal_us, silu_us, combine_us);
        }
    }

    (combined, router_us, 0.0, 0.0, 0.0)
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

/// Run a single transformer layer with per-operation timing instrumentation.
/// Returns (output, timings). Produces identical output to run_layer.
pub fn run_layer_timed(
    model: &Model,
    cache: &mut KvCache,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    pos: usize,
) -> Result<(Vec<f32>, LayerTimings)> {
    use std::time::Instant;
    let mut timings = LayerTimings::default();

    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();
    let lw = model.weights.layer_weights(layer)?;

    let input_norm_gamma = lw.input_norm.to_vec();
    let post_norm_gamma = lw.post_attn_norm.to_vec();

    // 1. RMSNorm1
    let t = Instant::now();
    let normed = rmsnorm(x, &input_norm_gamma, eps);
    timings.rmsnorm1_us = t.elapsed().as_secs_f64() * 1e6;

    // 2. Attention (projections + scores + O_proj — measured as one block)
    let t = Instant::now();
    let af = &model.attn_f32[layer];
    let attn_out = gqa_forward_f32(
        &normed,
        &af.q_proj, &af.k_proj, &af.v_proj, &af.o_proj,
        lw.q_norm, lw.k_norm,
        cache, layer, pos, &model.rope, &model.gqa_config, eps,
    );
    timings.attn_proj_us = t.elapsed().as_secs_f64() * 1e6;

    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    // 3. RMSNorm2
    let t = Instant::now();
    let normed2 = rmsnorm(&residual, &post_norm_gamma, eps);
    timings.rmsnorm2_us = t.elapsed().as_secs_f64() * 1e6;

    // 4. MoE FFN (timed)
    let expert_mmap = model.weights.expert_mmap(layer);
    let mmap_metal_buf = model.expert_metal_bufs.get(&layer).map(|b| b.as_ref());
    let (ffn_out, router_us, metal_us, silu_us, combine_us) = moe_ffn_timed(
        &normed2, &af.router, expert_mmap, mmap_metal_buf,
        router, model.metal.as_ref(), hidden,
        model.config.moe_inter_size(), model.config.num_experts(),
        model.config.quantization.group_size,
    );
    timings.router_us = router_us;
    timings.metal_dispatch_us = metal_us;
    timings.cpu_silu_us = silu_us;
    timings.combine_us = combine_us;

    for d in 0..hidden {
        residual[d] += ffn_out[d];
    }

    Ok((residual, timings))
}

/// Process one layer for ALL prefill tokens using ANE batched attention.
/// Modifies xs in place (accumulates residuals).
fn prefill_layer_ane(
    model: &Model,
    cache: &mut KvCache,
    router: &mut MoeRouter,
    layer: usize,
    xs: &mut [Vec<f32>],
    ane: &AnePrefillCache,
) -> Result<()> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();
    let q_dim = model.gqa_config.num_q_heads * model.gqa_config.head_dim;
    let kv_dim = model.gqa_config.num_kv_heads * model.gqa_config.head_dim;
    let lw = model.weights.layer_weights(layer)?;
    let af = &model.attn_f32[layer];
    let prompt_len = xs.len();
    let seq = ane.seq;

    let input_norm_gamma = lw.input_norm.to_vec();
    let post_norm_gamma = lw.post_attn_norm.to_vec();

    // 1. RMSNorm all tokens
    let normed: Vec<Vec<f32>> = xs.iter()
        .map(|x| rmsnorm(x, &input_norm_gamma, eps))
        .collect();

    // 2. Stage activations into IOSurface (tiny: hidden × seq = 512 KB)
    {
        let mut locked = ane.acts_td.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..hidden {
            for t in 0..seq {
                buf[c * seq + t] = if t < prompt_len { normed[t][c] } else { 0.0 };
            }
        }
    }

    // 3. ANE dispatch (weights pre-staged at load time)
    ane.exes[layer].run(&[&ane.acts_td, &ane.wts_tds[layer]], &[&ane.output_td])
        .map_err(|e| anyhow::anyhow!("ANE prefill L{layer}: {e:?}"))?;

    // 4. Read output and apply O_proj + cache write per token
    let output_data: Vec<f32>;
    {
        let locked = ane.output_td.as_f32_slice();
        output_data = locked.to_vec();
    }

    for t in 0..prompt_len {
        // Raw attention output [q_dim] from channels [0:q_dim]
        let mut attn_concat = vec![0.0f32; q_dim];
        for i in 0..q_dim {
            attn_concat[i] = output_data[i * seq + t];
        }

        // O_proj via BLAS (CPU)
        let attn_out = matvec_f32_direct(&af.o_proj, &attn_concat, hidden, q_dim);

        // K/V for cache from channels [q_dim:q_dim+kv_dim] and [q_dim+kv_dim:]
        let mut k_rope = vec![0.0f32; kv_dim];
        let mut v_vals = vec![0.0f32; kv_dim];
        for i in 0..kv_dim {
            k_rope[i] = output_data[(q_dim + i) * seq + t];
            v_vals[i] = output_data[(q_dim + kv_dim + i) * seq + t];
        }
        cache.store(layer, t, &k_rope, &v_vals);

        // Residual add (attention)
        for d in 0..hidden {
            xs[t][d] += attn_out[d];
        }
    }

    // 5. Per-token: RMSNorm → MoE FFN → Residual
    for t in 0..prompt_len {
        let normed2 = rmsnorm(&xs[t], &post_norm_gamma, eps);
        let expert_mmap = model.weights.expert_mmap(layer);
        let mmap_metal_buf = model.expert_metal_bufs.get(&layer).map(|b| b.as_ref());
        let ffn_out = moe_ffn(
            &normed2, &af.router, expert_mmap, mmap_metal_buf,
            router, model.metal.as_ref(),
            hidden, model.config.moe_inter_size(),
            model.config.num_experts(), model.config.quantization.group_size,
        );
        for d in 0..hidden {
            xs[t][d] += ffn_out[d];
        }
    }

    Ok(())
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
    let use_ane = model.ane_prefill.as_ref()
        .is_some_and(|ane| input_ids.len() <= ane.seq);

    if use_ane {
        // ANE batched prefill: all tokens through each layer together
        let ane = model.ane_prefill.as_ref().unwrap();
        let mut xs: Vec<Vec<f32>> = input_ids.iter()
            .map(|&id| embed_f16_to_f32(embed_table, id as usize, hidden))
            .collect();
        for layer in 0..num_layers {
            prefill_layer_ane(model, &mut cache, &mut router, layer, &mut xs, ane)?;
        }
        let last = &xs[xs.len() - 1];
        let normed = rmsnorm(last, &final_norm.to_vec(), model.config.rms_norm_eps());
        let logits = matvec_f32_direct(&model.lm_head_f32, &normed, vocab, hidden);
        let next_token = sample(&logits, sampling, (input_ids.len() - 1) as u64);
        all_ids.push(next_token);
    } else {
        // Sequential CPU prefill (fallback for long prompts or no ANE)
        for (i, &token_id) in input_ids.iter().enumerate() {
            let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
            let mut x = emb;
            for layer in 0..num_layers {
                x = run_layer(model, &mut cache, &mut router, layer, &x, i)?;
            }
            if i == input_ids.len() - 1 {
                let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
                let logits = matvec_f32_direct(&model.lm_head_f32, &normed, vocab, hidden);
                let next_token = sample(&logits, sampling, i as u64);
                all_ids.push(next_token);
            }
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
