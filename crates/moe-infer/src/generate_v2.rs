//! DeepSeek-V2 generation loop: MLA attention + dense/MoE FFN + YaRN RoPE.
//!
//! Layer structure:
//! - Layer 0 (dense): RMSNorm → MLA Attention → Residual → RMSNorm → Dense FFN → Residual
//! - Layers 1-26 (MoE): RMSNorm → MLA Attention → Residual → RMSNorm → MoE + Shared FFN → Residual

use half::f16;
use anyhow::{Result, bail};

use crate::config::InferConfig;
use crate::mla_attention::{MlaDecodeConfig, MlaLayerWeights as MlaAttnWeights, MlaKvCache, mla_forward_decode};
use crate::rmsnorm::rmsnorm;
use crate::sampler;
use crate::weights::BackboneWeights;
use crate::yarn_rope::{YarnRopeTables, compute_mscale, mla_attention_scale};

use moe_router::{MoeRouter, RouterConfig};
use moe_kernels::{MetalDequantGemv, ExpertGemvOp, FusedGateUpSiluOp};
use expert_pager::{ExpertPool, ExpertLoader};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use std::cell::RefCell;

/// Whether to run ExpertPool simulation (set RUSTANE_POOL_SIM=1).
fn pool_sim_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("RUSTANE_POOL_SIM").is_ok())
}

/// Whether to write back pread'd expert data to pool (set RUSTANE_POOL_WRITEBACK=1).
/// Previous attempt with 3000 slots (69 GB) caused 3x regression from page cache pressure.
/// Use small RUSTANE_POOL_CAP (100-500) to keep memory footprint under ~12 GB.
fn pool_writeback_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("RUSTANE_POOL_WRITEBACK").is_ok())
}

// Thread-local ExpertPool simulation for measuring decode-phase hit rates.
// When Some, moe_ffn_v2 calls pool.request() for each routed expert (tracking only).
// When None (default), no overhead. Gated by RUSTANE_POOL_SIM=1.
thread_local! {
    static POOL_SIM: RefCell<Option<ExpertPool>> = RefCell::new(None);
}

/// Pre-converted f32 MLA weights for one layer.
pub struct MlaLayerF32 {
    pub q_proj: Vec<f32>,
    pub kv_a_proj: Vec<f32>,
    pub kv_a_layernorm: Vec<f32>,
    pub w_uk: Vec<f32>,
    pub w_uv: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    // V3 Q LoRA (None for V2-Lite)
    pub q_a_proj: Option<Vec<f32>>,
    pub q_a_layernorm: Option<Vec<f32>>,
    pub q_b_proj: Option<Vec<f32>>,
    // FFN (either router+shared or dense)
    pub router: Option<Vec<f32>>,
    pub e_score_correction_bias: Option<Vec<f32>>,
    pub shared_gate: Option<Vec<f32>>,
    pub shared_up: Option<Vec<f32>>,
    pub shared_down: Option<Vec<f32>>,
    pub dense_gate: Option<Vec<f32>>,
    pub dense_up: Option<Vec<f32>>,
    pub dense_down: Option<Vec<f32>>,
}

impl MlaLayerF32 {
    /// Create an empty buffer (all Vecs with zero capacity).
    /// Used for double-buffer initialization; first convert_layer_into will allocate.
    pub fn empty() -> Self {
        Self {
            q_proj: Vec::new(),
            kv_a_proj: Vec::new(),
            kv_a_layernorm: Vec::new(),
            w_uk: Vec::new(),
            w_uv: Vec::new(),
            o_proj: Vec::new(),
            input_norm: Vec::new(),
            post_attn_norm: Vec::new(),
            q_a_proj: None,
            q_a_layernorm: None,
            q_b_proj: None,
            router: None,
            e_score_correction_bias: None,
            shared_gate: None,
            shared_up: None,
            shared_down: None,
            dense_gate: None,
            dense_up: None,
            dense_down: None,
        }
    }
}

/// Loaded DeepSeek-V2 model ready for generation.
pub struct ModelV2 {
    pub weights: BackboneWeights,
    pub config: InferConfig,
    pub mla_config: MlaDecodeConfig,
    pub rope: YarnRopeTables,
    pub attn_scale: f32,
    pub metal: Option<MetalDequantGemv>,
    pub expert_metal_bufs: std::collections::HashMap<usize, Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub layers_f32: Vec<MlaLayerF32>,
    pub lm_head_f32: Vec<f32>,
    /// Per-layer expert file loaders for pread-based dispatch (V3 optimization).
    pub expert_loaders: std::collections::HashMap<usize, ExpertLoader>,
    /// Expert stride in bytes (computed once at load time).
    pub expert_stride: usize,
    /// Staging buffer for packing selected experts before Metal dispatch.
    /// Size: top_k * expert_stride bytes. Reused across layers and tokens.
    pub expert_staging: Vec<u8>,
    /// Pre-wrapped Metal buffer for expert_staging (created once, reused).
    pub expert_staging_metal: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Expert pool: Least-Stale cache for hot experts (avoids pread on hit).
    /// UnsafeCell because pool state mutates during &self decode (single-threaded access).
    pub expert_pool: std::cell::UnsafeCell<Option<expert_pager::ExpertPool>>,
    /// Pool backing buffers: one Vec<u8> per pool slot, size = expert_stride.
    /// Each buffer is page-aligned for zero-copy Metal wrapping.
    pub pool_buffers: std::cell::UnsafeCell<Vec<Vec<u8>>>,
    /// Metal buffers wrapping pool_buffers (zero-copy: GPU reads pool memory directly).
    pub pool_metal_bufs: std::cell::UnsafeCell<Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>>,
}

/// Sampling configuration.
pub use crate::generate::SamplingConfig;

/// Generation output with metadata.
pub struct GenerateV2Output {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub tokens_generated: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub prompt_tokens: usize,
}

impl ModelV2 {
    /// Load model from weights directory + config TOML.
    pub fn load(weights_dir: &std::path::Path, config_path: &std::path::Path) -> Result<Self> {
        let config = InferConfig::from_toml(config_path)
            .map_err(|e| anyhow::anyhow!(e))?;
        assert!(config.is_mla(), "ModelV2 requires MLA attention (kind='mla')");

        let num_layers = config.num_layers();
        let weights = BackboneWeights::load_with_layers(weights_dir, num_layers)?;

        let mla_config = MlaDecodeConfig::from_infer_config(&config);

        // Build YaRN RoPE tables
        let rope_dim = mla_config.qk_rope_head_dim;
        let max_seq = config.model.max_position_embeddings.min(4096);
        let rope = if let Some(ref scaling) = config.attention.rope_scaling {
            YarnRopeTables::build(max_seq, rope_dim, config.rope_theta(), scaling)
        } else {
            // Standard RoPE (shouldn't happen for V2-Lite, but handle it)
            let dummy = crate::config::RopeScalingSection {
                factor: 1.0,
                original_max_position_embeddings: max_seq,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.0,
                mscale_all_dim: 1.0,
            };
            YarnRopeTables::build(max_seq, rope_dim, config.rope_theta(), &dummy)
        };

        // Compute attention scale
        let mscale_coeff = config.attention.rope_scaling.as_ref()
            .map(|s| s.mscale).unwrap_or(1.0);
        let factor = config.attention.rope_scaling.as_ref()
            .map(|s| s.factor).unwrap_or(1.0);
        let mscale = compute_mscale(mscale_coeff, factor);
        let attn_scale = mla_attention_scale(
            mla_config.qk_nope_head_dim,
            mla_config.qk_rope_head_dim,
            mscale,
        );
        eprintln!("MLA attn_scale={attn_scale:.6} (mscale={mscale:.4})");

        // Decide: pre-convert all layers (small models) or lazy per-layer (large models).
        // Use lazy mode (pread) for any model with >256 experts or >64 heads.
        // K2 has 64 heads but 384 experts (524 GB) — must use pread, not mmap.
        let lazy_mode = config.num_q_heads() >= 64 || config.num_experts() > 256;
        let mut layers_f32 = Vec::new();
        if !lazy_mode {
            let t = std::time::Instant::now();
            for layer in 0..num_layers {
                layers_f32.push(convert_layer_f32(&weights, &config, layer)?);
            }
            eprintln!("Pre-converted weights to f32: {:.1}s", t.elapsed().as_secs_f64());
        } else {
            eprintln!("Lazy weight mode: f16→f32 per layer on the fly (~2 GB vs ~54 GB)");
        }
        let lm_head_f32: Vec<f32> = weights.lm_head()?.iter().map(|v| v.to_f32()).collect();

        // Metal setup
        let mut metal = MetalDequantGemv::new();
        let mut expert_metal_bufs = std::collections::HashMap::new();
        if let Some(ref mut m) = metal {
            eprintln!("Metal GPU: enabled");
            m.init_scratch(
                config.hidden_size(),
                config.moe_inter_size(),
                config.num_experts_per_tok(),
            );
            if !lazy_mode {
                // Small models: wrap all expert mmaps (fits in memory)
                for layer in 0..num_layers {
                    if let Some(mmap) = weights.expert_mmap(layer) {
                        let buf = m.wrap_mmap(mmap);
                        expert_metal_bufs.insert(layer, buf);
                    }
                }
                eprintln!("Metal GPU: {} layers wrapped (mmap)", expert_metal_bufs.len());
            } else {
                // Large models: skip mmap wrapping, use pread-based loading below
                eprintln!("Metal GPU: expert pager mode (pread on demand)");
            }
        }

        // Expert pager: pread-based loading for large models (avoids 348 GB mmap thrashing)
        let mut expert_loaders = std::collections::HashMap::new();
        let hidden = config.hidden_size();
        let moe_inter = config.moe_inter_size();
        let group_size = config.quantization.group_size;

        // Compute expert stride (same calculation as moe_ffn_v2)
        let gu_packed = moe_inter * hidden / 2;
        let gu_groups = moe_inter * (hidden / group_size);
        let gu_scales = gu_groups * 2;
        let gu_total = gu_packed + gu_scales * 2;
        let dn_packed = hidden * moe_inter / 2;
        let dn_groups = hidden * (moe_inter / group_size);
        let dn_scales = dn_groups * 2;
        let dn_total = dn_packed + dn_scales * 2;
        let expert_stride = gu_total * 2 + dn_total;

        // Pre-allocate staging buffer for top-k experts
        let top_k = config.num_experts_per_tok();
        let expert_staging = vec![0u8; top_k * expert_stride];

        if lazy_mode {
            // Open file handles for each expert layer (pread-based, no mmap wrapping)
            let layout = expert_pager::ExpertFileLayout {
                expert_size: expert_stride,
                num_experts: config.num_experts(),
            };
            let n_experts = config.num_experts();
            let expected_file_size = expert_stride * n_experts;
            for layer in 0..num_layers {
                let expert_path = weights_dir.join(format!("layer_{layer:02}_experts.bin"));
                if expert_path.exists() {
                    // Validate expert file size matches config
                    let file_size = std::fs::metadata(&expert_path)
                        .map(|m| m.len() as usize).unwrap_or(0);
                    if file_size != expected_file_size {
                        eprintln!("WARNING: layer {layer:02} expert file is {} bytes, expected {} ({n_experts} experts × {expert_stride} bytes)",
                            file_size, expected_file_size);
                    }
                    let loader = ExpertLoader::open(expert_path.to_str().unwrap(), layout.clone())
                        .map_err(|e| anyhow::anyhow!("open expert file layer {layer}: {e}"))?;
                    expert_loaders.insert(layer, loader);
                }
            }
            eprintln!("Expert pager: {} layers with pread loaders, staging={:.1} MB",
                expert_loaders.len(), expert_staging.len() as f64 / 1e6);
        }

        // Pre-wrap staging buffer with Metal (created once, content updated per dispatch)
        let expert_staging_metal = if !expert_staging.is_empty() {
            metal.as_ref().map(|m| m.wrap_mmap(&expert_staging))
        } else {
            None
        };

        // Expert pool: cache hot experts in RAM to avoid pread on repeated access.
        // K2 profiling shows pread is 71-84% of layer time. Pool eliminates hits entirely.
        let (expert_pool, pool_buffers) = if lazy_mode && expert_stride > 0 {
            // Pool capacity: fit as many experts as RAM allows.
            // Each slot = expert_stride bytes (~23 MB for K2).
            // Budget: available RAM after backbone + staging + KV cache.
            // Conservative: use ~60% of available memory for pool.
            // Pool capacity: use most of available RAM but LAZY allocate.
            // Pre-allocating 91 GB at once evicts OS page cache — disaster.
            // Lazy: start with empty Vecs, allocate on first miss per slot.
            let pool_cap_env = std::env::var("RUSTANE_POOL_CAP")
                .ok().and_then(|v| v.parse::<usize>().ok());
            // Default 0: pool write-back is disabled, so pool tracking adds ~0.3ms/layer
            // of dead HashMap overhead with no caching benefit. RUSTANE_POOL_CAP=N re-enables.
            let pool_cap = pool_cap_env.unwrap_or(0);
            if pool_cap == 0 {
                eprintln!("ExpertPool: DISABLED (RUSTANE_POOL_CAP=0)");
                (None, Vec::new())
            } else {
                // Lazy: allocate empty Vecs (0 bytes each). Grow on first use.
                let bufs: Vec<Vec<u8>> = (0..pool_cap).map(|_| Vec::new()).collect();
                eprintln!("ExpertPool: {} slots × {:.1} MB = {:.1} GB",
                    pool_cap, expert_stride as f64 / 1e6,
                    pool_cap as f64 * expert_stride as f64 / 1e9);
                (Some(expert_pager::ExpertPool::new(pool_cap)), bufs)
            }
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            weights, config, mla_config, rope, attn_scale,
            metal, expert_metal_bufs, layers_f32, lm_head_f32,
            expert_loaders, expert_stride, expert_staging, expert_staging_metal,
            expert_pool: std::cell::UnsafeCell::new(expert_pool),
            pool_metal_bufs: std::cell::UnsafeCell::new(vec![None; pool_buffers.len()]),
            pool_buffers: std::cell::UnsafeCell::new(pool_buffers),
        })
    }
}

/// Fill dst Vec from f16 src using parallel bulk NEON conversion.
/// Reuses existing Vec capacity (zero allocs after first call).
/// Uses half's convert_to_f32_slice (FCVTL: 4 f16→4 f32 per instruction)
/// instead of per-element to_f32 (scalar fcvt + runtime detection overhead).
#[inline]
fn fill_f32(dst: &mut Vec<f32>, src: &[f16]) {
    use rayon::prelude::*;
    use half::slice::HalfFloatSliceExt;
    let n = src.len();
    dst.clear();
    if dst.capacity() < n {
        dst.reserve(n - dst.capacity());
    }
    unsafe { dst.set_len(n); }

    const PAR_THRESHOLD: usize = 500_000; // ~1 MB f16 = worth parallelizing
    if n >= PAR_THRESHOLD {
        // Parallel conversion: each rayon thread uses vectorized FCVTL on its chunk
        dst.par_chunks_mut(256 * 1024).enumerate().for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * 256 * 1024;
            src[base..base + chunk.len()].convert_to_f32_slice(chunk);
        });
    } else {
        src.convert_to_f32_slice(dst.as_mut_slice());
    }
}

/// Fill optional Vec from optional f16 src using SIMD-friendly conversion.
/// When src is None, sets dst to None.
#[inline]
fn fill_f32_opt(dst: &mut Option<Vec<f32>>, src: Option<&[f16]>) {
    match src {
        Some(data) => {
            let vec = dst.get_or_insert_with(Vec::new);
            fill_f32(vec, data);
        }
        None => {
            *dst = None;
        }
    }
}

/// Fill Vec from f32 src (norms, biases — already f32, just copy).
#[inline]
fn fill_copy(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.reserve(src.len().saturating_sub(dst.capacity()));
    dst.extend_from_slice(src);
}

/// Fill optional Vec from optional f32 src.
#[inline]
fn fill_copy_opt(dst: &mut Option<Vec<f32>>, src: Option<&[f32]>) {
    match src {
        Some(data) => {
            let vec = dst.get_or_insert_with(Vec::new);
            vec.clear();
            vec.reserve(data.len().saturating_sub(vec.capacity()));
            vec.extend_from_slice(data);
        }
        None => {
            *dst = None;
        }
    }
}

/// Convert one layer's weights INTO a pre-allocated buffer (zero allocs after warmup).
/// This is the hot path — reuses Vec capacity from previous layers.
fn convert_layer_into(buf: &mut MlaLayerF32, weights: &BackboneWeights, config: &InferConfig, layer: usize) -> Result<()> {
    let is_moe = config.is_moe_layer(layer);
    let lw = weights.mla_layer_weights(layer, is_moe)?;

    // Always-present fields: reuse capacity
    fill_f32(&mut buf.q_proj, lw.q_proj);
    fill_f32(&mut buf.kv_a_proj, lw.kv_a_proj);
    fill_copy(&mut buf.kv_a_layernorm, lw.kv_a_layernorm);
    fill_f32(&mut buf.w_uk, lw.w_uk);
    fill_f32(&mut buf.w_uv, lw.w_uv);
    fill_f32(&mut buf.o_proj, lw.o_proj);
    fill_copy(&mut buf.input_norm, lw.input_norm);
    fill_copy(&mut buf.post_attn_norm, lw.post_attn_norm);

    // Optional fields: reuse capacity where possible
    fill_f32_opt(&mut buf.q_a_proj, lw.q_a_proj);
    fill_copy_opt(&mut buf.q_a_layernorm, lw.q_a_layernorm);
    fill_f32_opt(&mut buf.q_b_proj, lw.q_b_proj);
    fill_f32_opt(&mut buf.router, lw.router);
    fill_copy_opt(&mut buf.e_score_correction_bias, lw.e_score_correction_bias);
    fill_f32_opt(&mut buf.shared_gate, lw.shared_gate_proj);
    fill_f32_opt(&mut buf.shared_up, lw.shared_up_proj);
    fill_f32_opt(&mut buf.shared_down, lw.shared_down_proj);
    fill_f32_opt(&mut buf.dense_gate, lw.dense_gate_proj);
    fill_f32_opt(&mut buf.dense_up, lw.dense_up_proj);
    fill_f32_opt(&mut buf.dense_down, lw.dense_down_proj);

    Ok(())
}

/// Allocating version for pre-conversion at load time (V2-Lite path).
fn convert_layer_f32(weights: &BackboneWeights, config: &InferConfig, layer: usize) -> Result<MlaLayerF32> {
    let mut buf = MlaLayerF32::empty();
    convert_layer_into(&mut buf, weights, config, layer)?;
    Ok(buf)
}

/// Per-layer timing breakdown (ms).
#[derive(Default)]
struct LayerTiming {
    convert_ms: f64,
    attn_ms: f64,
    ffn_ms: f64,
}

/// Construct MLA attention weight borrows from pre-converted f32 layer.
fn make_attn_weights(lf: &MlaLayerF32) -> MlaAttnWeights<'_> {
    MlaAttnWeights {
        q_proj: &lf.q_proj,
        q_a_proj: lf.q_a_proj.as_deref(),
        q_a_layernorm: lf.q_a_layernorm.as_deref(),
        q_b_proj: lf.q_b_proj.as_deref(),
        kv_a_proj: &lf.kv_a_proj,
        kv_a_layernorm: &lf.kv_a_layernorm,
        w_uk: &lf.w_uk,
        w_uv: &lf.w_uv,
        o_proj: &lf.o_proj,
        input_norm: &lf.input_norm,
        post_attn_norm: &lf.post_attn_norm,
    }
}

/// Run one layer's compute given pre-converted f32 weights.
/// This is the core compute path — no conversion, no allocation of weight buffers.
fn run_layer_compute(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    pos: usize,
    lf: &MlaLayerF32,
) -> Result<Vec<f32>> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();

    // 1. RMSNorm → MLA Attention → Residual
    let normed = rmsnorm(x, &lf.input_norm, eps);
    let attn_weights = make_attn_weights(lf);

    let attn_out = mla_forward_decode(
        &normed, &attn_weights, cache, layer, pos,
        &model.rope, &model.mla_config, model.attn_scale,
    );

    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden {
        residual[d] = x[d] + attn_out[d];
    }

    // 2. RMSNorm → FFN → Residual
    let normed2 = rmsnorm(&residual, &lf.post_attn_norm, eps);

    let ffn_out = if model.config.is_moe_layer(layer) {
        moe_ffn_v2(model, router, layer, &normed2, lf)?
    } else {
        dense_ffn(&normed2, lf)
    };

    for d in 0..hidden {
        residual[d] += ffn_out[d];
    }

    Ok(residual)
}

/// MLA attention only — returns (residual_after_attn, normed2_for_ffn).
/// Separated from FFN so the decode loop can defer convert(N+1) to overlap with
/// FFN instead of MLA, avoiding memory bandwidth contention during sgemv calls.
fn run_mla_only(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    layer: usize,
    x: &[f32],
    pos: usize,
    lf: &MlaLayerF32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();

    let normed = rmsnorm(x, &lf.input_norm, eps);
    let attn_weights = make_attn_weights(lf);

    let attn_out = mla_forward_decode(
        &normed, &attn_weights, cache, layer, pos,
        &model.rope, &model.mla_config, model.attn_scale,
    );

    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden { residual[d] = x[d] + attn_out[d]; }

    let normed2 = rmsnorm(&residual, &lf.post_attn_norm, eps);
    Ok((residual, normed2))
}

/// FFN only — adds FFN output into residual in place.
fn run_ffn_only(
    model: &ModelV2,
    router: &mut MoeRouter,
    layer: usize,
    normed2: &[f32],
    residual: &mut [f32],
    lf: &MlaLayerF32,
) -> Result<()> {
    let hidden = model.config.hidden_size();
    let ffn_out = if model.config.is_moe_layer(layer) {
        moe_ffn_v2(model, router, layer, normed2, lf)?
    } else {
        dense_ffn(normed2, lf)
    };
    for d in 0..hidden { residual[d] += ffn_out[d]; }
    Ok(())
}

/// Run one layer of the V2 model (MLA attention + FFN).
/// If `timing` is Some, accumulates per-phase timing.
fn run_layer_v2(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    pos: usize,
    mut timing: Option<&mut LayerTiming>,
) -> Result<Vec<f32>> {
    // Lazy conversion: use pre-converted if available, else convert on the fly
    let t_conv = std::time::Instant::now();
    let lazy_lf;
    let lf = if layer < model.layers_f32.len() {
        &model.layers_f32[layer]
    } else {
        lazy_lf = convert_layer_f32(&model.weights, &model.config, layer)?;
        &lazy_lf
    };
    if let Some(ref mut t) = timing {
        t.convert_ms = t_conv.elapsed().as_secs_f64() * 1000.0;
    }

    let t_attn = std::time::Instant::now();
    let result = run_layer_compute(model, cache, router, layer, x, pos, lf)?;

    // Split timing: approximate attn vs ffn (run_layer_compute handles both)
    if let Some(ref mut t) = timing {
        // Total compute time (attn + ffn combined)
        let compute_ms = t_attn.elapsed().as_secs_f64() * 1000.0;
        t.attn_ms = compute_ms; // report as attn for now (combined)
        t.ffn_ms = 0.0;
    }

    Ok(result)
}

/// Run one layer using f16 weights directly from backbone mmap — NO conversion pass.
/// Uses sgemv_f16 (chunked convert+AMX) for half the memory traffic.
fn run_layer_f16(
    model: &ModelV2,
    cache: &mut MlaKvCache,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    pos: usize,
) -> Result<Vec<f32>> {
    let hidden = model.config.hidden_size();
    let eps = model.config.rms_norm_eps();

    let is_moe = model.config.is_moe_layer(layer);
    let lw = model.weights.mla_layer_weights(layer, is_moe)?;

    // 1. RMSNorm → MLA Attention (f16 weights) → Residual
    let normed = rmsnorm(x, lw.input_norm, eps);
    let attn_out = crate::mla_attention::mla_forward_decode_f16(
        &normed, &lw, cache, layer, pos,
        &model.rope, &model.mla_config, model.attn_scale,
    );

    let mut residual = vec![0.0f32; hidden];
    for d in 0..hidden { residual[d] = x[d] + attn_out[d]; }

    // 2. RMSNorm → FFN → Residual
    let normed2 = rmsnorm(&residual, lw.post_attn_norm, eps);

    let ffn_out = if is_moe {
        // Need MlaLayerF32 for router/shared/dense weights + MoE dispatch
        // Router and shared FFN use sgemv_f16 directly
        moe_ffn_f16(model, router, layer, &normed2, &lw)?
    } else {
        dense_ffn_f16(&normed2, &lw)
    };

    for d in 0..hidden { residual[d] += ffn_out[d]; }
    Ok(residual)
}

/// Dense FFN with f16 weights: SiLU(x @ gate^T) * (x @ up^T) → @ down^T
fn dense_ffn_f16(x: &[f32], lw: &crate::weights::MlaLayerWeights) -> Vec<f32> {
    let gate_w = lw.dense_gate_proj.expect("dense gate");
    let up_w = lw.dense_up_proj.expect("dense up");
    let down_w = lw.dense_down_proj.expect("dense down");

    let hidden = x.len();
    let inter = gate_w.len() / hidden;

    let mut gate_out = vec![0.0f32; inter];
    let mut up_out = vec![0.0f32; inter];
    crate::blas::sgemv_f16(gate_w, x, &mut gate_out, inter, hidden);
    crate::blas::sgemv_f16(up_w, x, &mut up_out, inter, hidden);

    for i in 0..inter {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gate_out[i] = silu * up_out[i];
    }

    let mut out = vec![0.0f32; hidden];
    crate::blas::sgemv_f16(down_w, &gate_out, &mut out, hidden, inter);
    out
}

/// MoE FFN with f16 shared expert weights.
fn moe_ffn_f16(
    model: &ModelV2,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    lw: &crate::weights::MlaLayerWeights,
) -> Result<Vec<f32>> {
    let hidden = model.config.hidden_size();

    // Router (f16 weights)
    let router_w = lw.router.expect("MoE router");
    let num_experts = model.config.num_experts();
    let mut gate_logits = vec![0.0f32; num_experts];
    crate::blas::sgemv_f16(router_w, x, &mut gate_logits, num_experts, hidden);

    let route = if model.config.ffn.scoring_func == "sigmoid" {
        if let Some(bias) = lw.e_score_correction_bias {
            moe_router::route_sigmoid_v3(
                &gate_logits, bias,
                model.config.ffn.n_group, model.config.ffn.topk_group,
                model.config.num_experts_per_tok(), 1.0,
            )
        } else {
            router.route(&gate_logits)
        }
    } else {
        router.route_softmax(&gate_logits)
    };

    // Shared expert FFN (f16 weights)
    let mut combined = if let (Some(sg), Some(su), Some(sd)) =
        (lw.shared_gate_proj, lw.shared_up_proj, lw.shared_down_proj)
    {
        let inter = sg.len() / hidden;
        let mut gate_out = vec![0.0f32; inter];
        let mut up_out = vec![0.0f32; inter];
        crate::blas::sgemv_f16(sg, x, &mut gate_out, inter, hidden);
        crate::blas::sgemv_f16(su, x, &mut up_out, inter, hidden);
        for i in 0..inter {
            let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
            gate_out[i] = silu * up_out[i];
        }
        let mut out = vec![0.0f32; hidden];
        crate::blas::sgemv_f16(sd, &gate_out, &mut out, hidden, inter);
        out
    } else {
        vec![0.0f32; hidden]
    };

    // Routed experts (same pread+Metal dispatch as moe_ffn_v2)
    let routed_scale = model.config.ffn.routed_scaling_factor;
    let moe_inter = model.config.moe_inter_size();
    let group_size = model.config.quantization.group_size;

    let gu_packed = moe_inter * hidden / 2;
    let gu_groups = moe_inter * (hidden / group_size);
    let gu_scales = gu_groups * 2;
    let gu_total = gu_packed + gu_scales * 2;
    let dn_packed = hidden * moe_inter / 2;
    let dn_groups = hidden * (moe_inter / group_size);
    let dn_scales = dn_groups * 2;
    let dn_total = dn_packed + dn_scales * 2;
    let expert_stride = gu_total * 2 + dn_total;

    if let Some(m) = model.metal.as_ref() {
        if let Some(loader) = model.expert_loaders.get(&layer) {
            let staging = &model.expert_staging;
            let staging_ptr = staging.as_ptr() as *mut u8;

            let expert_ids: Vec<(usize, usize, f32)> = route.expert_ids.iter()
                .zip(route.weights.iter()).enumerate()
                .map(|(i, (&eid, &w))| (i, eid, w)).collect();

            use rayon::prelude::*;
            let staging_mut = unsafe { std::slice::from_raw_parts_mut(staging_ptr, staging.len()) };
            staging_mut[..expert_ids.len() * expert_stride]
                .chunks_mut(expert_stride)
                .zip(expert_ids.iter())
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|(chunk, &(_, eid, _))| {
                    loader.load_expert(eid as u32, chunk).unwrap();
                });

            let mut fused_ops = Vec::new();
            let mut down_ops = Vec::new();
            let mut routing_weights = Vec::new();
            for &(i, _, weight) in &expert_ids {
                let p = i * expert_stride;
                fused_ops.push(FusedGateUpSiluOp {
                    gate_packed_offset: p, gate_scales_offset: p + gu_packed,
                    gate_zeros_offset: p + gu_packed + gu_scales,
                    up_packed_offset: p + gu_total, up_scales_offset: p + gu_total + gu_packed,
                    up_zeros_offset: p + gu_total + gu_packed + gu_scales,
                    out_features: moe_inter, in_features: hidden, group_size,
                });
                down_ops.push(ExpertGemvOp {
                    packed_offset: p + 2 * gu_total, scales_offset: p + 2 * gu_total + dn_packed,
                    zeros_offset: p + 2 * gu_total + dn_packed + dn_scales,
                    out_features: hidden, in_features: moe_inter, group_size,
                });
                routing_weights.push(weight);
            }

            if !routing_weights.is_empty() {
                let metal_buf = model.expert_staging_metal.as_ref().unwrap();
                let down_results = m.fused_and_down_single_cmdbuf(metal_buf, &fused_ops, &down_ops, x);
                for (i, weight) in routing_weights.iter().enumerate() {
                    let sw = weight * routed_scale;
                    for d in 0..hidden { combined[d] += sw * down_results[i][d]; }
                }
            }
        }
    }
    Ok(combined)
}

/// Dense FFN: SiLU(x @ gate^T) * (x @ up^T) → @ down^T
fn dense_ffn(x: &[f32], lf: &MlaLayerF32) -> Vec<f32> {
    let gate_w = lf.dense_gate.as_ref().expect("dense layer needs gate_proj");
    let up_w = lf.dense_up.as_ref().expect("dense layer needs up_proj");
    let down_w = lf.dense_down.as_ref().expect("dense layer needs down_proj");

    let hidden = x.len();
    let inter = gate_w.len() / hidden;

    let mut gate_out = vec![0.0f32; inter];
    let mut up_out = vec![0.0f32; inter];
    crate::blas::sgemv_f32(gate_w, x, &mut gate_out, inter, hidden);
    crate::blas::sgemv_f32(up_w, x, &mut up_out, inter, hidden);

    // SiLU(gate) * up
    for i in 0..inter {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gate_out[i] = silu * up_out[i];
    }

    // down_proj
    let mut out = vec![0.0f32; hidden];
    crate::blas::sgemv_f32(down_w, &gate_out, &mut out, hidden, inter);
    out
}

/// Shared expert FFN (same SwiGLU as dense, but using shared expert weights).
fn shared_expert_ffn(x: &[f32], lf: &MlaLayerF32) -> Vec<f32> {
    let gate_w = lf.shared_gate.as_ref().expect("shared gate");
    let up_w = lf.shared_up.as_ref().expect("shared up");
    let down_w = lf.shared_down.as_ref().expect("shared down");

    let hidden = x.len();
    let inter = gate_w.len() / hidden;

    let mut gate_out = vec![0.0f32; inter];
    let mut up_out = vec![0.0f32; inter];
    crate::blas::sgemv_f32(gate_w, x, &mut gate_out, inter, hidden);
    crate::blas::sgemv_f32(up_w, x, &mut up_out, inter, hidden);

    for i in 0..inter {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gate_out[i] = silu * up_out[i];
    }

    let mut out = vec![0.0f32; hidden];
    crate::blas::sgemv_f32(down_w, &gate_out, &mut out, hidden, inter);
    out
}

/// MoE FFN: shared experts + routed experts
fn moe_ffn_v2(
    model: &ModelV2,
    router: &mut MoeRouter,
    layer: usize,
    x: &[f32],
    lf: &MlaLayerF32,
) -> Result<Vec<f32>> {
    let hidden = model.config.hidden_size();

    // FFN profiling (env-gated: RUSTANE_FFN_PROFILE=1)
    let ffn_profile = std::env::var("RUSTANE_FFN_PROFILE").is_ok();
    let t_ffn_start = std::time::Instant::now();

    // 1. Router
    let router_w = lf.router.as_ref().expect("MoE layer needs router");
    let num_experts = model.config.num_experts();
    let mut gate_logits = vec![0.0f32; num_experts];
    crate::blas::sgemv_f32(router_w, x, &mut gate_logits, num_experts, hidden);
    let t_after_router = std::time::Instant::now();

    let route = if model.config.ffn.scoring_func == "sigmoid" {
        if let Some(ref bias) = lf.e_score_correction_bias {
            // V3: grouped sigmoid routing with frozen bias
            moe_router::route_sigmoid_v3(
                &gate_logits,
                bias,
                model.config.ffn.n_group,
                model.config.ffn.topk_group,
                model.config.num_experts_per_tok(),
                1.0,  // scaling applied separately in weight accumulation
            )
        } else {
            router.route(&gate_logits)
        }
    } else {
        router.route_softmax(&gate_logits)
    };

    // Pool simulation: track which experts are routed (decode-phase only)
    POOL_SIM.with(|p| {
        if let Some(pool) = p.borrow_mut().as_mut() {
            for &eid in &route.expert_ids {
                pool.request(layer as u32, eid as u32);
            }
        }
    });

    // 2. Shared experts (computed eagerly on non-pread path, overlapped with pread on pread path)
    let use_pread = model.expert_loaders.contains_key(&layer);
    let mut combined = if !use_pread {
        // Non-pread path: compute shared FFN eagerly (no overlap opportunity)
        if lf.shared_gate.is_some() { shared_expert_ffn(x, lf) } else { vec![0.0f32; hidden] }
    } else {
        vec![0.0f32; hidden]  // placeholder — filled by thread::scope in pread path below
    };

    let routed_scale = model.config.ffn.routed_scaling_factor;

    // 3. Routed experts (4-bit quantized, Metal GPU or CPU)
    let moe_inter = model.config.moe_inter_size();
    let group_size = model.config.quantization.group_size;

    let gu_packed = moe_inter * hidden / 2;
    let gu_groups = moe_inter * (hidden / group_size);
    let gu_scales = gu_groups * 2;
    let gu_total = gu_packed + gu_scales * 2;
    let dn_packed = hidden * moe_inter / 2;
    let dn_groups = hidden * (moe_inter / group_size);
    let dn_scales = dn_groups * 2;
    let dn_total = dn_packed + dn_scales * 2;
    let expert_stride = gu_total * 2 + dn_total;

    if let Some(m) = model.metal.as_ref() {
        if use_pread {
            // === PREAD PATH: overlap expert pread with shared FFN ===
            let loader = &model.expert_loaders[&layer];
            let staging = &model.expert_staging;

            // SAFETY: single-threaded access (pread writes to non-overlapping regions)
            let staging_ptr = staging.as_ptr() as *mut u8;

            let expert_ids: Vec<(usize, usize, f32)> = route.expert_ids.iter()
                .zip(route.weights.iter())
                .enumerate()
                .map(|(i, (&eid, &w))| (i, eid, w))
                .collect();

            // Overlap: pread experts from SSD while CPU computes shared FFN.
            // Saves ~min(pread_ms, shared_ffn_ms) per MoE layer.
            use rayon::prelude::*;
            let staging_mut = unsafe { std::slice::from_raw_parts_mut(staging_ptr, staging.len()) };
            let pread_region = &mut staging_mut[..expert_ids.len() * expert_stride];

            // Expert pool: check cache before pread. Copy hits to staging, track misses.
            let mut need_pread: Vec<bool> = vec![true; expert_ids.len()];
            let mut pool_hits = 0usize;
            {
                // SAFETY: single-threaded access during decode. Pool is not accessed from thread::scope.
                let pool_opt = unsafe { &mut *model.expert_pool.get() };
                let pool_bufs = unsafe { &mut *model.pool_buffers.get() };
                if let Some(pool) = pool_opt.as_mut() {
                    for &(i, eid, _) in &expert_ids {
                        let (slot, is_hit) = pool.request(layer as u32, eid as u32);
                        if is_hit && !pool_bufs[slot].is_empty() {
                            // Cache hit AND buffer allocated: copy from pool to staging
                            let staging_offset = i * expert_stride;
                            pread_region[staging_offset..staging_offset + expert_stride]
                                .copy_from_slice(&pool_bufs[slot][..expert_stride]);
                            need_pread[i] = false;
                            pool_hits += 1;
                        }
                        // If hit but buffer empty: treat as miss (lazy alloc on write-back)
                    }
                }
            }

            // Overlap pread (misses only) with shared FFN via thread::scope.
            let t_before_scope = std::time::Instant::now();
            let need_pread_ref = &need_pread;
            let shared_result = std::thread::scope(|s| {
                let pread_handle = s.spawn(|| {
                    let t_pread = std::time::Instant::now();
                    pread_region
                        .chunks_mut(expert_stride)
                        .zip(expert_ids.iter())
                        .collect::<Vec<_>>()
                        .into_par_iter()
                        .for_each(|(chunk, &(i, eid, _))| {
                            if need_pread_ref[i] {
                                loader.load_expert(eid as u32, chunk).unwrap();
                            }
                        });
                    t_pread.elapsed()
                });

                let t_shared = std::time::Instant::now();
                let result = if lf.shared_gate.is_some() {
                    shared_expert_ffn(x, lf)
                } else {
                    vec![0.0f32; hidden]
                };
                let shared_elapsed = t_shared.elapsed();

                let pread_elapsed = pread_handle.join().unwrap();
                if ffn_profile && layer <= 3 {
                    eprintln!("  FFN L{layer:02} pread: {:.0}µs  shared_ffn: {:.0}µs  (overlapped)",
                        pread_elapsed.as_micros(), shared_elapsed.as_micros());
                }
                result
            });
            let t_after_scope = std::time::Instant::now();
            combined = shared_result;

            // Pool write-back: cache pread'd data so future tokens hit.
            // Gated by RUSTANE_POOL_WRITEBACK=1. Default: OFF.
            // Previous attempt with 3000 slots (69 GB) caused page cache eviction.
            // Use with small POOL_CAP (100-500) to stay under ~12 GB.
            if pool_writeback_enabled() {
                let pool_opt = unsafe { &*model.expert_pool.get() };
                let pool_bufs = unsafe { &mut *model.pool_buffers.get() };
                let pool_metal_wb = unsafe { &mut *model.pool_metal_bufs.get() };
                if let Some(pool) = pool_opt.as_ref() {
                    for &(i, eid, _) in &expert_ids {
                        if need_pread[i] {
                            if let Some(slot) = pool.entries_slot(layer as u32, eid as u32) {
                                if pool_bufs[slot].is_empty() {
                                    pool_bufs[slot] = vec![0u8; expert_stride];
                                    pool_metal_wb[slot] = None; // invalidate stale Metal buf
                                }
                                let staging_offset = i * expert_stride;
                                pool_bufs[slot][..expert_stride]
                                    .copy_from_slice(&pread_region[staging_offset..staging_offset + expert_stride]);
                            }
                        }
                    }
                }
            }

            if ffn_profile && layer <= 3 {
                eprintln!("  FFN L{layer:02} pool: {pool_hits} hits / {} total",
                    expert_ids.len());
            }

            // Build Metal ops with packed offsets
            let mut fused_ops = Vec::new();
            let mut down_ops = Vec::new();
            let mut routing_weights = Vec::new();

            for &(i, _eid, weight) in &expert_ids {
                let pack_offset = i * expert_stride;

                fused_ops.push(FusedGateUpSiluOp {
                    gate_packed_offset: pack_offset,
                    gate_scales_offset: pack_offset + gu_packed,
                    gate_zeros_offset: pack_offset + gu_packed + gu_scales,
                    up_packed_offset: pack_offset + gu_total,
                    up_scales_offset: pack_offset + gu_total + gu_packed,
                    up_zeros_offset: pack_offset + gu_total + gu_packed + gu_scales,
                    out_features: moe_inter, in_features: hidden, group_size,
                });

                down_ops.push(ExpertGemvOp {
                    packed_offset: pack_offset + 2 * gu_total,
                    scales_offset: pack_offset + 2 * gu_total + dn_packed,
                    zeros_offset: pack_offset + 2 * gu_total + dn_packed + dn_scales,
                    out_features: hidden, in_features: moe_inter, group_size,
                });

                routing_weights.push(weight);
            }

            if !routing_weights.is_empty() {
                let t_metal = std::time::Instant::now();

                // Zero-copy dispatch: each expert reads from its own buffer
                // Pool hits → pool Metal buffer (zero copy, no staging)
                // Pool misses → staging Metal buffer (pread'd data)
                let pool_metal = unsafe { &mut *model.pool_metal_bufs.get() };
                let pool_bufs_raw = unsafe { &*model.pool_buffers.get() };
                let pool_opt = unsafe { &*model.expert_pool.get() };
                let staging_metal = model.expert_staging_metal.as_ref().unwrap();

                let mut use_per_expert = false;

                // Pre-pass: ensure Metal buffers exist for all pool hits
                if let Some(pool) = pool_opt.as_ref() {
                    for &(i, eid, _) in &expert_ids {
                        if !need_pread[i] {
                            if let Some(slot) = pool.entries_slot(layer as u32, eid as u32) {
                                if pool_metal[slot].is_none() && !pool_bufs_raw[slot].is_empty() {
                                    pool_metal[slot] = Some(m.wrap_mmap(&pool_bufs_raw[slot]));
                                }
                            }
                        }
                    }
                }

                // Build per-expert buffer list
                let mut per_expert_bufs: Vec<&ProtocolObject<dyn MTLBuffer>> = Vec::with_capacity(expert_ids.len());
                if let Some(pool) = pool_opt.as_ref() {
                    for &(i, eid, _) in &expert_ids {
                        if !need_pread[i] {
                            if let Some(slot) = pool.entries_slot(layer as u32, eid as u32) {
                                if let Some(ref mbuf) = pool_metal[slot] {
                                    per_expert_bufs.push(mbuf.as_ref());
                                    use_per_expert = true;
                                    continue;
                                }
                            }
                        }
                        per_expert_bufs.push(staging_metal.as_ref());
                    }
                }

                let down_results = if use_per_expert {
                    // Mix of pool buffers + staging: use per-expert dispatch
                    // Adjust offsets: pool hits use offset 0, staging misses use i*expert_stride
                    let mut adj_fused = fused_ops.clone();
                    let mut adj_down = down_ops.clone();
                    if let Some(pool) = pool_opt.as_ref() {
                        for (idx, &(i, eid, _)) in expert_ids.iter().enumerate() {
                            if !need_pread[i] {
                                if let Some(_slot) = pool.entries_slot(layer as u32, eid as u32) {
                                    // Pool hit: offsets are relative to expert start (0-based in pool buf)
                                    // The FusedGateUpSiluOp offsets were built as i*expert_stride
                                    // Subtract i*expert_stride to make them 0-based
                                    let base = i * expert_stride;
                                    adj_fused[idx].gate_packed_offset -= base;
                                    adj_fused[idx].gate_scales_offset -= base;
                                    adj_fused[idx].gate_zeros_offset -= base;
                                    adj_fused[idx].up_packed_offset -= base;
                                    adj_fused[idx].up_scales_offset -= base;
                                    adj_fused[idx].up_zeros_offset -= base;
                                    adj_down[idx].packed_offset -= base;
                                    adj_down[idx].scales_offset -= base;
                                    adj_down[idx].zeros_offset -= base;
                                }
                            }
                        }
                    }
                    m.fused_and_down_per_expert_bufs(&per_expert_bufs, &adj_fused, &adj_down, x)
                } else {
                    // All misses or no pool: use original staging dispatch
                    m.fused_and_down_single_cmdbuf(staging_metal, &fused_ops, &down_ops, x)
                };
                let metal_elapsed = t_metal.elapsed();

                let t_accum = std::time::Instant::now();
                for (i, weight) in routing_weights.iter().enumerate() {
                    let scaled_weight = weight * routed_scale;
                    for d in 0..hidden {
                        combined[d] += scaled_weight * down_results[i][d];
                    }
                }
                let accum_elapsed = t_accum.elapsed();

                if ffn_profile && layer <= 3 {
                    let scope_ms = t_after_scope.duration_since(t_before_scope).as_micros();
                    eprintln!("  FFN L{layer:02} metal: {:.0}µs  accum: {:.0}µs  scope_total: {:.0}µs  LAYER_TOTAL: {:.0}µs",
                        metal_elapsed.as_micros(), accum_elapsed.as_micros(), scope_ms,
                        t_ffn_start.elapsed().as_micros());
                }
            }
        } else {
            // === MMAP PATH: original zero-copy Metal dispatch (small models) ===
            let expert_mmap = model.weights.expert_mmap(layer);
            let mmap_metal_buf = model.expert_metal_bufs.get(&layer).map(|b| b.as_ref());

            if let (Some(expert_data), Some(mmap_buf)) = (expert_mmap, mmap_metal_buf) {
                let mut fused_ops = Vec::new();
                let mut down_ops = Vec::new();
                let mut routing_weights = Vec::new();

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

                if !routing_weights.is_empty() {
                    let down_results = m.fused_and_down_single_cmdbuf(mmap_buf, &fused_ops, &down_ops, x);
                    for (i, weight) in routing_weights.iter().enumerate() {
                        let scaled_weight = weight * routed_scale;
                        for d in 0..hidden {
                            combined[d] += scaled_weight * down_results[i][d];
                        }
                    }
                }
            }
        }
    }

    Ok(combined)
}

/// Embed a token: extract row from f16 table, convert to f32.
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

/// Matrix-vector multiply using BLAS.
fn matvec_f32(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    crate::blas::sgemv_f32(w, x, &mut y, out_dim, in_dim);
    y
}

/// Generate tokens from a prompt using DeepSeek-V2 model.
pub fn generate_v2(
    model: &ModelV2,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &SamplingConfig,
) -> Result<GenerateV2Output> {
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Prepend BOS token if the tokenizer didn't add it
    let bos = model.config.model.bos_token_id;
    if input_ids.first() != Some(&bos) {
        input_ids.insert(0, bos);
    }

    if input_ids.is_empty() {
        bail!("empty prompt after tokenization");
    }

    let hidden = model.config.hidden_size();
    let vocab = model.config.vocab_size();
    let num_layers = model.config.num_layers();
    let max_seq = model.rope.max_seq;

    // Create MLA KV cache
    let mut cache = MlaKvCache::new(
        num_layers,
        model.mla_config.kv_lora_rank,
        model.mla_config.qk_rope_head_dim,
        max_seq,
    );
    eprintln!("MLA KV cache: {:.1} MB", cache.memory_bytes() as f64 / 1e6);

    // Create router
    let router_config = RouterConfig {
        num_experts: model.config.num_experts(),
        top_k: model.config.num_experts_per_tok(),
        norm_topk_prob: model.config.ffn.norm_topk_prob,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(router_config);

    let embed_table = model.weights.embed_table()?;
    let final_norm = model.weights.final_norm()?;

    let mut all_ids = input_ids.clone();
    let prompt_tokens = input_ids.len();

    let lazy_mode = model.layers_f32.is_empty();

    // For lazy mode: single reusable buffer, sequential convert+compute.
    // ONE MlaLayerF32 buffer is reused across all layers and tokens.
    // After warmup (~3 layers), Vec capacities are established → ZERO allocations.
    // Simple and eliminates the ~1.8 GB alloc/dealloc per layer that was the bottleneck.

    let (prefill_secs, decode_secs) = if lazy_mode {
        // TWO reusable buffers for pipeline: convert layer N+1 while computing layer N
        let mut buf_a = MlaLayerF32::empty();
        let mut buf_b = MlaLayerF32::empty();

        // Cache dense layers (0..first_k_dense_replace) permanently in f32.
        // Dense layers have large FFN weights ([18432,7168] × 3), making their
        // f16→f32 conversion ~10ms — longer than FFN compute (~7ms). This causes
        // the pipeline to stall waiting for conversion. Caching eliminates:
        // 1. Serial convert(0) at start of each token (~10ms)
        // 2. Convert stalls for dense-to-dense transitions (~3ms × 2)
        let num_dense = model.config.ffn.first_k_dense_replace;
        let mut cached_dense: Vec<MlaLayerF32> = (0..num_dense)
            .map(|_| MlaLayerF32::empty())
            .collect();

        // Warmup: pre-fault all backbone pages into page cache
        let t_warmup = std::time::Instant::now();
        for layer in 0..num_layers {
            if layer < num_dense {
                convert_layer_into(&mut cached_dense[layer], &model.weights, &model.config, layer)?;
            } else {
                convert_layer_into(&mut buf_a, &model.weights, &model.config, layer)?;
            }
        }
        eprintln!("Backbone warmup: {:.1}s (pre-faulted {} layers, cached {} dense)",
            t_warmup.elapsed().as_secs_f64(), num_layers, num_dense);

        // --- Prefill ---
        let t_prefill = std::time::Instant::now();
        for (i, &token_id) in input_ids.iter().enumerate() {
            let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
            let mut x = emb;
            for layer in 0..num_layers {
                let lf = if layer < num_dense {
                    &cached_dense[layer]
                } else {
                    convert_layer_into(&mut buf_a, &model.weights, &model.config, layer)?;
                    &buf_a
                };
                x = run_layer_compute(model, &mut cache, &mut router, layer, &x, i, lf)?;
            }
            cache.advance();
            if i == input_ids.len() - 1 {
                let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
                let logits = matvec_f32(&model.lm_head_f32, &normed, vocab, hidden);
                let next_token = sample(&logits, sampling, i as u64);
                all_ids.push(next_token);
            }
        }
        let prefill_secs = t_prefill.elapsed().as_secs_f64();

        // --- Decode (deferred-convert pipeline, double-buffered) ---
        // Key insight: MLA attention is memory-bandwidth-bound (sgemv reads ~750 MB f32).
        // Running f16→f32 conversion concurrently steals ~50% of bandwidth from sgemv.
        // Fix: run MLA attention FIRST with no background thread (full bandwidth),
        // then overlap convert(N+1) with FFN (which uses Metal GPU + SSD pread,
        // not competing for memory bandwidth).

        // Initialize ExpertPool simulation (RUSTANE_POOL_SIM=1 to enable).
        if pool_sim_enabled() {
            POOL_SIM.with(|p| *p.borrow_mut() = Some(ExpertPool::new(2000)));
        }

        let t_decode = std::time::Instant::now();
        let mut pos = input_ids.len();
        let mut first_token_logged = false;
        for step in 0..max_new_tokens.saturating_sub(1) {
            if pos >= max_seq { break; }
            let token_id = *all_ids.last().unwrap();
            if token_id == model.config.model.eos_token_id { break; }

            let t_tok = std::time::Instant::now();
            let profile_tok = step < 2; // profile first 2 decode tokens
            let mut tok_mla_us = 0u64;
            let mut tok_ffn_us = 0u64;
            let mut tok_join_us = 0u64;

            let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
            let mut x = emb;

            // Extract field refs so spawn closure doesn't capture &ModelV2
            let weights_ref = &model.weights;
            let config_ref = &model.config;
            // No serial convert(0) — dense layers use cached_dense
            for layer in 0..num_layers {
                // Use cached weights for dense layers, dynamic buffer for MoE
                let lf: &MlaLayerF32 = if layer < num_dense {
                    &cached_dense[layer]
                } else {
                    &buf_a
                };

                // Phase 1: MLA attention — no background thread, full memory bandwidth
                let t_mla = std::time::Instant::now();
                let (mut residual, normed2) = run_mla_only(
                    model, &mut cache, layer, &x, pos, lf,
                )?;
                if profile_tok { tok_mla_us += t_mla.elapsed().as_micros() as u64; }

                // Phase 2: FFN, with optional convert(N+1) overlap
                if layer + 1 < num_layers {
                    let next = layer + 1;
                    if next < num_dense {
                        let t_ffn = std::time::Instant::now();
                        run_ffn_only(model, &mut router, layer, &normed2, &mut residual, lf)?;
                        if profile_tok { tok_ffn_us += t_ffn.elapsed().as_micros() as u64; }
                    } else {
                        // Next layer needs conversion — pipeline overlap
                        std::thread::scope(|s| -> Result<()> {
                            let h = s.spawn(|| {
                                convert_layer_into(&mut buf_b, weights_ref, config_ref, next)
                            });
                            let t_ffn = std::time::Instant::now();
                            run_ffn_only(model, &mut router, layer, &normed2, &mut residual, lf)?;
                            if profile_tok { tok_ffn_us += t_ffn.elapsed().as_micros() as u64; }
                            let t_join = std::time::Instant::now();
                            h.join().unwrap()?;
                            if profile_tok { tok_join_us += t_join.elapsed().as_micros() as u64; }
                            Ok(())
                        })?;
                        std::mem::swap(&mut buf_a, &mut buf_b);
                    }
                } else {
                    let t_ffn = std::time::Instant::now();
                    run_ffn_only(model, &mut router, layer, &normed2, &mut residual, lf)?;
                    if profile_tok { tok_ffn_us += t_ffn.elapsed().as_micros() as u64; }
                }

                x = residual;
            }
            cache.advance();

            let t_lm = std::time::Instant::now();
            let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
            let logits = matvec_f32(&model.lm_head_f32, &normed, vocab, hidden);
            let lm_us = t_lm.elapsed().as_micros() as u64;
            let next_token = sample(&logits, sampling, (pos + step) as u64);
            all_ids.push(next_token);

            let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
            if profile_tok {
                let accounted = (tok_mla_us + tok_ffn_us + tok_join_us + lm_us) as f64 / 1000.0;
                eprintln!("--- decode token {step}: {tok_ms:.0}ms ({:.1}ms/layer) ---",
                    tok_ms / num_layers as f64);
                eprintln!("  MLA: {:.1}ms  FFN: {:.1}ms  convert_wait: {:.1}ms  lm_head: {:.1}ms  other: {:.1}ms",
                    tok_mla_us as f64 / 1000.0, tok_ffn_us as f64 / 1000.0,
                    tok_join_us as f64 / 1000.0, lm_us as f64 / 1000.0,
                    tok_ms - accounted);
            }
            if !first_token_logged {
                first_token_logged = true;
            }

            pos += 1;
        }
        let decode_secs = t_decode.elapsed().as_secs_f64();

        // Log ExpertPool simulation results (if enabled)
        if pool_sim_enabled() {
            POOL_SIM.with(|p| {
                if let Some(pool) = p.borrow().as_ref() {
                    eprintln!("ExpertPool sim (cap=2000): hits={} misses={} rate={:.1}% evictions={}",
                        pool.stats.hits, pool.stats.misses,
                        pool.stats.hit_rate() * 100.0, pool.stats.evictions);
                }
                *p.borrow_mut() = None;
            });
        }

        (prefill_secs, decode_secs)
    } else {
        // --- Pre-converted path (small models like V2-Lite) ---
        let t_prefill = std::time::Instant::now();
        for (i, &token_id) in input_ids.iter().enumerate() {
            let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
            let mut x = emb;
            for layer in 0..num_layers {
                x = run_layer_v2(model, &mut cache, &mut router, layer, &x, i, None)?;
            }
            cache.advance();
            if i == input_ids.len() - 1 {
                let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
                let logits = matvec_f32(&model.lm_head_f32, &normed, vocab, hidden);
                let next_token = sample(&logits, sampling, i as u64);
                all_ids.push(next_token);
            }
        }
        let prefill_secs = t_prefill.elapsed().as_secs_f64();

        let t_decode = std::time::Instant::now();
        let mut pos = input_ids.len();
        let mut first_token_logged = false;
        for step in 0..max_new_tokens.saturating_sub(1) {
            if pos >= max_seq { break; }
            let token_id = *all_ids.last().unwrap();
            if token_id == model.config.model.eos_token_id { break; }

            let t_tok = std::time::Instant::now();
            let profile_this = !first_token_logged;
            let mut layer_timings: Vec<LayerTiming> = if profile_this {
                (0..num_layers).map(|_| LayerTiming::default()).collect()
            } else {
                Vec::new()
            };

            let emb = embed_f16_to_f32(embed_table, token_id as usize, hidden);
            let mut x = emb;
            for layer in 0..num_layers {
                let timing = if profile_this { Some(&mut layer_timings[layer]) } else { None };
                x = run_layer_v2(model, &mut cache, &mut router, layer, &x, pos, timing)?;
            }
            cache.advance();

            let normed = rmsnorm(&x, &final_norm.to_vec(), model.config.rms_norm_eps());
            let logits = matvec_f32(&model.lm_head_f32, &normed, vocab, hidden);
            let next_token = sample(&logits, sampling, (pos + step) as u64);
            all_ids.push(next_token);

            if profile_this {
                first_token_logged = true;
                let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
                let total_conv: f64 = layer_timings.iter().map(|t| t.convert_ms).sum();
                let total_compute: f64 = layer_timings.iter().map(|t| t.attn_ms).sum();
                eprintln!("--- decode token 0 profile ({tok_ms:.0}ms total) ---");
                eprintln!("  convert: {total_conv:.0}ms  compute: {total_compute:.0}ms");
            }

            pos += 1;
        }
        let decode_secs = t_decode.elapsed().as_secs_f64();
        (prefill_secs, decode_secs)
    };

    let generated_ids = all_ids[input_ids.len()..].to_vec();
    let text = tokenizer.decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

    Ok(GenerateV2Output {
        token_ids: generated_ids.clone(),
        text,
        tokens_generated: generated_ids.len(),
        prefill_secs,
        decode_secs,
        prompt_tokens,
    })
}
