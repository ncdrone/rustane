//! Multi-Latent Attention (MLA) for 10x KV cache reduction.
//!
//! Instead of storing full KV heads, MLA stores compressed latent vectors
//! and reconstructs K/V on the fly via LoRA-style low-rank projections.
//!
//! Key properties:
//! - KV cache stores compressed latent (kv_lora_rank dim, not full KV)
//! - Absorbed attention avoids explicit KV decompression
//! - Rolling buffer with GQA support
//!
//! For DeepSeek-V3: kv_lora_rank=512, hidden=7168, num_heads=128, kv_heads=128
//! For Kimi-K2:    kv_lora_rank=512, hidden=7168, num_heads=64,  kv_heads=64

/// MLA configuration.
#[derive(Clone, Debug)]
pub struct MlaConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Compressed KV latent dimension.
    pub kv_lora_rank: usize,
    /// Nope head dim (non-positional part of Q/K).
    pub qk_nope_head_dim: usize,
    /// RoPE head dim (positional part of Q/K).
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_layers: usize,
}

impl MlaConfig {
    /// DeepSeek-V3 (671B) configuration.
    /// Source: configs/deepseek-v3.toml (verified against HF config.json)
    pub fn deepseek_v3() -> Self {
        Self {
            hidden_size: 7168,
            num_heads: 128,
            num_kv_heads: 128,
            head_dim: 128,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_layers: 61,
        }
    }

    /// Kimi-K2 (1T) configuration.
    /// Source: configs/kimi-k2.toml (verified against HF config.json)
    pub fn kimi_k2() -> Self {
        Self {
            hidden_size: 7168,
            num_heads: 64,
            num_kv_heads: 64,
            head_dim: 128,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_layers: 61,
        }
    }

    /// Q total dimension: num_heads × (nope + rope).
    pub fn q_total_dim(&self) -> usize {
        self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
    }

    /// V total dimension: num_heads × v_head_dim (= o_proj input dim).
    pub fn v_total_dim(&self) -> usize {
        self.num_heads * self.v_head_dim
    }

    /// Bytes per token in the compressed KV cache.
    pub fn kv_bytes_per_token(&self) -> usize {
        // Store compressed latent (kv_lora_rank) + rope key (qk_rope_head_dim * kv_heads)
        // in fp16 (2 bytes per element)
        (self.kv_lora_rank + self.qk_rope_head_dim * self.num_kv_heads) * 2
    }

    /// Total KV cache size for a given context length across all layers.
    pub fn total_kv_cache_bytes(&self, ctx_len: usize) -> usize {
        ctx_len * self.num_layers * self.kv_bytes_per_token()
    }
}

/// Rolling KV cache that stores compressed MLA latents.
pub struct MlaKvCache {
    config: MlaConfig,
    max_seq: usize,
    /// Compressed latents: [num_layers][max_seq][kv_lora_rank] in f32.
    latents: Vec<Vec<f32>>,
    /// RoPE keys: [num_layers][max_seq][rope_dim * kv_heads] in f32.
    rope_keys: Vec<Vec<f32>>,
    /// Current sequence length (how many tokens have been cached).
    pub seq_len: usize,
}

impl MlaKvCache {
    pub fn new(config: MlaConfig, max_seq: usize) -> Self {
        let nl = config.num_layers;
        let latent_size = max_seq * config.kv_lora_rank;
        let rope_size = max_seq * config.qk_rope_head_dim * config.num_kv_heads;

        Self {
            latents: (0..nl).map(|_| vec![0.0f32; latent_size]).collect(),
            rope_keys: (0..nl).map(|_| vec![0.0f32; rope_size]).collect(),
            config,
            max_seq,
            seq_len: 0,
        }
    }

    /// Append a compressed latent + rope key for one token at one layer.
    pub fn append(
        &mut self,
        layer: usize,
        latent: &[f32],    // [kv_lora_rank]
        rope_key: &[f32],  // [rope_dim * kv_heads]
    ) {
        assert_eq!(latent.len(), self.config.kv_lora_rank);
        let rope_dim = self.config.qk_rope_head_dim * self.config.num_kv_heads;
        assert_eq!(rope_key.len(), rope_dim);

        let pos = self.seq_len % self.max_seq; // rolling buffer

        let lat_off = pos * self.config.kv_lora_rank;
        self.latents[layer][lat_off..lat_off + self.config.kv_lora_rank]
            .copy_from_slice(latent);

        let rope_off = pos * rope_dim;
        self.rope_keys[layer][rope_off..rope_off + rope_dim]
            .copy_from_slice(rope_key);
    }

    /// Advance sequence position (call after appending to all layers).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Get latent at a specific position for a layer.
    pub fn get_latent(&self, layer: usize, pos: usize) -> &[f32] {
        let off = (pos % self.max_seq) * self.config.kv_lora_rank;
        &self.latents[layer][off..off + self.config.kv_lora_rank]
    }

    /// Get rope key at a specific position for a layer.
    pub fn get_rope_key(&self, layer: usize, pos: usize) -> &[f32] {
        let rope_dim = self.config.qk_rope_head_dim * self.config.num_kv_heads;
        let off = (pos % self.max_seq) * rope_dim;
        &self.rope_keys[layer][off..off + rope_dim]
    }

    /// Effective sequence length (capped at max_seq for rolling buffer).
    pub fn effective_len(&self) -> usize {
        self.seq_len.min(self.max_seq)
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let latent_bytes: usize = self.latents.iter().map(|l| l.len() * 4).sum();
        let rope_bytes: usize = self.rope_keys.iter().map(|r| r.len() * 4).sum();
        latent_bytes + rope_bytes
    }
}

/// CPU reference: MLA Q projection (absorbed form).
/// Computes Q from hidden state using low-rank decomposition.
/// q_nope = x @ W_dq (down-project to latent, then up-project to nope heads)
/// q_rope = x @ W_qr (project to rope heads)
pub fn mla_q_projection(
    x: &[f32],          // [hidden_size]
    w_dq: &[f32],       // [hidden_size, num_heads * qk_nope_head_dim]
    w_qr: &[f32],       // [hidden_size, num_heads * qk_rope_head_dim]
    config: &MlaConfig,
) -> (Vec<f32>, Vec<f32>) {
    let nope_dim = config.num_heads * config.qk_nope_head_dim;
    let rope_dim = config.num_heads * config.qk_rope_head_dim;

    let q_nope = matvec(x, w_dq, config.hidden_size, nope_dim);
    let q_rope = matvec(x, w_qr, config.hidden_size, rope_dim);

    (q_nope, q_rope)
}

/// CPU reference: MLA KV compression.
/// Compress hidden state to low-rank latent + rope key.
pub fn mla_kv_compress(
    x: &[f32],          // [hidden_size]
    w_kv_down: &[f32],  // [hidden_size, kv_lora_rank]
    w_kr: &[f32],       // [hidden_size, rope_dim * kv_heads]
    config: &MlaConfig,
) -> (Vec<f32>, Vec<f32>) {
    let latent = matvec(x, w_kv_down, config.hidden_size, config.kv_lora_rank);
    let rope_dim = config.qk_rope_head_dim * config.num_kv_heads;
    let rope_key = matvec(x, w_kr, config.hidden_size, rope_dim);
    (latent, rope_key)
}

fn matvec(x: &[f32], w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    for i in 0..rows {
        let xi = x[i];
        for j in 0..cols {
            out[j] += xi * w[i * cols + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ANE Conv1x1 graph builders for MLA decode projections.
//
// These use Conv1x1 (3x faster than matmul on ANE per maderix benchmarks).
// Pattern from engine/src/kernels/dyn_matmul.rs:88 (build_conv).
//
// For decode (seq_len=1), we use baked-weight Conv1x1 graphs:
//   Input:  [1, IC, 1, 1]   — single token activation
//   Weight: [OC, IC, 1, 1]  — baked at compile time (same dims every layer)
//   Output: [1, OC, 1, 1]
//
// Shape-sharing: all 61 layers use the same dimensions, so we compile 1 graph
// per projection type and swap weights via IOSurface staging (maderix pattern).
// ---------------------------------------------------------------------------

use ane_bridge::ane::{Graph, Shape};

/// Pad spatial width to multiple of 16, minimum 64.
/// ANE requires spatial width ≥ 64 (silent failure below) and multiple of 16 (silent corruption).
pub fn pad16(x: usize) -> usize {
    let padded = (x + 15) & !15;
    if padded < 64 { 64 } else { padded }
}

/// Build a Conv1x1 projection graph with dynamic weights (staged in spatial dim).
///
/// Input IOSurface: [1, IC, 1, pad16(seq + OC)]
///   spatial[0..seq]     = activation
///   spatial[seq..seq+OC] = weight column-major (IC values per output channel)
///
/// Output: [1, OC, 1, seq]
///
/// This is the maderix dynamic-weight pattern: weights packed alongside activations
/// in the spatial dimension. Compile once, stage weights per-layer via memcpy.
pub fn build_mla_conv1x1(ic: usize, oc: usize, seq: usize) -> Graph {
    // ANE requires all spatial dims ≥ 64. For decode (seq=1), pad to 64.
    // Activation goes at spatial[0], weights at spatial[padded_seq..padded_seq+oc].
    let padded_seq = pad16(seq);
    let sp = pad16(padded_seq + oc);
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });

    // Slice activations: [1, IC, 1, padded_seq]
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, padded_seq]);

    // Slice weights: [1, IC, 1, OC]
    let wts = g.slice(input, [0, 0, 0, padded_seq], [1, ic, 1, oc]);

    // Transpose weights: [1, IC, 1, OC] → [1, OC, 1, IC]
    let wts_t = g.transpose(wts, [0, 3, 2, 1]);

    // Reshape to conv filter: [1, OC, 1, IC] → [OC, IC, 1, 1]
    let wts_conv = g.reshape(wts_t, Shape { batch: oc, channels: ic, height: 1, width: 1 });

    // Conv1x1: [1, IC, 1, padded_seq] * [OC, IC, 1, 1] → [1, OC, 1, padded_seq]
    // For decode: position 0 has the result, positions 1..63 are zero-padded.
    let _out = g.convolution_2d_1x1_dynamic(acts, wts_conv);

    g
}

/// Build a grouped Conv1x1 for batched per-head absorption (W_UK or W_UV).
///
/// Instead of 64 sequential sgemv([nope, kv_rank]) calls, dispatch one grouped conv:
///   Input:  [1, kv_rank, 1, pad16(1 + num_heads * nope)]
///     spatial[0]                          = compressed KV token (kv_rank channels)
///     spatial[1..1+num_heads*nope]        = per-head W_UK weights (kv_rank × nope each)
///
///   Output: [1, num_heads * nope, 1, 1]   = all heads absorbed
///
/// Uses grouped convolution: groups=num_heads, each group maps kv_rank → nope channels.
pub fn build_mla_grouped_absorption(
    kv_rank: usize,
    nope: usize,
    num_heads: usize,
) -> Graph {
    let total_oc = num_heads * nope;
    let sp = pad16(1 + total_oc); // 1 token + flattened per-head weights
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: kv_rank, height: 1, width: sp });

    // Slice activation: [1, kv_rank, 1, 1] (single compressed KV token)
    let acts = g.slice(input, [0, 0, 0, 0], [1, kv_rank, 1, 1]);

    // Slice weights: [1, kv_rank, 1, total_oc]
    let wts = g.slice(input, [0, 0, 0, 1], [1, kv_rank, 1, total_oc]);

    // Transpose: [1, kv_rank, 1, total_oc] → [1, total_oc, 1, kv_rank]
    let wts_t = g.transpose(wts, [0, 3, 2, 1]);

    // Reshape to grouped conv filter: [total_oc, kv_rank, 1, 1]
    // With groups=num_heads: each group has nope output channels, kv_rank input channels
    let wts_conv = g.reshape(wts_t, Shape {
        batch: total_oc,
        channels: kv_rank,
        height: 1,
        width: 1,
    });

    // Grouped Conv1x1: groups=num_heads
    // Input [1, kv_rank, 1, 1] is broadcast across groups (each group sees all kv_rank channels)
    // Weight [total_oc, kv_rank, 1, 1] with groups → each group: [nope, kv_rank, 1, 1]
    // Output: [1, total_oc, 1, 1]
    let _out = g.convolution_2d_1x1_dynamic(acts, wts_conv);

    g
}

/// Build a Conv1x1 projection with BAKED weights (compile-time constants).
///
/// Input IOSurface: [1, IC, 1, padded_seq] — activation only (no weight staging!)
/// Output: [1, OC, 1, padded_seq]
///
/// Weights are embedded in the compiled program via g.constant().
/// Zero per-token staging overhead. Compile once per layer, dispatch forever.
pub fn build_mla_conv1x1_baked(ic: usize, oc: usize, seq: usize, weights: &[f32]) -> Graph {
    assert_eq!(weights.len(), oc * ic, "weights must be [oc, ic] = [{oc}, {ic}]");
    let padded_seq = pad16(seq);
    let mut g = Graph::new();

    // Input: activation only [1, IC, 1, padded_seq]
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: padded_seq });

    // Baked weight constant: [OC, IC, 1, 1] (conv filter format)
    // Transpose weights from [OC, IC] row-major to conv filter layout
    let mut filter_data = vec![0.0f32; oc * ic];
    for j in 0..oc {
        for i in 0..ic {
            filter_data[j * ic + i] = weights[j * ic + i];
        }
    }
    let weight_const = g.constant(&filter_data, Shape { batch: oc, channels: ic, height: 1, width: 1 });

    // Conv1x1 with baked weights: [1, IC, 1, padded_seq] * [OC, IC, 1, 1] → [1, OC, 1, padded_seq]
    let _out = g.convolution_2d_1x1(input, weight_const, None);

    g
}

/// Spatial width needed for a Conv1x1 projection IOSurface.
/// Accounts for ANE minimum spatial width ≥ 64.
pub fn conv1x1_spatial_width(seq: usize, oc: usize) -> usize {
    let padded_seq = pad16(seq);
    pad16(padded_seq + oc)
}

/// Padded sequence length (for output buffer sizing).
pub fn padded_seq(seq: usize) -> usize {
    pad16(seq)
}

/// Spatial width needed for grouped absorption IOSurface.
pub fn absorption_spatial_width(num_heads: usize, nope: usize) -> usize {
    pad16(1 + num_heads * nope)
}
