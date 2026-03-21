//! Multi-Latent Attention (MLA) forward pass for DeepSeek-V2/V3.
//!
//! Corrected architecture from research:
//! - Score = TWO separate dot products (nope + rope) SUMMED, not concatenated
//! - Scale = 1/sqrt(192) * mscale^2, NOT 1/sqrt(576)
//! - kv_a_layernorm BEFORE cache write
//! - k_pe stored POST-RoPE in cache; q_pe gets RoPE at decode time
//! - W_UK absorption via batched per-head GEMV

use crate::config::InferConfig;
use crate::rmsnorm::rmsnorm;
use crate::yarn_rope::YarnRopeTables;

/// MLA configuration extracted from InferConfig.
#[derive(Clone, Debug)]
pub struct MlaDecodeConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub rms_eps: f32,
}

impl MlaDecodeConfig {
    pub fn from_infer_config(cfg: &InferConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size(),
            num_heads: cfg.num_q_heads(),
            qk_nope_head_dim: cfg.attention.qk_nope_head_dim.unwrap_or(128),
            qk_rope_head_dim: cfg.attention.qk_rope_head_dim.unwrap_or(64),
            v_head_dim: cfg.attention.v_head_dim.unwrap_or(128),
            kv_lora_rank: cfg.attention.kv_lora_rank.unwrap_or(512),
            rms_eps: cfg.rms_norm_eps(),
        }
    }

    /// Total Q output dim: num_heads * (nope + rope)
    pub fn q_total_dim(&self) -> usize {
        self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
    }
}

/// Pre-split MLA weights for one layer (all f32, pre-converted at load time).
pub struct MlaLayerWeights {
    /// Q projection: [num_heads * (nope + rope), hidden] row-major
    pub q_proj: Vec<f32>,
    /// KV down-projection: [kv_lora_rank + rope_dim, hidden] row-major
    pub kv_a_proj: Vec<f32>,
    /// KV layernorm weights: [kv_lora_rank]
    pub kv_a_layernorm: Vec<f32>,
    /// W_UK: [num_heads, nope_dim, kv_lora_rank] — split from kv_b_proj at load time
    pub w_uk: Vec<f32>,
    /// W_UV: [num_heads, v_head_dim, kv_lora_rank] — split from kv_b_proj at load time
    pub w_uv: Vec<f32>,
    /// O projection: [hidden, num_heads * v_head_dim] row-major
    pub o_proj: Vec<f32>,
    /// Input layernorm: [hidden]
    pub input_norm: Vec<f32>,
    /// Post-attention layernorm: [hidden]
    pub post_attn_norm: Vec<f32>,
}

/// MLA KV cache: stores compressed latent + rope key per position per layer.
pub struct MlaKvCache {
    /// kv_latent: [num_layers][max_seq * kv_lora_rank] in f32
    latents: Vec<Vec<f32>>,
    /// k_pe (post-RoPE): [num_layers][max_seq * rope_dim] in f32
    rope_keys: Vec<Vec<f32>>,
    pub kv_lora_rank: usize,
    pub rope_dim: usize,
    pub max_seq: usize,
    pub num_layers: usize,
    /// Number of tokens cached so far.
    pub seq_len: usize,
}

impl MlaKvCache {
    pub fn new(num_layers: usize, kv_lora_rank: usize, rope_dim: usize, max_seq: usize) -> Self {
        Self {
            latents: (0..num_layers).map(|_| vec![0.0f32; max_seq * kv_lora_rank]).collect(),
            rope_keys: (0..num_layers).map(|_| vec![0.0f32; max_seq * rope_dim]).collect(),
            kv_lora_rank,
            rope_dim,
            max_seq,
            num_layers,
            seq_len: 0,
        }
    }

    /// Write kv_latent (post-norm) and k_pe (post-RoPE) for one token at one layer.
    pub fn write(&mut self, layer: usize, pos: usize, latent: &[f32], k_pe: &[f32]) {
        debug_assert_eq!(latent.len(), self.kv_lora_rank);
        debug_assert_eq!(k_pe.len(), self.rope_dim);
        debug_assert!(pos < self.max_seq);

        let lat_off = pos * self.kv_lora_rank;
        self.latents[layer][lat_off..lat_off + self.kv_lora_rank].copy_from_slice(latent);

        let rope_off = pos * self.rope_dim;
        self.rope_keys[layer][rope_off..rope_off + self.rope_dim].copy_from_slice(k_pe);
    }

    /// Get cached kv_latent for a layer up to seq_len positions: [seq_len * kv_lora_rank]
    pub fn get_latents(&self, layer: usize, seq_len: usize) -> &[f32] {
        &self.latents[layer][..seq_len * self.kv_lora_rank]
    }

    /// Get cached k_pe for a layer up to seq_len positions: [seq_len * rope_dim]
    pub fn get_rope_keys(&self, layer: usize, seq_len: usize) -> &[f32] {
        &self.rope_keys[layer][..seq_len * self.rope_dim]
    }

    /// Advance position counter (call after writing to all layers for one token).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Effective sequence length.
    pub fn effective_len(&self) -> usize {
        self.seq_len.min(self.max_seq)
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let lat: usize = self.latents.iter().map(|l| l.len() * 4).sum();
        let rope: usize = self.rope_keys.iter().map(|r| r.len() * 4).sum();
        lat + rope
    }
}

/// MLA forward pass for single-token decode (CPU path).
///
/// Steps:
/// 1. Q = x @ q_proj^T → split into q_nope [H, nope] + q_pe [H, rope]
/// 2. q_pe = apply_rope(q_pe, pos)
/// 3. kv_out = x @ kv_a_proj^T → split [kv_latent, k_pe_raw]
/// 4. kv_latent = RMSNorm(kv_latent)
/// 5. k_pe = apply_rope(k_pe_raw, pos)
/// 6. cache.write(layer, pos, kv_latent, k_pe)
/// 7. Absorbed attention: q_absorbed = per-head(q_nope @ W_UK) → scores_nope
///    scores_rope = q_pe @ k_pe_cache^T (broadcast across heads)
///    scores = (scores_nope + scores_rope) * scale → softmax → weights
/// 8. v_latent = weights @ kv_latent_cache → v = per-head(v_latent @ W_UV)
/// 9. output = concat(v) @ o_proj^T
pub fn mla_forward_decode(
    x: &[f32],
    weights: &MlaLayerWeights,
    cache: &mut MlaKvCache,
    layer: usize,
    pos: usize,
    rope: &YarnRopeTables,
    cfg: &MlaDecodeConfig,
    attn_scale: f32,
) -> Vec<f32> {
    let h = cfg.num_heads;
    let nope = cfg.qk_nope_head_dim;
    let rope_dim = cfg.qk_rope_head_dim;
    let v_dim = cfg.v_head_dim;
    let kv_rank = cfg.kv_lora_rank;
    let hidden = cfg.hidden_size;

    // 1. Q projection
    let q_total = h * (nope + rope_dim);
    let mut q = vec![0.0f32; q_total];
    crate::blas::sgemv_f32(&weights.q_proj, x, &mut q, q_total, hidden);

    // Split into q_nope [H * nope] and q_pe [H * rope_dim]
    let mut q_nope = vec![0.0f32; h * nope];
    let mut q_pe = vec![0.0f32; h * rope_dim];
    for head in 0..h {
        let src = head * (nope + rope_dim);
        q_nope[head * nope..(head + 1) * nope]
            .copy_from_slice(&q[src..src + nope]);
        q_pe[head * rope_dim..(head + 1) * rope_dim]
            .copy_from_slice(&q[src + nope..src + nope + rope_dim]);
    }

    // 2. Apply RoPE to q_pe (all heads get same rotation)
    rope.apply_all_heads(&mut q_pe, h, pos);

    // 3. KV compression: kv_out = x @ kv_a_proj^T
    let kv_out_dim = kv_rank + rope_dim;
    let mut kv_out = vec![0.0f32; kv_out_dim];
    crate::blas::sgemv_f32(&weights.kv_a_proj, x, &mut kv_out, kv_out_dim, hidden);

    let kv_latent_raw = &kv_out[..kv_rank];
    let k_pe_raw = &kv_out[kv_rank..];

    // 4. kv_a_layernorm BEFORE cache (CRITICAL)
    let kv_latent = rmsnorm(kv_latent_raw, &weights.kv_a_layernorm, cfg.rms_eps);

    // 5. Apply RoPE to k_pe (stored post-RoPE in cache)
    let mut k_pe = k_pe_raw.to_vec();
    rope.apply(&mut k_pe, pos);

    // 6. Cache write
    cache.write(layer, pos, &kv_latent, &k_pe);
    let seq_len = pos + 1;

    // 7. Absorbed attention — TWO SEPARATE DOT PRODUCTS SUMMED

    // 7a. q_absorbed = per-head q_nope @ W_UK: [H, nope] × [H, nope, kv_rank] → [H, kv_rank]
    let mut q_absorbed = vec![0.0f32; h * kv_rank];
    for head in 0..h {
        let q_head = &q_nope[head * nope..(head + 1) * nope];
        let w_uk_head = &weights.w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
        // q_absorbed[head] = q_head @ W_UK[head]
        // W_UK[head] is [nope, kv_rank] row-major
        let out = &mut q_absorbed[head * kv_rank..(head + 1) * kv_rank];
        crate::blas::sgemv_f32(w_uk_head, q_head, out, kv_rank, nope);
    }

    // 7b. Compute attention scores
    let latent_cache = cache.get_latents(layer, seq_len);
    let rope_cache = cache.get_rope_keys(layer, seq_len);

    let mut scores = vec![0.0f32; h * seq_len];
    for head in 0..h {
        let q_abs = &q_absorbed[head * kv_rank..(head + 1) * kv_rank];
        let q_rope = &q_pe[head * rope_dim..(head + 1) * rope_dim];

        for t in 0..seq_len {
            let lat_t = &latent_cache[t * kv_rank..(t + 1) * kv_rank];
            let rope_t = &rope_cache[t * rope_dim..(t + 1) * rope_dim];

            // scores_nope = q_absorbed · kv_latent[t]
            let mut dot_nope = 0.0f64;
            for d in 0..kv_rank {
                dot_nope += q_abs[d] as f64 * lat_t[d] as f64;
            }

            // scores_rope = q_pe · k_pe[t] (k_pe is MQA — shared across all heads)
            let mut dot_rope = 0.0f64;
            for d in 0..rope_dim {
                dot_rope += q_rope[d] as f64 * rope_t[d] as f64;
            }

            scores[head * seq_len + t] = (dot_nope + dot_rope) as f32 * attn_scale;
        }
    }

    // 7c. Softmax per head
    let mut attn_weights = vec![0.0f32; h * seq_len];
    for head in 0..h {
        let s = &scores[head * seq_len..(head + 1) * seq_len];
        let w = &mut attn_weights[head * seq_len..(head + 1) * seq_len];
        let max_s = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for t in 0..seq_len {
            w[t] = (s[t] - max_s).exp();
            sum += w[t];
        }
        for t in 0..seq_len {
            w[t] /= sum;
        }
    }

    // 8. Value combination: v_latent = weights @ kv_latent_cache, then v = v_latent @ W_UV
    let mut v_concat = vec![0.0f32; h * v_dim];
    for head in 0..h {
        let w = &attn_weights[head * seq_len..(head + 1) * seq_len];

        // v_latent = sum_t(w[t] * kv_latent[t]) — [kv_rank]
        let mut v_latent = vec![0.0f32; kv_rank];
        for t in 0..seq_len {
            let lat_t = &latent_cache[t * kv_rank..(t + 1) * kv_rank];
            let wt = w[t];
            for d in 0..kv_rank {
                v_latent[d] += wt * lat_t[d];
            }
        }

        // v = v_latent @ W_UV[head]: [kv_rank] × [v_dim, kv_rank]^T → [v_dim]
        let w_uv_head = &weights.w_uv[head * v_dim * kv_rank..(head + 1) * v_dim * kv_rank];
        let v_out = &mut v_concat[head * v_dim..(head + 1) * v_dim];
        crate::blas::sgemv_f32(w_uv_head, &v_latent, v_out, v_dim, kv_rank);
    }

    // 9. Output projection: output = v_concat @ o_proj^T
    let out_dim = hidden;
    let v_total = h * v_dim;
    let mut output = vec![0.0f32; out_dim];
    crate::blas::sgemv_f32(&weights.o_proj, &v_concat, &mut output, out_dim, v_total);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_kv_cache_basic() {
        let mut cache = MlaKvCache::new(2, 512, 64, 100);
        let latent = vec![1.0f32; 512];
        let k_pe = vec![2.0f32; 64];
        cache.write(0, 0, &latent, &k_pe);
        cache.advance();

        assert_eq!(cache.effective_len(), 1);
        let lat = cache.get_latents(0, 1);
        assert_eq!(lat.len(), 512);
        assert_eq!(lat[0], 1.0);
        let rk = cache.get_rope_keys(0, 1);
        assert_eq!(rk.len(), 64);
        assert_eq!(rk[0], 2.0);
    }

    #[test]
    fn mla_kv_cache_multi_pos() {
        let mut cache = MlaKvCache::new(1, 4, 2, 100);
        cache.write(0, 0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0]);
        cache.write(0, 1, &[5.0, 6.0, 7.0, 8.0], &[30.0, 40.0]);
        cache.seq_len = 2;

        let lat = cache.get_latents(0, 2);
        assert_eq!(lat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rk = cache.get_rope_keys(0, 2);
        assert_eq!(rk, &[10.0, 20.0, 30.0, 40.0]);
    }
}
