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
//! For DeepSeek-V3/Kimi-K2:
//!   kv_lora_rank = 512, hidden_size = 7168, num_heads = 56, kv_heads = 8

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
    /// DeepSeek-V3 / target-1T configuration.
    pub fn deepseek_v3() -> Self {
        Self {
            hidden_size: 7168,
            num_heads: 56,
            num_kv_heads: 8,
            head_dim: 128,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_layers: 61,
        }
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
