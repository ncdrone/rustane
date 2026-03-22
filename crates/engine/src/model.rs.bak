//! Model configuration for transformer architectures.

/// Transformer model config. All 10 ANE kernels are parameterized by this.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,        // dim / heads
    pub seq: usize,
    pub nlayers: usize,
    pub vocab: usize,
    pub q_dim: usize,     // heads * hd
    pub kv_dim: usize,    // kv_heads * hd
    pub gqa_ratio: usize, // heads / kv_heads
}

impl ModelConfig {
    /// GPT-Karpathy: NL=6, DIM=768, HEADS=6, MHA, climbmix-400B
    pub fn gpt_karpathy() -> Self {
        Self {
            dim: 768,
            hidden: 2048,
            heads: 6,
            kv_heads: 6,
            hd: 128,
            seq: 512,
            nlayers: 6,
            vocab: 8192,
            q_dim: 768,
            kv_dim: 768,
            gqa_ratio: 1,
        }
    }

    /// GPT-1024: NL=8, DIM=1024, HEADS=8, MHA — ~110M params
    pub fn gpt_1024() -> Self {
        Self {
            dim: 1024,
            hidden: 2816,
            heads: 8,
            kv_heads: 8,
            hd: 128,
            seq: 512,
            nlayers: 8,
            vocab: 8192,
            q_dim: 1024,
            kv_dim: 1024,
            gqa_ratio: 1,
        }
    }

    /// MHA-28L: Qwen3-0.6B skeleton with MHA — ~390M params
    /// Mirrors Qwen3-0.6B (28L/1024/3072/s256) but MHA instead of GQA.
    /// IOSurface alignment verified: SDPA=3328(÷16), FFN=9728(÷16), WO=1280(÷16).
    pub fn mha_28l() -> Self {
        Self {
            dim: 1024,
            hidden: 3072,
            heads: 8,
            kv_heads: 8,
            hd: 128,
            seq: 256,
            nlayers: 28,
            vocab: 8192,
            q_dim: 1024,
            kv_dim: 1024,
            gqa_ratio: 1,
        }
    }
}
