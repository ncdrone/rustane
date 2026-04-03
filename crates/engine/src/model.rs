//! Model configuration for transformer architectures.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SwiGlu,
    LeakyReluSq,
}

impl FfnActivation {
    pub fn checkpoint_code(self) -> u32 {
        match self {
            Self::SwiGlu => 0,
            Self::LeakyReluSq => 1,
        }
    }

    pub fn from_checkpoint_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::SwiGlu),
            1 => Some(Self::LeakyReluSq),
            _ => None,
        }
    }
}

/// Transformer model config. All 10 ANE kernels are parameterized by this.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize, // dim / heads
    pub seq: usize,
    pub nlayers: usize,
    pub vocab: usize,
    pub q_dim: usize,     // heads * hd
    pub kv_dim: usize,    // kv_heads * hd
    pub gqa_ratio: usize, // heads / kv_heads
    pub ffn_activation: FfnActivation,
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
            ffn_activation: FfnActivation::SwiGlu,
        }
    }

    /// GPT-Karpathy with grouped-query attention: 6 query heads, 3 KV heads.
    /// Keeps head_dim at 128 and halves KV projection width.
    pub fn gpt_karpathy_gqa() -> Self {
        Self {
            dim: 768,
            hidden: 2048,
            heads: 6,
            kv_heads: 3,
            hd: 128,
            seq: 512,
            nlayers: 6,
            vocab: 8192,
            q_dim: 768,
            kv_dim: 384,
            gqa_ratio: 2,
            ffn_activation: FfnActivation::SwiGlu,
        }
    }

    /// GPT-Karpathy with the FFN gate activation swapped from SiLU to leaky_relu(x, 0.5)^2.
    pub fn gpt_karpathy_leaky_relu_sq() -> Self {
        Self {
            ffn_activation: FfnActivation::LeakyReluSq,
            ..Self::gpt_karpathy()
        }
    }

    /// Parameter-golf scaffold at Rustane scale:
    /// - 3x MLP expansion
    /// - 6 query heads / 2 KV heads (GQA)
    /// - current Rustane leaky_relu^2 FFN activation variant
    ///
    /// SmearGate, BigramHash, and the full ungated PG MLP structure land separately.
    pub fn gpt_karpathy_pg() -> Self {
        Self {
            hidden: 3072,
            kv_heads: 2,
            kv_dim: 256,
            gqa_ratio: 3,
            ffn_activation: FfnActivation::LeakyReluSq,
            ..Self::gpt_karpathy()
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
            ffn_activation: FfnActivation::SwiGlu,
        }
    }

    /// Target 600M: NL=20, DIM=1536, HEADS=12, MHA — ~579M params
    /// The actual model we're training. 710ms/step baseline.
    pub fn target_600m() -> Self {
        Self {
            dim: 1536,
            hidden: 4096,
            heads: 12,
            kv_heads: 12,
            hd: 128,
            seq: 512,
            nlayers: 20,
            vocab: 8192,
            q_dim: 1536,
            kv_dim: 1536,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        }
    }

    /// Target 800M: NL=24, DIM=1792, HEADS=14, MHA — ~830M params
    /// Stress test config for benchmarking at scale beyond 600M.
    pub fn target_800m() -> Self {
        Self {
            dim: 1792,
            hidden: 4864,
            heads: 14,
            kv_heads: 14,
            hd: 128,
            seq: 512,
            nlayers: 24,
            vocab: 8192,
            q_dim: 1792,
            kv_dim: 1792,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        }
    }

    /// Target 1B: NL=28, DIM=2048, HEADS=16, MHA — ~1.2B params
    /// Stress test for 1B+ scale. Tests IOSurface alignment at 2048-dim.
    pub fn target_1b() -> Self {
        Self {
            dim: 2048,
            hidden: 5632,
            heads: 16,
            kv_heads: 16,
            hd: 128,
            seq: 512,
            nlayers: 28,
            vocab: 8192,
            q_dim: 2048,
            kv_dim: 2048,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        }
    }

    /// Estimated parameter count (transformer layers + embedding).
    pub fn param_count(&self) -> usize {
        let per_layer = 4 * self.dim * self.dim + 3 * self.dim * self.hidden;
        let embed = self.vocab * self.dim;
        let gamma = self.dim * (2 * self.nlayers + 1);
        self.nlayers * per_layer + embed + gamma
    }

    /// Target 1.5B: NL=32, DIM=2304, HEADS=18, MHA — ~1.5B params
    /// Stress test: ffnFused IOSurface ~179MB, total alloc ~1.8GB.
    pub fn target_1_5b() -> Self {
        Self {
            dim: 2304,
            hidden: 6144,
            heads: 18,
            kv_heads: 18,
            hd: 128,
            seq: 512,
            nlayers: 32,
            vocab: 8192,
            q_dim: 2304,
            kv_dim: 2304,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
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
            ffn_activation: FfnActivation::SwiGlu,
        }
    }
}
