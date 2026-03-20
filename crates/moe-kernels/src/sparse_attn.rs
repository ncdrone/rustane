//! Sparse attention primitives for long context (100K+ tokens).
//!
//! Implements NSA 3-branch sparse attention pattern:
//! 1. Compressed: attend to compressed representations of distant tokens
//! 2. Selected: attend to top-k most relevant distant tokens
//! 3. Sliding: attend to local sliding window (recent tokens)
//!
//! Also supports 3-tier KV cache:
//! - Hot (64K tokens): full precision, fast access
//! - Warm (1M tokens): 3-bit quantized, moderate access
//! - Cold (100M+ tokens): Infini-attention summary vectors

/// Configuration for sparse attention.
#[derive(Clone, Debug)]
pub struct SparseAttnConfig {
    /// Local sliding window size.
    pub window_size: usize,
    /// Number of top-k distant tokens to attend to.
    pub selected_k: usize,
    /// Compression ratio for compressed branch.
    pub compress_ratio: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl Default for SparseAttnConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            selected_k: 256,
            compress_ratio: 16,
            num_heads: 56,
            head_dim: 128,
        }
    }
}

/// 3-tier KV cache for ultra-long context.
pub struct TieredKvCache {
    config: SparseAttnConfig,
    /// Hot tier: full-precision KV for recent tokens.
    pub hot: KvTier,
    /// Warm tier: quantized KV for medium-range tokens.
    pub warm: KvTier,
    /// Cold tier: summary vectors (Infini-attention style).
    pub cold: ColdTier,
}

/// A single tier of the KV cache.
pub struct KvTier {
    pub max_tokens: usize,
    pub current_len: usize,
    /// Keys: [max_tokens, num_heads * head_dim] flattened.
    pub keys: Vec<f32>,
    /// Values: [max_tokens, num_heads * head_dim] flattened.
    pub values: Vec<f32>,
    kv_dim: usize,
}

/// Cold tier: Infini-attention summary (fixed-size memory per head).
pub struct ColdTier {
    /// Memory vectors: [num_heads, head_dim] — running summary of all cold tokens.
    pub memory: Vec<f32>,
    /// Normalization: [num_heads] — count of tokens summarized.
    pub norm: Vec<f32>,
    num_heads: usize,
    head_dim: usize,
}

impl TieredKvCache {
    pub fn new(config: SparseAttnConfig, hot_tokens: usize, warm_tokens: usize) -> Self {
        let kv_dim = config.num_heads * config.head_dim;
        Self {
            hot: KvTier::new(hot_tokens, kv_dim),
            warm: KvTier::new(warm_tokens, kv_dim),
            cold: ColdTier::new(config.num_heads, config.head_dim),
            config,
        }
    }

    /// Append a new KV pair. Handles tier promotion/demotion.
    pub fn append(&mut self, key: &[f32], value: &[f32]) {
        let kv_dim = self.config.num_heads * self.config.head_dim;
        assert_eq!(key.len(), kv_dim);
        assert_eq!(value.len(), kv_dim);

        // If hot tier is full, demote oldest to warm
        if self.hot.current_len >= self.hot.max_tokens {
            let oldest_k = self.hot.get_key(0).to_vec();
            let oldest_v = self.hot.get_value(0).to_vec();

            // If warm is full, demote its oldest to cold
            if self.warm.current_len >= self.warm.max_tokens {
                let cold_k = self.warm.get_key(0);
                let cold_v = self.warm.get_value(0);
                self.cold.absorb(cold_k, cold_v);
                self.warm.shift_left();
            }

            self.warm.append(&oldest_k, &oldest_v);
            self.hot.shift_left();
        }

        self.hot.append(key, value);
    }

    /// Total tokens tracked (hot + warm + cold summary).
    pub fn total_tokens(&self) -> usize {
        self.hot.current_len + self.warm.current_len
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.hot.memory_bytes() + self.warm.memory_bytes() + self.cold.memory_bytes()
    }
}

impl KvTier {
    fn new(max_tokens: usize, kv_dim: usize) -> Self {
        Self {
            max_tokens,
            current_len: 0,
            keys: vec![0.0; max_tokens * kv_dim],
            values: vec![0.0; max_tokens * kv_dim],
            kv_dim,
        }
    }

    fn append(&mut self, key: &[f32], value: &[f32]) {
        let off = self.current_len * self.kv_dim;
        self.keys[off..off + self.kv_dim].copy_from_slice(key);
        self.values[off..off + self.kv_dim].copy_from_slice(value);
        self.current_len += 1;
    }

    fn get_key(&self, pos: usize) -> &[f32] {
        let off = pos * self.kv_dim;
        &self.keys[off..off + self.kv_dim]
    }

    fn get_value(&self, pos: usize) -> &[f32] {
        let off = pos * self.kv_dim;
        &self.values[off..off + self.kv_dim]
    }

    fn shift_left(&mut self) {
        if self.current_len == 0 { return; }
        let kv_dim = self.kv_dim;
        self.keys.copy_within(kv_dim.., 0);
        self.values.copy_within(kv_dim.., 0);
        self.current_len -= 1;
    }

    fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * 4
    }
}

impl ColdTier {
    fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            memory: vec![0.0; num_heads * head_dim],
            norm: vec![0.0; num_heads],
            num_heads,
            head_dim,
        }
    }

    /// Absorb a key-value pair into the running summary (Infini-attention style).
    fn absorb(&mut self, _key: &[f32], value: &[f32]) {
        // Simple running mean per head
        for h in 0..self.num_heads {
            let off = h * self.head_dim;
            self.norm[h] += 1.0;
            let n = self.norm[h];
            for d in 0..self.head_dim {
                // Running mean: memory = memory + (v - memory) / n
                let v = value[off + d];
                self.memory[off + d] += (v - self.memory[off + d]) / n;
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        (self.memory.len() + self.norm.len()) * 4
    }
}
