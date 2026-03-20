//! GQA KV cache for Qwen3 inference.
//!
//! Flat storage: one contiguous buffer per (layer, kv_head) for K and V.
//! Supports sequential appending (decode) and retrieval of all cached K/V.

/// GQA KV cache.
pub struct KvCache {
    /// K storage: [num_layers][num_kv_heads][max_seq * head_dim]
    k_store: Vec<Vec<f32>>,
    /// V storage: [num_layers][num_kv_heads][max_seq * head_dim]
    v_store: Vec<Vec<f32>>,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    /// Current sequence length (number of stored positions).
    pub seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let entries = num_layers * num_kv_heads;
        let k_store = (0..entries)
            .map(|_| vec![0.0f32; max_seq * head_dim])
            .collect();
        let v_store = (0..entries)
            .map(|_| vec![0.0f32; max_seq * head_dim])
            .collect();
        Self {
            k_store,
            v_store,
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
        }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let entries = self.num_layers * self.num_kv_heads;
        entries * 2 * self.max_seq * self.head_dim * 4 // K + V, f32
    }

    fn idx(&self, layer: usize, kv_head: usize) -> usize {
        layer * self.num_kv_heads + kv_head
    }

    /// Store K and V for all KV heads at the current position for a given layer.
    /// `k` shape: [num_kv_heads * head_dim], `v` shape: [num_kv_heads * head_dim]
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(v.len(), self.num_kv_heads * self.head_dim);
        assert!(pos < self.max_seq, "pos {pos} >= max_seq {}", self.max_seq);

        for h in 0..self.num_kv_heads {
            let idx = self.idx(layer, h);
            let src_off = h * self.head_dim;
            let dst_off = pos * self.head_dim;
            self.k_store[idx][dst_off..dst_off + self.head_dim]
                .copy_from_slice(&k[src_off..src_off + self.head_dim]);
            self.v_store[idx][dst_off..dst_off + self.head_dim]
                .copy_from_slice(&v[src_off..src_off + self.head_dim]);
        }
    }

    /// Get cached K for a specific layer and KV head, up to `seq_len` positions.
    /// Returns slice of length `seq_len * head_dim`.
    pub fn get_k(&self, layer: usize, kv_head: usize, seq_len: usize) -> &[f32] {
        let idx = self.idx(layer, kv_head);
        &self.k_store[idx][..seq_len * self.head_dim]
    }

    /// Get cached V for a specific layer and KV head, up to `seq_len` positions.
    pub fn get_v(&self, layer: usize, kv_head: usize, seq_len: usize) -> &[f32] {
        let idx = self.idx(layer, kv_head);
        &self.v_store[idx][..seq_len * self.head_dim]
    }

    /// Advance sequence position by 1.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Reset cache (for new generation).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        // No need to zero — we only read up to seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_retrieve_single() {
        let mut cache = KvCache::new(2, 4, 128, 4096);
        let k = vec![1.0f32; 4 * 128];
        let v = vec![2.0f32; 4 * 128];
        cache.store(0, 0, &k, &v);

        let k_out = cache.get_k(0, 0, 1);
        assert_eq!(k_out.len(), 128);
        assert_eq!(k_out[0], 1.0);

        let v_out = cache.get_v(0, 0, 1);
        assert_eq!(v_out.len(), 128);
        assert_eq!(v_out[0], 2.0);
    }

    #[test]
    fn multi_position() {
        let mut cache = KvCache::new(1, 2, 4, 100);
        // Store at pos 0
        let k0: Vec<f32> = (0..8).map(|i| i as f32).collect(); // 2 heads * 4 dim
        let v0 = vec![10.0f32; 8];
        cache.store(0, 0, &k0, &v0);

        // Store at pos 1
        let k1: Vec<f32> = (8..16).map(|i| i as f32).collect();
        let v1 = vec![20.0f32; 8];
        cache.store(0, 1, &k1, &v1);

        // Retrieve head 0, 2 positions
        let k = cache.get_k(0, 0, 2);
        assert_eq!(k.len(), 8); // 2 * head_dim=4
        assert_eq!(k[0], 0.0); // pos 0, dim 0
        assert_eq!(k[4], 8.0); // pos 1, dim 0

        let v = cache.get_v(0, 0, 2);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[4], 20.0);
    }

    #[test]
    fn multi_layer() {
        let mut cache = KvCache::new(48, 4, 128, 4096);
        let k = vec![42.0f32; 4 * 128];
        let v = vec![99.0f32; 4 * 128];
        cache.store(47, 0, &k, &v);

        let k_out = cache.get_k(47, 0, 1);
        assert_eq!(k_out[0], 42.0);

        let v_out = cache.get_v(47, 3, 1);
        assert_eq!(v_out[0], 99.0);
    }

    #[test]
    fn memory_budget() {
        let cache = KvCache::new(48, 4, 128, 4096);
        let gb = cache.memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        // 48 layers * 4 heads * 2 (K+V) * 4096 * 128 * 4 bytes = ~24 GB
        println!("GQA KV cache: {gb:.2} GB (f32, 4K ctx, 48 layers)");
        assert!(gb < 30.0);
    }
}
