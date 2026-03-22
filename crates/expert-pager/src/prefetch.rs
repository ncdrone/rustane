//! ExpertPrefetcher: DISABLED.
//!
//! Research (stage 2) showed cross-layer L-2 prefetching caused -18% performance
//! with only 25% hit rate. Trusting the OS page cache is better.
//! This module is kept as a no-op stub for API compatibility.

/// No-op prefetcher (disabled by research findings).
pub struct ExpertPrefetcher;

impl ExpertPrefetcher {
    pub fn new(_prefetch_k: usize) -> Self {
        Self
    }

    /// No-op: record routing scores.
    pub fn record(&mut self, _layer_idx: usize, _scores: Vec<f32>) {}

    /// Always returns empty — prefetching disabled.
    pub fn predict(&self, _target_layer: usize) -> Vec<u32> {
        Vec::new()
    }
}
