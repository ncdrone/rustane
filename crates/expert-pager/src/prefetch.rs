//! ExpertPrefetcher: predict which experts to preload based on routing patterns.
//!
//! Uses cross-layer gate similarity (Fate-style, zero training cost):
//! Experts that score highly at layer L tend to score highly at layer L+2.
//! Prefetch predicted experts in the background while current layer computes.

use std::collections::HashSet;

/// Prefetch predictor based on routing history.
pub struct ExpertPrefetcher {
    /// Number of experts to prefetch per layer.
    prefetch_k: usize,
    /// Recent routing history: scores from last N layers.
    /// Each entry is (layer_idx, expert_scores).
    history: Vec<(usize, Vec<f32>)>,
    /// Max history depth.
    max_history: usize,
}

impl ExpertPrefetcher {
    pub fn new(prefetch_k: usize) -> Self {
        Self {
            prefetch_k,
            history: Vec::new(),
            max_history: 4,
        }
    }

    /// Record routing scores for a layer.
    pub fn record(&mut self, layer_idx: usize, scores: Vec<f32>) {
        self.history.push((layer_idx, scores));
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Predict which experts to prefetch for `target_layer`.
    /// Returns expert indices sorted by predicted likelihood.
    pub fn predict(&self, target_layer: usize) -> Vec<u32> {
        // Look for scores from 2 layers back (cross-layer similarity)
        let source_layer = target_layer.saturating_sub(2);
        if let Some((_, scores)) = self.history.iter().find(|(l, _)| *l == source_layer) {
            top_k_indices(scores, self.prefetch_k)
        } else if let Some((_, scores)) = self.history.last() {
            // Fallback: use most recent scores
            top_k_indices(scores, self.prefetch_k)
        } else {
            Vec::new()
        }
    }

    /// Compute which experts need loading (predicted - already_resident).
    pub fn experts_to_load(&self, target_layer: usize, resident: &HashSet<u32>) -> Vec<u32> {
        self.predict(target_layer)
            .into_iter()
            .filter(|id| !resident.contains(id))
            .collect()
    }
}

fn top_k_indices(scores: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(scores.len());
    let mut indexed: Vec<(u32, f32)> = scores.iter().copied().enumerate().map(|(i, s)| (i as u32, s)).collect();
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().map(|(i, _)| i).collect()
}
