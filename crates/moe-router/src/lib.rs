//! MoE expert routing: sigmoid scoring, top-k selection, bias-based load balancing.
//!
//! Implements the DeepSeek-V3/Kimi-K2 routing style:
//! - Sigmoid gate (not softmax) for independent expert scoring
//! - Top-k selection with bias-based load balancing
//! - Optional normalization of selected expert weights
//! - Prefetch prediction via `predict_next()` for cross-layer lookahead

/// Router configuration.
#[derive(Clone, Debug)]
pub struct RouterConfig {
    pub num_experts: usize,
    pub top_k: usize,
    /// If true, normalize selected expert weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Bias learning rate for load balancing (0 = no balancing).
    pub bias_lr: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            num_experts: 128,
            top_k: 8,
            norm_topk_prob: true,
            bias_lr: 0.001,
        }
    }
}

/// Result of routing: which experts to use and their weights.
#[derive(Clone, Debug)]
pub struct RouteResult {
    /// Selected expert indices, sorted by score descending. Length = top_k.
    pub expert_ids: Vec<usize>,
    /// Corresponding weights (sigmoid scores, optionally normalized). Length = top_k.
    pub weights: Vec<f32>,
    /// Raw scores for all experts (for prefetch prediction). Length = num_experts.
    pub all_scores: Vec<f32>,
}

/// MoE router with sigmoid scoring and bias-based load balancing.
pub struct MoeRouter {
    config: RouterConfig,
    /// Per-expert bias for load balancing. Updated after each route() call.
    biases: Vec<f32>,
    /// Running count of how many times each expert has been selected.
    usage_counts: Vec<u64>,
    /// Total number of route() calls.
    total_calls: u64,
}

impl MoeRouter {
    pub fn new(config: RouterConfig) -> Self {
        let n = config.num_experts;
        Self {
            config,
            biases: vec![0.0; n],
            usage_counts: vec![0; n],
            total_calls: 0,
        }
    }

    /// Route a hidden state through the gate.
    /// `gate_logits` has length `num_experts` — the raw gate output for this token.
    /// Returns the top-k experts and their weights.
    pub fn route(&mut self, gate_logits: &[f32]) -> RouteResult {
        assert_eq!(gate_logits.len(), self.config.num_experts);

        // Sigmoid scoring with bias
        let all_scores: Vec<f32> = gate_logits
            .iter()
            .zip(self.biases.iter())
            .map(|(&logit, &bias)| sigmoid(logit + bias))
            .collect();

        // Top-k selection
        let (expert_ids, mut weights) = top_k(&all_scores, self.config.top_k);

        // Optional normalization
        if self.config.norm_topk_prob {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }

        // Update load balancing biases
        if self.config.bias_lr > 0.0 {
            self.total_calls += 1;
            for &eid in &expert_ids {
                self.usage_counts[eid] += 1;
            }
            self.update_biases();
        }

        RouteResult { expert_ids, weights, all_scores }
    }

    /// Route using softmax scoring (Qwen3 style).
    /// gate_logits → softmax over all experts → top-k → renormalize to sum=1.0
    pub fn route_softmax(&mut self, gate_logits: &[f32]) -> RouteResult {
        assert_eq!(gate_logits.len(), self.config.num_experts);

        // Numerically stable softmax
        let max_val = gate_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = gate_logits.iter().map(|&l| (l - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let all_scores: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Top-k selection
        let (expert_ids, mut weights) = top_k(&all_scores, self.config.top_k);

        // Renormalize top-k to sum=1.0
        if self.config.norm_topk_prob {
            let wsum: f32 = weights.iter().sum();
            if wsum > 0.0 {
                for w in &mut weights {
                    *w /= wsum;
                }
            }
        }

        // Update load balancing biases (same as sigmoid route)
        if self.config.bias_lr > 0.0 {
            self.total_calls += 1;
            for &eid in &expert_ids {
                self.usage_counts[eid] += 1;
            }
            self.update_biases();
        }

        RouteResult { expert_ids, weights, all_scores }
    }

    /// Predict which experts layer L+2 will likely use, based on current layer's scores.
    /// Returns top-k expert indices sorted by predicted likelihood.
    /// This is a simple heuristic: experts with high scores at layer L tend to stay hot.
    pub fn predict_next(&self, current_scores: &[f32], lookahead_k: usize) -> Vec<usize> {
        top_k(current_scores, lookahead_k).0
    }

    /// Current per-expert usage fraction.
    pub fn usage_fractions(&self) -> Vec<f32> {
        if self.total_calls == 0 {
            return vec![0.0; self.config.num_experts];
        }
        let expected = self.total_calls as f32 * self.config.top_k as f32;
        self.usage_counts
            .iter()
            .map(|&c| c as f32 / expected * self.config.num_experts as f32)
            .collect()
    }

    /// Reset usage statistics.
    pub fn reset_stats(&mut self) {
        self.usage_counts.fill(0);
        self.total_calls = 0;
    }

    fn update_biases(&mut self) {
        if self.total_calls == 0 {
            return;
        }
        let n = self.config.num_experts as f32;
        let expected_frac = self.config.top_k as f32 / n;
        let total = self.total_calls as f32;

        for i in 0..self.config.num_experts {
            let actual_frac = self.usage_counts[i] as f32 / total;
            // Decrease bias for overused experts, increase for underused
            let error = expected_frac - actual_frac;
            self.biases[i] += self.config.bias_lr * error;
        }
    }
}

/// DeepSeek-V3 sigmoid routing with grouped top-k.
///
/// Algorithm from FINAL.md:
/// 1. scores = sigmoid(logits)
/// 2. biased = scores + e_score_correction_bias (element-wise, bias frozen)
/// 3. Group score = sum of top-2 biased scores within each group
/// 4. Select top topk_group groups
/// 5. From those groups only, select top_k experts by biased score
/// 6. Output weights = unbiased scores[selected], normalized, scaled by routed_scaling_factor
pub fn route_sigmoid_v3(
    gate_logits: &[f32],
    bias: &[f32],
    n_group: usize,
    topk_group: usize,
    num_top_k: usize,
    scaling_factor: f32,
) -> RouteResult {
    let num_experts = gate_logits.len();
    assert_eq!(bias.len(), num_experts);
    assert_eq!(num_experts % n_group, 0);
    assert!(topk_group <= n_group,
        "topk_group ({topk_group}) > n_group ({n_group}) — K2 uses n_group=1, topk_group must be 1");
    let group_size = num_experts / n_group;

    // 1. Sigmoid scores (unbiased)
    let scores: Vec<f32> = gate_logits.iter().map(|&l| sigmoid(l)).collect();

    // 2. Biased scores (for selection only)
    let biased: Vec<f32> = scores.iter().zip(bias.iter())
        .map(|(&s, &b)| s + b).collect();

    // 3. Group score = sum of top-2 biased scores within each group
    let mut group_scores = vec![0.0f32; n_group];
    for g in 0..n_group {
        let start = g * group_size;
        let group = &biased[start..start + group_size];
        let (_, top2_vals) = top_k(group, 2);
        group_scores[g] = top2_vals.iter().sum();
    }

    // 4. Select top topk_group groups
    let (top_groups, _) = top_k(&group_scores, topk_group);

    // 5. Build mask of allowed experts (from selected groups)
    let mut allowed = vec![false; num_experts];
    for &g in &top_groups {
        let start = g * group_size;
        for i in start..start + group_size {
            allowed[i] = true;
        }
    }

    // 6. Select num_top_k experts from allowed, by biased score
    let mut candidates: Vec<(usize, f32)> = biased.iter().copied().enumerate()
        .filter(|(i, _)| allowed[*i])
        .collect();
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(num_top_k);

    let expert_ids: Vec<usize> = candidates.iter().map(|(i, _)| *i).collect();

    // 7. Output weights = unbiased scores, normalized, scaled
    let mut weights: Vec<f32> = expert_ids.iter().map(|&i| scores[i]).collect();
    let wsum: f32 = weights.iter().sum::<f32>() + 1e-20;
    for w in &mut weights {
        *w = (*w / wsum) * scaling_factor;
    }

    RouteResult {
        expert_ids,
        weights,
        all_scores: scores,
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Select top-k indices and values from scores, sorted by score descending.
fn top_k(scores: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(scores.len());
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // Partial sort: move top-k to front
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);
    // Sort the top-k by score descending
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn top_k_basic() {
        let scores = vec![0.1, 0.9, 0.5, 0.7, 0.3];
        let (ids, vals) = top_k(&scores, 3);
        assert_eq!(ids, vec![1, 3, 2]); // indices of 0.9, 0.7, 0.5
        assert!((vals[0] - 0.9).abs() < 1e-6);
        assert!((vals[1] - 0.7).abs() < 1e-6);
        assert!((vals[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_gate_weights_sum_to_one() {
        let config = RouterConfig {
            num_experts: 128,
            top_k: 8,
            norm_topk_prob: true,
            bias_lr: 0.0,
        };
        let mut router = MoeRouter::new(config);
        let logits: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let result = router.route_softmax(&logits);
        let wsum: f32 = result.weights.iter().sum();
        assert!(
            (wsum - 1.0).abs() < 1e-5,
            "gate weights should sum to 1.0, got {wsum}"
        );
        assert_eq!(result.expert_ids.len(), 8);
        // Top experts should be the highest-logit ones (indices 120-127)
        assert!(result.expert_ids[0] == 127, "top expert should be 127");
    }

    #[test]
    fn softmax_vs_sigmoid_different() {
        let config = RouterConfig {
            num_experts: 8,
            top_k: 3,
            norm_topk_prob: true,
            bias_lr: 0.0,
        };
        let mut router = MoeRouter::new(config);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sig_result = router.route(&logits);
        let soft_result = router.route_softmax(&logits);
        // Both should pick the same top-3 experts
        assert_eq!(sig_result.expert_ids, soft_result.expert_ids);
        // But weights differ (softmax concentrates more on top)
    }

    #[test]
    fn top_k_equal_scores() {
        let scores = vec![0.5; 10];
        let (ids, _vals) = top_k(&scores, 3);
        assert_eq!(ids.len(), 3);
    }
}
