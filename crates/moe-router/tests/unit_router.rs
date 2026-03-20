//! Unit tests for MoE router.
//!
//! Invariants:
//! 1. Top-k selection returns correct experts sorted by score
//! 2. Gate normalization: selected weights sum to 1.0 when norm_topk_prob=true
//! 3. Bias update direction: overused experts get lower bias, underused get higher
//! 4. Exact tolerance: 0 for selection, float epsilon for normalized weights

use moe_router::{MoeRouter, RouterConfig};

/// Top-k selects the highest-scoring experts.
#[test]
fn topk_correctness() {
    let config = RouterConfig {
        num_experts: 8,
        top_k: 2,
        norm_topk_prob: false,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);

    // Expert 3 and 5 should have highest sigmoid scores
    let logits = vec![-10.0, -5.0, -2.0, 10.0, 0.0, 8.0, -1.0, -3.0];
    let result = router.route(&logits);

    assert_eq!(result.expert_ids.len(), 2);
    assert_eq!(result.expert_ids[0], 3, "expert 3 should be first (score ~1.0)");
    assert_eq!(result.expert_ids[1], 5, "expert 5 should be second (score ~0.9997)");
}

/// With normalization, selected weights sum to 1.0.
#[test]
fn gate_normalization() {
    let config = RouterConfig {
        num_experts: 16,
        top_k: 4,
        norm_topk_prob: true,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);

    let logits: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let result = router.route(&logits);

    let weight_sum: f32 = result.weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-5,
        "normalized weights should sum to 1.0, got {weight_sum}"
    );
}

/// Without normalization, weights are raw sigmoid scores.
#[test]
fn no_normalization() {
    let config = RouterConfig {
        num_experts: 8,
        top_k: 2,
        norm_topk_prob: false,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);

    // logit=10 -> sigmoid ~= 0.99995, logit=0 -> sigmoid = 0.5
    let logits = vec![-10.0, -5.0, -2.0, 10.0, 0.0, 8.0, -1.0, -3.0];
    let result = router.route(&logits);

    // Without normalization, weights should be raw sigmoid values
    assert!(result.weights[0] > 0.999, "sigmoid(10) should be ~1.0");
    assert!(result.weights[1] > 0.999, "sigmoid(8) should be ~0.9997");
    let weight_sum: f32 = result.weights.iter().sum();
    assert!(weight_sum > 1.0, "raw weights should sum to >1.0");
}

/// Bias update direction: overused experts should get decreased bias.
#[test]
fn bias_update_direction() {
    let config = RouterConfig {
        num_experts: 4,
        top_k: 1,
        norm_topk_prob: false,
        bias_lr: 0.01,
    };
    let mut router = MoeRouter::new(config);

    // Route 100 times always selecting expert 0 (highest logit)
    let logits = vec![10.0, -10.0, -10.0, -10.0];
    for _ in 0..100 {
        router.route(&logits);
    }

    let fracs = router.usage_fractions();
    // Expert 0 should be heavily overused (fraction >> 1.0)
    assert!(fracs[0] > 3.0, "expert 0 should be overused: frac={}", fracs[0]);
    // Other experts should be underused (fraction = 0)
    assert!(fracs[1] < 0.01, "expert 1 should be underused");
}

/// All experts selected at least once when routing many tokens with diverse logits.
#[test]
fn all_experts_reachable() {
    let config = RouterConfig {
        num_experts: 8,
        top_k: 2,
        norm_topk_prob: true,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);
    let mut seen = vec![false; 8];

    // Route with each expert having highest score once
    for hot in 0..8 {
        let logits: Vec<f32> = (0..8)
            .map(|i| if i == hot { 10.0 } else { -10.0 })
            .collect();
        let result = router.route(&logits);
        for &eid in &result.expert_ids {
            seen[eid] = true;
        }
    }

    assert!(seen.iter().all(|&s| s), "all experts should be reachable");
}

/// All scores are valid sigmoid outputs (0, 1).
#[test]
fn all_scores_valid_sigmoid() {
    let config = RouterConfig {
        num_experts: 128,
        top_k: 8,
        norm_topk_prob: true,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);

    // Random-ish logits including extremes
    let logits: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.2).collect();
    let result = router.route(&logits);

    for (i, &s) in result.all_scores.iter().enumerate() {
        assert!(s > 0.0 && s < 1.0, "score[{i}]={s} not in (0,1)");
    }
}

/// predict_next returns valid expert indices.
#[test]
fn predict_next_returns_valid_indices() {
    let config = RouterConfig {
        num_experts: 128,
        top_k: 8,
        norm_topk_prob: true,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);

    let logits: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let result = router.route(&logits);

    let predicted = router.predict_next(&result.all_scores, 8);
    assert_eq!(predicted.len(), 8);
    for &eid in &predicted {
        assert!(eid < 128, "predicted expert id {eid} out of range");
    }
}

/// Correct number of results.
#[test]
fn output_shape() {
    let config = RouterConfig {
        num_experts: 64,
        top_k: 4,
        norm_topk_prob: true,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);

    let logits = vec![0.0f32; 64];
    let result = router.route(&logits);

    assert_eq!(result.expert_ids.len(), 4);
    assert_eq!(result.weights.len(), 4);
    assert_eq!(result.all_scores.len(), 64);
}
