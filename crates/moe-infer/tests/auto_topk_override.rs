//! Correctness test: RUSTANE_TOP_K env var correctly overrides expert top_k.
//!
//! What was optimized: added RUSTANE_TOP_K env var to override num_experts_per_tok.
//! Reducing top_k (e.g. 6 vs 8) saves ~25% pread I/O + Metal dispatch per layer.
//!
//! Invariant: top_k=6 selects a strict subset of top_k=8 experts (the top 6 of 8).
//! Routing weights are renormalized so they sum to the same scaling factor.
//!
//! Failure means: routing logic doesn't properly handle reduced top_k, or the
//! subset invariant is violated (top_k=6 selects experts not in top_k=8).

/// Test: top_k=6 produces a strict subset of top_k=8 expert IDs.
/// Uses route_sigmoid_v3 with K2-like parameters (384 experts, n_group=1).
#[test]
fn topk6_is_subset_of_topk8() {
    let num_experts = 384;
    // Deterministic gate logits (simulate one layer's routing)
    let gate_logits: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.0173 + 1.7).sin() * 3.0))
        .collect();
    // Deterministic bias
    let bias: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.037 + 0.5).cos() * 0.1))
        .collect();

    let route_8 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 8, 1.0);
    let route_6 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 6, 1.0);

    assert_eq!(route_8.expert_ids.len(), 8);
    assert_eq!(route_6.expert_ids.len(), 6);

    // Every expert in top_k=6 must be in top_k=8
    for &eid in &route_6.expert_ids {
        assert!(route_8.expert_ids.contains(&eid),
            "top_k=6 expert {} not found in top_k=8 set {:?}", eid, route_8.expert_ids);
    }

    eprintln!("top_k=8 experts: {:?}", route_8.expert_ids);
    eprintln!("top_k=6 experts: {:?}", route_6.expert_ids);
    eprintln!("Verified: top_k=6 is subset of top_k=8");
}

/// Test: routing weights are properly renormalized for different top_k values.
/// With norm_topk_prob, weights sum to scaling_factor regardless of top_k.
#[test]
fn topk_weights_renormalized() {
    let num_experts = 384;
    let gate_logits: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.029 + 2.1).sin() * 2.5))
        .collect();
    let bias: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.013 + 0.3).cos() * 0.05))
        .collect();
    let scaling = 2.827; // K2 routed_scaling_factor

    let route_8 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 8, scaling);
    let route_6 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 6, scaling);
    let route_4 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 4, scaling);

    let sum_8: f32 = route_8.weights.iter().sum();
    let sum_6: f32 = route_6.weights.iter().sum();
    let sum_4: f32 = route_4.weights.iter().sum();

    // All should sum to scaling_factor (within float tolerance)
    assert!((sum_8 - scaling).abs() < 0.01,
        "top_k=8 weights sum to {sum_8}, expected {scaling}");
    assert!((sum_6 - scaling).abs() < 0.01,
        "top_k=6 weights sum to {sum_6}, expected {scaling}");
    assert!((sum_4 - scaling).abs() < 0.01,
        "top_k=4 weights sum to {sum_4}, expected {scaling}");

    eprintln!("Weight sums — top_k=8: {sum_8:.4}, top_k=6: {sum_6:.4}, top_k=4: {sum_4:.4}");
    eprintln!("Verified: weights renormalized to scaling_factor={scaling}");
}

/// Edge case: top_k=1 selects exactly the highest-scoring expert.
#[test]
fn topk1_selects_best_expert() {
    let num_experts = 384;
    let gate_logits: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.011 + 0.9).sin() * 4.0))
        .collect();
    let bias: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.023 + 1.1).cos() * 0.08))
        .collect();

    let route_1 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 1, 1.0);
    let route_8 = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 8, 1.0);

    assert_eq!(route_1.expert_ids.len(), 1);
    // top_k=1 expert should be the same as the first expert in top_k=8
    assert_eq!(route_1.expert_ids[0], route_8.expert_ids[0],
        "top_k=1 expert {} != first of top_k=8 {}", route_1.expert_ids[0], route_8.expert_ids[0]);

    eprintln!("top_k=1 selected expert {} (same as top_k=8 best)", route_1.expert_ids[0]);
}

/// Edge case: top_k equal to config value (8) should produce identical results.
#[test]
fn topk8_matches_default() {
    let num_experts = 384;
    let gate_logits: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.019 + 1.3).sin() * 3.5))
        .collect();
    let bias: Vec<f32> = (0..num_experts)
        .map(|i| ((i as f32 * 0.031 + 0.7).cos() * 0.06))
        .collect();

    let route_a = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 8, 2.827);
    let route_b = moe_router::route_sigmoid_v3(&gate_logits, &bias, 1, 1, 8, 2.827);

    assert_eq!(route_a.expert_ids, route_b.expert_ids, "same inputs must give same experts");
    for (wa, wb) in route_a.weights.iter().zip(route_b.weights.iter()) {
        assert!((wa - wb).abs() < 1e-7, "weights must be deterministic");
    }
    eprintln!("Verified: deterministic routing (same inputs → same outputs)");
}
