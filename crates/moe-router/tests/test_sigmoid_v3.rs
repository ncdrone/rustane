//! V3 prep: sigmoid routing with grouped top-k.

use moe_router::route_sigmoid_v3;

#[test]
fn sigmoid_v3_basic_routing() {
    // 16 experts, 4 groups of 4, select top-2 groups, top-4 experts
    let logits: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let bias = vec![0.0f32; 16]; // no bias

    let result = route_sigmoid_v3(&logits, &bias, 4, 2, 4, 1.0);

    assert_eq!(result.expert_ids.len(), 4);
    // Top experts should come from groups 2 and 3 (highest logits: 8-15)
    for &eid in &result.expert_ids {
        assert!(eid >= 8, "expert {eid} should be from top groups (8-15)");
    }
    // Expert 15 should be first (highest logit)
    assert_eq!(result.expert_ids[0], 15);
    eprintln!("Selected experts: {:?}", result.expert_ids);
    eprintln!("Weights: {:?}", result.weights);
}

#[test]
fn sigmoid_v3_weights_are_unbiased() {
    // Bias shifts selection but output weights should use unbiased scores
    let logits = vec![0.0f32; 16];
    let mut bias = vec![0.0f32; 16];
    // Bias expert 0 way up to force its selection
    bias[0] = 10.0;

    let result = route_sigmoid_v3(&logits, &bias, 4, 2, 4, 1.0);

    // Expert 0 should be selected (due to bias)
    assert!(result.expert_ids.contains(&0), "expert 0 should be selected due to bias");

    // But its weight should be based on unbiased sigmoid(0.0) = 0.5
    // All unbiased scores are sigmoid(0) = 0.5, so all weights should be equal
    let w0 = result.weights[result.expert_ids.iter().position(|&e| e == 0).unwrap()];
    let expected = 1.0 / 4.0; // 4 experts, each 0.5, normalized → 0.25, scaled by 1.0
    assert!((w0 - expected).abs() < 0.01,
        "weight for expert 0 should be ~{expected:.3} (unbiased), got {w0:.3}");
}

#[test]
fn sigmoid_v3_scaling_factor() {
    let logits: Vec<f32> = (0..16).map(|i| i as f32 * 0.3).collect();
    let bias = vec![0.0f32; 16];

    let result = route_sigmoid_v3(&logits, &bias, 4, 2, 4, 2.5);

    // Weights should sum to ~scaling_factor (2.5)
    let wsum: f32 = result.weights.iter().sum();
    assert!((wsum - 2.5).abs() < 0.01,
        "weights should sum to ~2.5, got {wsum:.4}");
}

#[test]
fn sigmoid_v3_group_selection() {
    // Make one group clearly dominant
    let mut logits = vec![-5.0f32; 32]; // 32 experts, 8 groups of 4
    // Group 3 (experts 12-15) gets high logits
    for i in 12..16 {
        logits[i] = 5.0;
    }
    // Group 7 (experts 28-31) also gets high logits
    for i in 28..32 {
        logits[i] = 4.0;
    }
    let bias = vec![0.0f32; 32];

    let result = route_sigmoid_v3(&logits, &bias, 8, 2, 4, 1.0);

    // All 4 selected experts should come from groups 3 and 7
    for &eid in &result.expert_ids {
        let group = eid / 4;
        assert!(group == 3 || group == 7,
            "expert {eid} (group {group}) should be from group 3 or 7");
    }
}
