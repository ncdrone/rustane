//! Attention integration tests: RoPE, single-head, GQA, causal mask.

use moe_infer::attention::{GqaConfig, RopeTables};

#[test]
fn rope_tables_basic() {
    let rope = RopeTables::build(1024, 128, 1_000_000.0);

    // Position 0: identity rotation
    let mut x = vec![0.0f32; 128];
    x[0] = 1.0;
    let original = x.clone();
    rope.apply(&mut x, 0);
    // cos(0) = 1, sin(0) = 0, so x should be unchanged
    for i in 0..128 {
        assert!(
            (x[i] - original[i]).abs() < 1e-6,
            "RoPE at pos 0 should be identity, dim {i}"
        );
    }
}

#[test]
fn rope_rotation_property() {
    let rope = RopeTables::build(1024, 128, 1_000_000.0);

    // After RoPE, the norm should be preserved
    let mut x: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    rope.apply(&mut x, 42);
    let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_before - norm_after).abs() < 1e-4,
        "RoPE should preserve norm: {norm_before} vs {norm_after}"
    );
}

#[test]
fn gqa_config_group_size() {
    let cfg = GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 4,
        head_dim: 128,
        max_seq: 4096,
    };
    assert_eq!(cfg.group_size(), 8, "32 Q / 4 KV = group size 8");
}

// Integration tests that need real weights are below

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::path::Path::new(&manifest)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
#[ignore = "requires converted weights"]
fn test_gqa_forward_one_token() {
    use moe_infer::attention::gqa_forward;
    use moe_infer::kv_cache::KvCache;
    use moe_infer::weights::BackboneWeights;

    let dir = ws_root().join("weights/rustane-qwen3");
    let weights = BackboneWeights::load(&dir).expect("load backbone");
    let lw = weights.layer_weights(1).expect("layer 1");
    let mut cache = KvCache::new(48, 4, 128, 4096);
    let rope = RopeTables::build(4096, 128, 1_000_000.0);
    let cfg = GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 4,
        head_dim: 128,
        max_seq: 4096,
    };

    let x = vec![0.01f32; 2048];
    let out = gqa_forward(&x, &lw, &mut cache, 1, 0, &rope, &cfg, 1e-6);

    assert_eq!(out.len(), 2048);
    assert!(out.iter().all(|v| v.is_finite()), "NaN/Inf in attention output");
    assert!(out.iter().any(|v| *v != 0.0), "all-zero attention output");
}

#[test]
#[ignore = "requires converted weights"]
fn test_causal_mask_multi_token() {
    use moe_infer::attention::gqa_forward;
    use moe_infer::kv_cache::KvCache;
    use moe_infer::weights::BackboneWeights;

    let dir = ws_root().join("weights/rustane-qwen3");
    let weights = BackboneWeights::load(&dir).expect("load backbone");
    let lw = weights.layer_weights(1).expect("layer 1");
    let mut cache = KvCache::new(48, 4, 128, 4096);
    let rope = RopeTables::build(4096, 128, 1_000_000.0);
    let cfg = GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 4,
        head_dim: 128,
        max_seq: 4096,
    };

    // Process 5 tokens sequentially
    let mut outputs = Vec::new();
    for pos in 0..5 {
        let x: Vec<f32> = (0..2048).map(|i| ((i + pos * 100) as f32).sin() * 0.01).collect();
        let out = gqa_forward(&x, &lw, &mut cache, 1, pos, &rope, &cfg, 1e-6);
        assert_eq!(out.len(), 2048);
        assert!(out.iter().all(|v| v.is_finite()), "NaN at pos {pos}");
        outputs.push(out);
    }

    // Later positions should differ from earlier ones (KV cache is building up)
    let diff: f32 = outputs[0]
        .iter()
        .zip(outputs[4].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.001, "outputs at pos 0 and 4 should differ (diff={diff})");
}
