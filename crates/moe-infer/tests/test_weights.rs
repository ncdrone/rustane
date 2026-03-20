//! Weight loading integration tests.

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

const WEIGHTS_DIR: &str = "weights/rustane-qwen3";

#[test]
#[ignore = "requires converted weights"]
fn test_embedding_load() {
    let dir = ws_root().join(WEIGHTS_DIR);
    let weights = moe_infer::weights::BackboneWeights::load(&dir).expect("load backbone");

    let emb = weights.embedding(0).expect("token 0 embedding");
    assert_eq!(emb.len(), 2048, "embedding dim should be 2048");
    assert!(
        emb.iter().any(|v| v.to_f32() != 0.0),
        "embedding should not be all zeros"
    );
}

#[test]
#[ignore = "requires converted weights"]
fn test_layer_0_dense() {
    let dir = ws_root().join(WEIGHTS_DIR);
    let weights = moe_infer::weights::BackboneWeights::load(&dir).expect("load backbone");

    let lw = weights.layer_weights(0).expect("layer 0 weights");
    assert_eq!(lw.input_norm.len(), 2048);
    assert_eq!(lw.q_proj.len(), 4096 * 2048); // [32*128, 2048]
    assert_eq!(lw.k_proj.len(), 512 * 2048);  // [4*128, 2048]
    assert_eq!(lw.v_proj.len(), 512 * 2048);
    assert_eq!(lw.o_proj.len(), 2048 * 4096);
    assert_eq!(lw.q_norm.len(), 128); // head_dim
    assert_eq!(lw.k_norm.len(), 128);

    // Layer 0 is dense
    assert!(lw.gate_proj.is_some(), "layer 0 should have dense FFN gate_proj");
    assert!(lw.up_proj.is_some());
    assert!(lw.down_proj.is_some());
    assert!(lw.router.is_none(), "layer 0 should NOT have router");
}

#[test]
#[ignore = "requires converted weights"]
fn test_layer_1_moe() {
    let dir = ws_root().join(WEIGHTS_DIR);
    let weights = moe_infer::weights::BackboneWeights::load(&dir).expect("load backbone");

    let lw = weights.layer_weights(1).expect("layer 1 weights");
    assert_eq!(lw.input_norm.len(), 2048);
    assert_eq!(lw.q_proj.len(), 4096 * 2048);

    // Layer 1 is MoE
    assert!(lw.router.is_some(), "MoE layer should have router");
    assert!(lw.shared_gate_proj.is_some(), "MoE layer should have shared expert");
    assert!(lw.gate_proj.is_none(), "MoE layer should NOT have dense FFN");
}

#[test]
#[ignore = "requires converted weights"]
fn test_final_norm_and_lm_head() {
    let dir = ws_root().join(WEIGHTS_DIR);
    let weights = moe_infer::weights::BackboneWeights::load(&dir).expect("load backbone");

    let norm = weights.final_norm().expect("final norm");
    assert_eq!(norm.len(), 2048);

    let lm_head = weights.lm_head().expect("lm head");
    assert_eq!(lm_head.len(), 151936 * 2048); // [vocab, hidden]
}

#[test]
#[ignore = "requires converted weights"]
fn test_expert_file_exists() {
    let dir = ws_root().join(WEIGHTS_DIR);
    let weights = moe_infer::weights::BackboneWeights::load(&dir).expect("load backbone");

    // Layer 1 should have an expert file
    let mmap = weights.expert_mmap(1);
    assert!(mmap.is_some(), "layer 1 should have expert file");
    let mmap = mmap.unwrap();

    // Verify stride: 128 experts, each with gate+up+down
    let stride = mmap.len() / 128;
    println!("Expert stride: {stride} bytes ({:.2} MB)", stride as f64 / 1e6);
    assert!(stride > 0);
}
