//! Generation integration tests.

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires converted weights + tokenizer"]
fn test_generate_one_token() {
    use moe_infer::generate::{Model, SamplingConfig};

    let root = ws_root();
    let model = Model::load(
        &root.join("weights/rustane-qwen3"),
        &root.join("configs/qwen3-moe-30b.toml"),
    )
    .expect("load model");

    let tok = tokenizers::Tokenizer::from_file(root.join("weights/qwen3-30b-a3b/tokenizer.json"))
        .expect("load tokenizer");

    let output = moe_infer::generate::generate(
        &model,
        &tok,
        "The capital of France is",
        1,
        &SamplingConfig::greedy(),
    );

    assert!(output.is_ok(), "generation should not error: {:?}", output.err());
    let out = output.unwrap();
    assert!(out.tokens_generated >= 1, "should generate at least 1 token");
    println!("Generated: {}", out.text);
}

#[test]
#[ignore = "requires converted weights + tokenizer"]
fn test_generate_20_tokens() {
    use moe_infer::generate::{Model, SamplingConfig};

    let root = ws_root();
    let model = Model::load(
        &root.join("weights/rustane-qwen3"),
        &root.join("configs/qwen3-moe-30b.toml"),
    )
    .expect("load model");

    let tok = tokenizers::Tokenizer::from_file(root.join("weights/qwen3-30b-a3b/tokenizer.json"))
        .expect("load tokenizer");

    let output = moe_infer::generate::generate(
        &model,
        &tok,
        "The capital of France is",
        20,
        &SamplingConfig::greedy(),
    )
    .expect("generate");

    println!("Generated {} tokens: {}", output.tokens_generated, output.text);
    assert!(output.tokens_generated >= 5, "should generate multiple tokens");
}

#[test]
#[ignore = "requires converted weights + HF references"]
fn test_generation_matches_hf() {
    use moe_infer::generate::{Model, SamplingConfig};

    let root = ws_root();
    let model = Model::load(
        &root.join("weights/rustane-qwen3"),
        &root.join("configs/qwen3-moe-30b.toml"),
    )
    .expect("load model");

    let tok = tokenizers::Tokenizer::from_file(root.join("weights/qwen3-30b-a3b/tokenizer.json"))
        .expect("load tokenizer");

    // Load HF reference
    let ref_path = root.join("weights/references/greedy_generation.json");
    let ref_content = std::fs::read_to_string(&ref_path).expect("read generation ref");
    let ref_data: serde_json::Value = serde_json::from_str(&ref_content).expect("parse ref");
    let ref_generated: Vec<u32> = ref_data["generated_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let prompt = ref_data["prompt"].as_str().unwrap();

    // Generate with same prompt
    let output = moe_infer::generate::generate(
        &model,
        &tok,
        prompt,
        20,
        &SamplingConfig::greedy(),
    )
    .expect("generate");

    let our_generated = &output.token_ids;

    let match_count = ref_generated
        .iter()
        .zip(our_generated.iter())
        .take_while(|(a, b)| a == b)
        .count();

    println!(
        "Token match: {}/{} (HF: {:?}, ours: {:?})",
        match_count,
        ref_generated.len().min(our_generated.len()),
        &ref_generated[..ref_generated.len().min(10)],
        &our_generated[..our_generated.len().min(10)]
    );

    assert!(
        match_count >= 15,
        "greedy output should match HF for at least 15/20 tokens, got {match_count}"
    );
}
