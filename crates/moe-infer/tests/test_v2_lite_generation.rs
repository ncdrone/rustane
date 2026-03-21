//! E2E generation test: DeepSeek-V2-Lite vs HF greedy reference.
//!
//! Requires:
//! - Converted weights at weights/rustane-deepseek-v2-lite/
//! - Config at configs/deepseek-v2-lite.toml
//! - Tokenizer at weights/deepseek-v2-lite/tokenizer.json
//! - Reference at weights/references/deepseek_v2_lite_greedy.json
//!
//! Run: cargo test -p moe-infer --test test_v2_lite_generation --release -- --ignored --nocapture

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires converted weights + tokenizer"]
fn test_v2_lite_generation_matches_hf() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-deepseek-v2-lite");
    let config_path = root.join("configs/deepseek-v2-lite.toml");
    let tokenizer_path = root.join("weights/deepseek-v2-lite/tokenizer.json");
    let ref_path = root.join("weights/references/deepseek_v2_lite_greedy.json");

    // Check prerequisites
    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: converted weights not found at {}", weights_dir.display());
        eprintln!("Run: cargo run -p moe-infer --release --bin convert_deepseek -- --model-dir weights/deepseek-v2-lite --output-dir weights/rustane-deepseek-v2-lite");
        return;
    }
    if !tokenizer_path.exists() {
        eprintln!("SKIP: tokenizer not found at {}", tokenizer_path.display());
        return;
    }
    if !ref_path.exists() {
        eprintln!("SKIP: reference not found at {}", ref_path.display());
        return;
    }

    // Load reference
    let ref_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&ref_path).unwrap()
    ).unwrap();
    let ref_tokens: Vec<u32> = ref_json["generated_token_ids"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let prompt = ref_json["prompt"].as_str().unwrap();
    eprintln!("Reference: prompt={prompt:?}, {} generated tokens", ref_tokens.len());
    eprintln!("Reference tokens: {ref_tokens:?}");

    // Load model
    eprintln!("Loading model...");
    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load model");

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("load tokenizer");

    // Generate
    let sampling = moe_infer::generate::SamplingConfig::greedy();
    eprintln!("Generating...");
    let output = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, prompt, 20, &sampling,
    ).expect("generation");

    eprintln!("Generated {} tokens in {:.1}s prefill + {:.1}s decode",
        output.tokens_generated, output.prefill_secs, output.decode_secs);
    eprintln!("Generated tokens: {:?}", output.token_ids);
    eprintln!("Generated text: {:?}", output.text);

    // Compare
    let max_compare = ref_tokens.len().min(output.token_ids.len());
    let mut matches = 0;
    for i in 0..max_compare {
        if output.token_ids[i] == ref_tokens[i] {
            matches += 1;
        } else {
            eprintln!("  token {i}: ours={} vs ref={} (MISMATCH)",
                output.token_ids[i], ref_tokens[i]);
        }
    }

    eprintln!("\nMatch: {matches}/{max_compare} tokens ({:.0}%)",
        100.0 * matches as f64 / max_compare as f64);

    // Gate: >=15/20
    assert!(matches >= 15,
        "Expected >=15/20 token matches, got {matches}/{max_compare}");
}
