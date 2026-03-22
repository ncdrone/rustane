//! DeepSeek-V3 throughput benchmark.
//!
//! Measures tok/s for decode, expert cache hit rate, and cold/warm performance.
//! Target: 3-5 tok/s warm, 0.3-0.9 tok/s cold.
//!
//! Run: cargo test -p moe-infer --test bench_v3_tok_per_sec --release -- --ignored --nocapture

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires converted V3 weights"]
fn bench_v3_decode_throughput() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-v3");
    let config_path = root.join("configs/deepseek-v3.toml");
    let tokenizer_path = root.join("weights/deepseek-v3/tokenizer.json");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: V3 weights not found at {}", weights_dir.display());
        return;
    }
    if !tokenizer_path.exists() {
        eprintln!("SKIP: tokenizer not found");
        return;
    }

    eprintln!("Loading DeepSeek-V3 model...");
    let t_load = std::time::Instant::now();
    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load V3 model");
    eprintln!("Model loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("load tokenizer");

    let sampling = moe_infer::generate::SamplingConfig {
        greedy: true,
        temperature: 1.0,
        top_k: 1,
    };

    // Cold run
    eprintln!("\n--- Cold Run ---");
    let output = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, "The capital of France is", 20, &sampling,
    ).expect("generate");

    let cold_tok_s = output.tokens_generated as f64 / output.decode_secs;
    eprintln!("Cold decode: {:.2} tok/s ({} tokens in {:.2}s)",
        cold_tok_s, output.tokens_generated, output.decode_secs);
    eprintln!("Cold prefill: {:.2}s ({} prompt tokens)",
        output.prefill_secs, output.prompt_tokens);
    eprintln!("Output: {:?}", output.text);

    // Warm run
    eprintln!("\n--- Warm Run ---");
    let output2 = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, "In machine learning, a transformer is", 20, &sampling,
    ).expect("generate");

    let warm_tok_s = output2.tokens_generated as f64 / output2.decode_secs;
    eprintln!("Warm decode: {:.2} tok/s ({} tokens in {:.2}s)",
        warm_tok_s, output2.tokens_generated, output2.decode_secs);
    eprintln!("Output: {:?}", output2.text);

    // Summary
    eprintln!("\n=== V3 Benchmark Summary ===");
    eprintln!("Cold: {cold_tok_s:.2} tok/s");
    eprintln!("Warm: {warm_tok_s:.2} tok/s");
    eprintln!("Target: 3-5 tok/s warm, 0.3-0.9 tok/s cold");
}
