//! Benchmark: DeepSeek-V2-Lite tok/s on M4 Max.
//!
//! Measures decode throughput (single token at a time, sequential layers).
//! Baseline for MLA attention path — to be optimized with Metal W_UK kernel.
//!
//! Run: cargo test -p moe-infer --test bench_v2_lite_tok_per_sec --release -- --ignored --nocapture

use std::path::Path;
use std::time::Instant;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires converted weights + tokenizer"]
fn bench_v2_lite_decode_throughput() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-deepseek-v2-lite");
    let config_path = root.join("configs/deepseek-v2-lite.toml");
    let tokenizer_path = root.join("weights/deepseek-v2-lite/tokenizer.json");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: converted weights not found");
        return;
    }
    if !tokenizer_path.exists() {
        eprintln!("SKIP: tokenizer not found");
        return;
    }

    eprintln!("Loading model...");
    let t = Instant::now();
    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load model");
    eprintln!("Model loaded in {:.1}s", t.elapsed().as_secs_f64());

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("load tokenizer");

    let sampling = moe_infer::generate::SamplingConfig::greedy();

    // Warm-up: generate 5 tokens
    eprintln!("Warm-up...");
    let _ = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, "Hello", 5, &sampling,
    );

    // Benchmark: generate 20 tokens, measure decode time
    eprintln!("Benchmarking 20-token generation...");
    let prompt = "The quick brown fox";

    let t = Instant::now();
    let output = moe_infer::generate_v2::generate_v2(
        &model, &tokenizer, prompt, 20, &sampling,
    ).expect("generation");
    let total_secs = t.elapsed().as_secs_f64();

    let decode_tokens = output.tokens_generated;
    let decode_secs = output.decode_secs;
    let tok_per_sec = if decode_secs > 0.0 { decode_tokens as f64 / decode_secs } else { 0.0 };
    let ms_per_tok = if decode_tokens > 0 { decode_secs * 1000.0 / decode_tokens as f64 } else { 0.0 };

    eprintln!("\n=== DeepSeek-V2-Lite Benchmark ===");
    eprintln!("Prompt: {prompt:?} ({} tokens)", output.prompt_tokens);
    eprintln!("Generated: {} tokens", decode_tokens);
    eprintln!("Generated text: {:?}", output.text);
    eprintln!("Prefill: {:.2}s", output.prefill_secs);
    eprintln!("Decode:  {:.2}s ({tok_per_sec:.1} tok/s, {ms_per_tok:.0} ms/tok)", decode_secs);
    eprintln!("Total:   {:.2}s", total_secs);
    eprintln!("==================================");

    // Record for experiments log
    eprintln!("\nFor experiments-infer.tsv:");
    eprintln!("2026-03-21\tv2-lite-baseline\tmodel=DeepSeek-V2-Lite\t{tok_per_sec:.1} tok/s\t{ms_per_tok:.0} ms/tok\tbaseline");
}
