//! THE metric: tokens per second on real Qwen3-MoE-30B.

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "benchmark: requires converted weights + tokenizer"]
fn bench_tok_per_sec() {
    use moe_infer::generate::{Model, SamplingConfig};

    let root = ws_root();
    let model = Model::load(
        &root.join("weights/rustane-qwen3"),
        &root.join("configs/qwen3-moe-30b.toml"),
    )
    .expect("load model");

    let tok = tokenizers::Tokenizer::from_file(root.join("weights/qwen3-30b-a3b/tokenizer.json"))
        .expect("load tokenizer");

    let prompt = "Explain what a mixture of experts model is in one paragraph.";
    let sampling = SamplingConfig::greedy();

    // Warmup
    println!("Warmup...");
    let warmup = moe_infer::generate::generate(&model, &tok, "Hello", 3, &sampling);
    if let Err(e) = &warmup {
        eprintln!("Warmup failed: {e}");
    }

    // Timed run
    println!("Benchmarking...");
    let t0 = std::time::Instant::now();
    let output = moe_infer::generate::generate(&model, &tok, prompt, 50, &sampling)
        .expect("generate");
    let elapsed = t0.elapsed();

    let tok_per_sec = output.tokens_generated as f64 / elapsed.as_secs_f64();
    let decode_tok_per_sec = if output.decode_secs > 0.0 {
        (output.tokens_generated.saturating_sub(1)) as f64 / output.decode_secs
    } else { 0.0 };

    println!("=== BENCHMARK RESULTS ===");
    println!(
        "Generated {} tokens in {:.1}s = {:.1} tok/s",
        output.tokens_generated,
        elapsed.as_secs_f64(),
        tok_per_sec
    );
    println!(
        "  Prefill: {} tokens in {:.2}s ({:.0} tok/s)",
        output.prompt_tokens,
        output.prefill_secs,
        output.prompt_tokens as f64 / output.prefill_secs.max(0.001),
    );
    println!(
        "  Decode: {} tokens in {:.2}s ({:.1} tok/s)",
        output.tokens_generated.saturating_sub(1),
        output.decode_secs,
        decode_tok_per_sec,
    );
    println!("Output: {}", output.text);
    println!("=========================");
}
