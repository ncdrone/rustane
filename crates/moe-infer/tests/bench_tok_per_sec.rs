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

    println!("=== BENCHMARK RESULTS ===");
    println!(
        "Generated {} tokens in {:.1}s = {:.1} tok/s",
        output.tokens_generated,
        elapsed.as_secs_f64(),
        tok_per_sec
    );
    println!("Output: {}", output.text);
    println!("=========================");
}
