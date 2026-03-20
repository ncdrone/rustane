//! Benchmark: scale validation ladder.
//!
//! Estimates throughput and memory for MoE models at various scales.
//! Does not require actual weights — uses synthetic data to validate
//! that the pipeline can handle the data volumes.
//!
//! Run with: cargo test -p moe-infer --test bench_scale --release -- --ignored --nocapture

use moe_infer::pipeline::MoePipeline;
use std::time::Instant;

/// Scale validation: measure pipeline throughput at increasing model sizes.
#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn scale_ladder() {
    println!("\n{}", "=".repeat(70));
    println!("  MoE Pipeline Scale Ladder (CPU reference)");
    println!("{}", "=".repeat(70));
    println!(
        "  {:20} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "Config", "Dim", "Hidden", "Experts", "Time (ms)", "tok/s"
    );
    println!("{}", "-".repeat(70));

    let configs = [
        ("Tiny (test)", 128, 256, 8, 2, 256, 2),
        ("Small (30B-like)", 256, 512, 16, 4, 512, 1),
        ("Medium (100B-like)", 512, 1024, 32, 4, 1024, 1),
    ];

    for (label, dim, hidden, n_experts, top_k, vocab, n_layers) in configs {
        let mut pipeline = MoePipeline::random(dim, hidden, n_layers, n_experts, top_k, vocab);

        // Warmup
        pipeline.forward(0);

        // Timed
        let n_tokens = 10;
        let t0 = Instant::now();
        for t in 0..n_tokens {
            pipeline.forward(t % vocab);
        }
        let elapsed = t0.elapsed();

        let ms_per_tok = elapsed.as_secs_f64() * 1000.0 / n_tokens as f64;
        let tok_per_sec = n_tokens as f64 / elapsed.as_secs_f64();

        println!(
            "  {:20} {:>8} {:>8} {:>8} {:>10.1} {:>10.1}",
            label, dim, hidden, n_experts, ms_per_tok, tok_per_sec,
        );
    }

    println!("{}", "-".repeat(70));
    println!("  Note: CPU reference only. Real throughput with ANE/Metal will be much higher.");
    println!();
}

/// Memory estimation for various model scales.
#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn memory_estimates() {
    println!("\n{}", "=".repeat(70));
    println!("  MoE Model Memory Estimates");
    println!("{}", "=".repeat(70));

    let models: &[(&str, u64, u64, u64, u64, u64)] = &[
        ("Qwen3-MoE-30B", 2048, 768, 48, 128, 8),
        ("DeepSeek-V3 (700B)", 7168, 2048, 61, 256, 8),
        ("Target 1T", 7168, 2048, 61, 384, 8),
        ("Kimi-K2 (1T)", 7168, 2048, 61, 384, 8),
    ];

    for &(name, dim, expert_hidden, layers, experts, top_k) in models {
        let expert_params = expert_hidden * dim * 3; // gate + up + down
        let expert_bytes_4bit = expert_params / 2;
        let expert_bytes_2bit = expert_params / 4;
        let total_4bit = experts * expert_bytes_4bit * layers;
        let total_2bit = experts * expert_bytes_2bit * layers;
        let active_4bit = top_k * expert_bytes_4bit;

        // Attention weights (non-MoE): ~dim^2 * 4 * layers * 0.5 bytes (4-bit)
        let attn_params = dim * dim * 4; // Q, K, V, O
        let attn_bytes = attn_params * layers / 2;

        println!("\n  {name}:");
        println!("    Expert weights (4-bit): {:.1} GB", total_4bit as f64 / 1e9);
        println!("    Expert weights (2-bit): {:.1} GB", total_2bit as f64 / 1e9);
        println!("    Active per token (4-bit): {:.1} MB", active_4bit as f64 / 1e6);
        println!("    Attention weights (4-bit): {:.1} GB", attn_bytes as f64 / 1e9);
        println!("    Total (mixed 4/2-bit): {:.1} GB",
            (total_4bit / 2 + total_2bit / 2 + attn_bytes) as f64 / 1e9);
    }

    println!("\n{}", "-".repeat(70));
    println!("  M4 Max 128GB budget: ~40 GB expert cache + 12 GB attention + 8 GB KV");
    println!();
}
