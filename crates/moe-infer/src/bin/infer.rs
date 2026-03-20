//! MoE inference CLI.
//!
//! Usage:
//!   cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --benchmark
//!   cargo run -p moe-infer --release --bin infer -- --config configs/target-1t.toml --interactive

use moe_infer::config::InferConfig;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut config_path = None;
    let mut mode = "benchmark";
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config_path = Some(args[i].clone());
            }
            "--benchmark" => mode = "benchmark",
            "--interactive" => mode = "interactive",
            "--help" | "-h" => {
                println!("rustane MoE inference engine");
                println!();
                println!("Usage:");
                println!("  infer --config <path.toml> [--benchmark|--interactive]");
                println!();
                println!("Options:");
                println!("  --config <path>   TOML model config file");
                println!("  --benchmark       Run decode throughput benchmark (default)");
                println!("  --interactive     Interactive text generation");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let config_path = config_path.unwrap_or_else(|| {
        eprintln!("Error: --config <path.toml> is required");
        std::process::exit(1);
    });

    let config = InferConfig::from_toml(Path::new(&config_path)).unwrap_or_else(|e| {
        eprintln!("Error loading config: {e}");
        std::process::exit(1);
    });

    println!("Model: {}", config.model_name);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  layers: {}", config.num_hidden_layers);
    println!("  experts: {} (top-{})", config.num_experts, config.num_experts_per_tok);
    println!("  expert FFN dim: {}", config.moe_intermediate_size);
    println!("  quantization: {}-bit, group_size={}", config.bits, config.group_size);
    println!("  vocab: {}", config.vocab_size);
    if let Some(kv_rank) = config.kv_lora_rank {
        println!("  MLA kv_lora_rank: {kv_rank}");
    }
    println!();

    match mode {
        "benchmark" => {
            println!("Benchmark mode — requires converted weights (not yet available).");
            println!("Run `cargo run -p expert-pager --bin convert` first to convert weights.");
            println!();
            println!("Estimated memory budget:");
            let expert_params = config.moe_intermediate_size * config.hidden_size * 3; // gate+up+down
            let expert_bytes_4bit = expert_params / 2; // 4-bit = 0.5 bytes per param
            let active_bytes = config.num_experts_per_tok * expert_bytes_4bit;
            let total_expert_bytes = config.num_experts * expert_bytes_4bit * config.num_hidden_layers;
            println!("  Per-expert (4-bit): {:.1} MB", expert_bytes_4bit as f64 / 1e6);
            println!("  Active experts per token: {:.1} MB", active_bytes as f64 / 1e6);
            println!("  Total expert weights: {:.1} GB", total_expert_bytes as f64 / 1e9);
        }
        "interactive" => {
            println!("Interactive mode — requires converted weights + tokenizer (not yet available).");
        }
        _ => unreachable!(),
    }
}
