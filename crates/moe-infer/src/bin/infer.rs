//! MoE inference CLI.
//!
//! Usage:
//!   cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --benchmark
//!   cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --prompt "Hello"

use moe_infer::config::InferConfig;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut config_path = None;
    let mut weights_dir = None;
    let mut prompt = None;
    let mut mode = "benchmark";
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config_path = Some(args[i].clone());
            }
            "--weights" => {
                i += 1;
                weights_dir = Some(args[i].clone());
            }
            "--prompt" => {
                i += 1;
                prompt = Some(args[i].clone());
                mode = "generate";
            }
            "--benchmark" => mode = "benchmark",
            "--help" | "-h" => {
                println!("rustane MoE inference engine");
                println!();
                println!("Usage:");
                println!("  infer --config <path.toml> [--benchmark|--prompt <text>]");
                println!();
                println!("Options:");
                println!("  --config <path>     TOML model config file");
                println!("  --weights <dir>     Converted weights directory");
                println!("  --benchmark         Run decode throughput benchmark (default)");
                println!("  --prompt <text>     Generate text from prompt");
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

    println!("Model: {}", config.model_name());
    println!("  hidden_size: {}", config.hidden_size());
    println!("  layers: {}", config.num_layers());
    println!("  attention: {} ({} Q heads, {} KV heads, head_dim={})",
        config.attention.kind, config.num_q_heads(), config.num_kv_heads(), config.head_dim());
    println!("  rope_theta: {:.0}", config.rope_theta());
    println!("  experts: {} (top-{})", config.num_experts(), config.num_experts_per_tok());
    println!("  dense layer: {} (inter_size={})", config.ffn.dense_layer, config.ffn.dense_inter_size);
    println!("  moe expert FFN dim: {}", config.moe_inter_size());
    println!("  shared experts: {}", config.ffn.shared_expert_count);
    println!("  quantization: {}-bit, group_size={}", config.quantization.bits, config.quantization.group_size);
    println!("  vocab: {}", config.vocab_size());
    println!();

    let _weights_dir = weights_dir.unwrap_or_else(|| "weights/rustane-qwen3".to_string());

    match mode {
        "benchmark" => {
            println!("Benchmark mode — requires converted weights.");
            println!();
            let expert_params = config.moe_inter_size() * config.hidden_size() * 3;
            let expert_bytes_4bit = expert_params / 2;
            let active_bytes = config.num_experts_per_tok() * expert_bytes_4bit;
            let total_expert_bytes = config.num_experts() * expert_bytes_4bit * config.num_layers();
            println!("Estimated memory budget:");
            println!("  Per-expert (4-bit): {:.1} MB", expert_bytes_4bit as f64 / 1e6);
            println!("  Active experts per token: {:.1} MB", active_bytes as f64 / 1e6);
            println!("  Total expert weights: {:.1} GB", total_expert_bytes as f64 / 1e9);
        }
        "generate" => {
            let prompt = prompt.unwrap();
            println!("Generate mode — prompt: \"{prompt}\"");
            println!("Requires converted weights + tokenizer (use --weights <dir>).");
        }
        _ => unreachable!(),
    }
}
