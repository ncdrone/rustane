//! MoE inference CLI.
//!
//! Usage:
//!   cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --benchmark
//!   cargo run -p moe-infer --release --bin infer -- --config configs/qwen3-moe-30b.toml --prompt "Hello"

use moe_infer::config::InferConfig;
use moe_infer::generate::{Model, SamplingConfig};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut config_path = None;
    let mut weights_dir = None;
    let mut tokenizer_path = None;
    let mut prompt = None;
    let mut max_tokens: usize = 100;
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
            "--tokenizer" => {
                i += 1;
                tokenizer_path = Some(args[i].clone());
            }
            "--prompt" => {
                i += 1;
                prompt = Some(args[i].clone());
                mode = "generate";
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("--max-tokens must be a number");
            }
            "--benchmark" => mode = "benchmark",
            "--help" | "-h" => {
                println!("rustane MoE inference engine");
                println!();
                println!("Usage:");
                println!("  infer --config <path.toml> [--benchmark|--prompt <text>]");
                println!();
                println!("Options:");
                println!("  --config <path>       TOML model config file");
                println!("  --weights <dir>       Converted weights directory (default: weights/rustane-qwen3)");
                println!("  --tokenizer <path>    tokenizer.json path (default: weights/qwen3-30b-a3b/tokenizer.json)");
                println!("  --prompt <text>       Generate text from prompt");
                println!("  --max-tokens <n>      Max tokens to generate (default: 100)");
                println!("  --benchmark           Show model config and memory estimates (default)");
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
    println!("  all layers MoE: {}", config.ffn.all_moe);
    println!("  moe expert FFN dim: {}", config.moe_inter_size());
    if config.ffn.shared_expert_count > 0 {
        println!("  shared experts: {}", config.ffn.shared_expert_count);
    }
    println!("  quantization: {}-bit, group_size={}", config.quantization.bits, config.quantization.group_size);
    println!("  vocab: {}", config.vocab_size());
    println!();

    let weights_dir = weights_dir.unwrap_or_else(|| "weights/rustane-qwen3".to_string());
    let tokenizer_path = tokenizer_path
        .unwrap_or_else(|| "weights/qwen3-30b-a3b/tokenizer.json".to_string());

    match mode {
        "benchmark" => {
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
            let prompt_text = prompt.unwrap();
            println!("Loading model from {weights_dir}...");

            let model = Model::load(
                Path::new(&weights_dir),
                Path::new(&config_path),
            )
            .unwrap_or_else(|e| {
                eprintln!("Error loading model: {e}");
                std::process::exit(1);
            });

            println!("Loading tokenizer from {tokenizer_path}...");
            let tok = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .unwrap_or_else(|e| {
                    eprintln!("Error loading tokenizer: {e}");
                    std::process::exit(1);
                });

            println!("Generating (max {max_tokens} tokens)...\n");

            let t0 = std::time::Instant::now();
            let sampling = SamplingConfig::greedy();
            let output = moe_infer::generate::generate(
                &model, &tok, &prompt_text, max_tokens, &sampling,
            )
            .unwrap_or_else(|e| {
                eprintln!("Generation error: {e}");
                std::process::exit(1);
            });
            let elapsed = t0.elapsed();

            println!("{}{}", prompt_text, output.text);
            println!("\n---");
            let decode_tok_per_sec = if output.decode_secs > 0.0 {
                (output.tokens_generated.saturating_sub(1)) as f64 / output.decode_secs
            } else { 0.0 };
            let total_tok_per_sec = output.tokens_generated as f64 / elapsed.as_secs_f64();
            println!(
                "{} tokens in {:.1}s = {:.1} tok/s (decode: {:.1} tok/s)",
                output.tokens_generated,
                elapsed.as_secs_f64(),
                total_tok_per_sec,
                decode_tok_per_sec,
            );
            println!(
                "  Prefill: {} tokens in {:.1}s ({:.0} tok/s)",
                output.prompt_tokens,
                output.prefill_secs,
                output.prompt_tokens as f64 / output.prefill_secs.max(0.001),
            );
            println!(
                "  Decode: {} tokens in {:.1}s ({:.1} tok/s)",
                output.tokens_generated.saturating_sub(1),
                output.decode_secs,
                decode_tok_per_sec,
            );
            println!("  Backend: Metal GPU + Accelerate BLAS");
        }
        _ => unreachable!(),
    }
}
