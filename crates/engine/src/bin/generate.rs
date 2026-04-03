//! Minimal checkpoint-backed generation CLI.
//!
//! Usage:
//! cargo run -p engine --bin generate --release -- \
//!   --checkpoint /path/to/ckpt_001000.bin \
//!   --prompt-ids 1,2,3,4 \
//!   --steps 32

use engine::checkpoint::load_checkpoint;
use engine::decode::MAX_GPU_TOPK;
use engine::inference::{
    AUTO_METAL_PARAM_THRESHOLD, DecodeBackendRequest, DecodeMode, GenerationRequest,
    InferenceSession, select_decode_mode,
};
use engine::layer::CompiledKernels;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

struct Args {
    checkpoint: PathBuf,
    prompt_ids: Option<Vec<u32>>,
    prompt: Option<String>,
    tokenizer: Option<PathBuf>,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    decode_backend: DecodeBackendRequest,
    steps: usize,
    temperature: f32,
    top_k: usize,
    softcap: f32,
    samples: usize,
    seed: u64,
    jsonl_session: bool,
}

#[derive(Debug, Deserialize)]
struct JsonlRequest {
    append_prompt_ids: Option<Vec<u32>>,
    append_prompt: Option<String>,
    add_special_tokens: Option<bool>,
    decode_backend: Option<String>,
    steps: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    samples: Option<usize>,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct JsonlSampleResponse {
    seed: u64,
    generated_ids: Vec<u32>,
    full_sequence: Vec<u32>,
    generated_text: Option<String>,
    full_text: Option<String>,
    decode_time_ms: f64,
    decode_tok_per_s: f64,
}

#[derive(Serialize)]
struct JsonlResponse {
    decode_backend_request: &'static str,
    decode_mode: &'static str,
    fallback_reason: Option<&'static str>,
    auto_reason: Option<&'static str>,
    base_prompt_ids: Vec<u32>,
    prompt_ids: Vec<u32>,
    base_prompt_text: Option<String>,
    prompt_text: Option<String>,
    base_prefill_ms: f64,
    base_prefill_tok_per_s: f64,
    request_prefill_ms: f64,
    total_prefill_ms: f64,
    request_decode_ms_total: f64,
    request_decode_ms_avg: f64,
    request_avg_decode_tok_per_s: f64,
    end_to_end_ms_including_prefill: f64,
    end_to_end_tok_per_s_including_prefill: f64,
    results: Vec<JsonlSampleResponse>,
}

#[derive(Serialize)]
struct JsonlError {
    error: String,
}

impl JsonlRequest {
    fn to_generation_request(&self, defaults: &Args) -> Result<GenerationRequest, String> {
        let decode_backend = match self.decode_backend.as_deref() {
            Some(raw) => DecodeBackendRequest::parse(raw).ok_or_else(|| {
                format!("decode_backend must be one of: auto, naive, metal; got {raw}")
            })?,
            None => defaults.decode_backend,
        };
        let samples = self.samples.unwrap_or(defaults.samples);
        if samples == 0 {
            return Err("samples must be >= 1".into());
        }
        Ok(GenerationRequest {
            decode_backend,
            steps: self.steps.unwrap_or(defaults.steps),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            samples,
            seed: self.seed.unwrap_or(defaults.seed),
        })
    }
}

fn resolve_jsonl_prompt_suffix(
    req: &JsonlRequest,
    tokenizer: Option<&Tokenizer>,
    default_add_special_tokens: bool,
) -> Result<Vec<u32>, String> {
    if req.append_prompt_ids.is_some() && req.append_prompt.is_some() {
        return Err("pass either append_prompt_ids or append_prompt, not both".into());
    }

    if let Some(ids) = req.append_prompt_ids.as_ref() {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        return Ok(ids.clone());
    }

    if let Some(prompt) = req.append_prompt.as_ref() {
        let tokenizer = tokenizer.ok_or_else(|| {
            "append_prompt requires the session to be started with --tokenizer".to_string()
        })?;
        let add_special_tokens = req.add_special_tokens.unwrap_or(default_add_special_tokens);
        let ids = tokenizer
            .encode(prompt.as_str(), add_special_tokens)
            .map_err(|err| format!("failed to encode append_prompt: {err}"))?
            .get_ids()
            .to_vec();
        return Ok(ids);
    }

    Ok(Vec::new())
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut checkpoint = None;
    let mut prompt_ids = None;
    let mut prompt = None;
    let mut tokenizer = None;
    let mut add_special_tokens = false;
    let mut skip_special_tokens = true;
    let mut decode_backend = DecodeBackendRequest::Auto;
    let mut steps = 32usize;
    let mut temperature = 0.0f32;
    let mut top_k = 0usize;
    let mut softcap = 0.0f32;
    let mut samples = 1usize;
    let mut seed = 1u64;
    let mut jsonl_session = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--prompt-ids" => {
                prompt_ids = Some(parse_prompt_ids(&args[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--add-special-tokens" => {
                add_special_tokens = true;
                i += 1;
            }
            "--no-skip-special-tokens" => {
                skip_special_tokens = false;
                i += 1;
            }
            "--kv-cache" => {
                decode_backend = DecodeBackendRequest::Metal;
                i += 1;
            }
            "--decode-backend" => {
                decode_backend = DecodeBackendRequest::parse(&args[i + 1]).unwrap_or_else(|| {
                    eprintln!("--decode-backend must be one of: auto, naive, metal");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--steps" => {
                steps = args[i + 1].parse().expect("--steps must be an integer");
                i += 2;
            }
            "--temperature" => {
                temperature = args[i + 1].parse().expect("--temperature must be an f32");
                i += 2;
            }
            "--top-k" => {
                top_k = args[i + 1].parse().expect("--top-k must be an integer");
                i += 2;
            }
            "--softcap" => {
                softcap = args[i + 1].parse().expect("--softcap must be an f32");
                i += 2;
            }
            "--samples" => {
                samples = args[i + 1].parse().expect("--samples must be an integer");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed must be an integer");
                i += 2;
            }
            "--jsonl-session" => {
                jsonl_session = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage_and_exit(0);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                print_usage_and_exit(1);
            }
        }
    }

    if prompt_ids.is_some() && prompt.is_some() {
        eprintln!("pass either --prompt-ids or --prompt, not both");
        print_usage_and_exit(1);
    }
    if prompt.is_some() && tokenizer.is_none() {
        eprintln!("--prompt requires --tokenizer");
        print_usage_and_exit(1);
    }
    if prompt_ids.is_none() && prompt.is_none() {
        eprintln!("pass one of --prompt-ids or --prompt");
        print_usage_and_exit(1);
    }
    if let Some(ref ids) = prompt_ids
        && ids.is_empty()
    {
        eprintln!("--prompt-ids must contain at least one token id");
        std::process::exit(1);
    }
    if samples == 0 {
        eprintln!("--samples must be >= 1");
        std::process::exit(1);
    }

    Args {
        checkpoint: checkpoint.unwrap_or_else(|| {
            eprintln!("--checkpoint required");
            print_usage_and_exit(1);
        }),
        prompt_ids,
        prompt,
        tokenizer,
        add_special_tokens,
        skip_special_tokens,
        decode_backend,
        steps,
        temperature,
        top_k,
        softcap,
        samples,
        seed,
        jsonl_session,
    }
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage: generate --checkpoint PATH (--prompt-ids 1,2,3 | --tokenizer tokenizer.json --prompt \"text\") [--add-special-tokens] [--decode-backend auto|naive|metal] [--kv-cache] [--steps N] [--temperature T] [--top-k K] [--softcap S] [--samples N] [--seed N] [--jsonl-session]"
    );
    std::process::exit(code);
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn parse_prompt_ids(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { Some(s) }
        })
        .map(|s| s.parse().expect("prompt ids must be integers"))
        .collect()
}

fn decode_ids(
    tokenizer: Option<&Tokenizer>,
    ids: &[u32],
    skip_special_tokens: bool,
) -> Option<String> {
    tokenizer.and_then(|tok| tok.decode(ids, skip_special_tokens).ok())
}

fn print_decode_backend_notes(mode: DecodeMode, temperature: f32, top_k: usize) {
    let greedy_sampling = temperature <= 0.0 || top_k == 1;
    if mode != DecodeMode::KvCacheMetal || greedy_sampling {
        return;
    }

    if top_k > 0 {
        if top_k <= MAX_GPU_TOPK {
            println!(
                "cached_sampling_path=gpu_topk_candidates top_k={} gpu_topk_capacity={}",
                top_k, MAX_GPU_TOPK
            );
        } else {
            println!(
                "cached_sampling_path=full_logits_cpu_sample reason=top_k_exceeds_gpu_capacity requested_top_k={} gpu_topk_capacity={}",
                top_k, MAX_GPU_TOPK
            );
        }
    } else {
        println!("cached_sampling_path=gpu_block_stats_exact");
    }
}

fn write_json_line<W: Write, T: Serialize>(out: &mut W, value: &T) -> io::Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()
}

fn build_jsonl_response(
    tokenizer: Option<&Tokenizer>,
    base_prompt_ids: &[u32],
    prompt_ids: &[u32],
    base_prompt_text: Option<&str>,
    prompt_text: Option<&str>,
    skip_special_tokens: bool,
    req: &GenerationRequest,
    batch: &engine::inference::GenerationBatch,
    base_prompt_len: usize,
) -> JsonlResponse {
    let decode_steps = batch
        .results
        .first()
        .map(|result| result.sampled.len().saturating_sub(1))
        .unwrap_or(0);
    let base_prefill_tok_per_s = if batch.base_prefill_elapsed.as_secs_f64() > 0.0 {
        base_prompt_len as f64 / batch.base_prefill_elapsed.as_secs_f64()
    } else {
        0.0
    };
    let request_avg_decode_tok_per_s =
        if decode_steps > 0 && batch.avg_decode_elapsed.as_secs_f64() > 0.0 {
            decode_steps as f64 / batch.avg_decode_elapsed.as_secs_f64()
        } else {
            0.0
        };
    let end_to_end_tok_per_s_including_prefill = if batch.total_elapsed.as_secs_f64() > 0.0 {
        (req.steps * req.samples) as f64 / batch.total_elapsed.as_secs_f64()
    } else {
        0.0
    };
    let results = batch
        .results
        .iter()
        .map(|result| JsonlSampleResponse {
            seed: result.seed,
            generated_ids: result.sampled.clone(),
            full_sequence: result.generated.clone(),
            generated_text: decode_ids(tokenizer, &result.sampled, skip_special_tokens),
            full_text: decode_ids(tokenizer, &result.generated, skip_special_tokens),
            decode_time_ms: result.decode_elapsed.as_secs_f64() * 1000.0,
            decode_tok_per_s: if decode_steps > 0 && result.decode_elapsed.as_secs_f64() > 0.0 {
                decode_steps as f64 / result.decode_elapsed.as_secs_f64()
            } else {
                0.0
            },
        })
        .collect();

    JsonlResponse {
        decode_backend_request: req.decode_backend.as_str(),
        decode_mode: batch.decode_selection.mode.as_str(),
        fallback_reason: batch.decode_selection.fallback_reason,
        auto_reason: batch.decode_selection.auto_reason,
        base_prompt_ids: base_prompt_ids.to_vec(),
        prompt_ids: prompt_ids.to_vec(),
        base_prompt_text: base_prompt_text.map(str::to_owned),
        prompt_text: prompt_text.map(str::to_owned),
        base_prefill_ms: batch.base_prefill_elapsed.as_secs_f64() * 1000.0,
        base_prefill_tok_per_s,
        request_prefill_ms: batch.request_prefill_elapsed.as_secs_f64() * 1000.0,
        total_prefill_ms: batch.prefill_elapsed.as_secs_f64() * 1000.0,
        request_decode_ms_total: batch.total_decode_elapsed.as_secs_f64() * 1000.0,
        request_decode_ms_avg: batch.avg_decode_elapsed.as_secs_f64() * 1000.0,
        request_avg_decode_tok_per_s,
        end_to_end_ms_including_prefill: batch.total_elapsed.as_secs_f64() * 1000.0,
        end_to_end_tok_per_s_including_prefill,
        results,
    }
}

fn run_jsonl_session(
    args: &Args,
    tokenizer: Option<&Tokenizer>,
    prompt_ids: &[u32],
    session: &mut InferenceSession<'_>,
) {
    let base_prompt_text = decode_ids(tokenizer, prompt_ids, args.skip_special_tokens);
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                eprintln!("failed to read stdin: {err}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let parsed = match serde_json::from_str::<JsonlRequest>(&line) {
            Ok(req) => req,
            Err(err) => {
                let _ = write_json_line(
                    &mut out,
                    &JsonlError {
                        error: format!("invalid json request: {err}"),
                    },
                );
                continue;
            }
        };
        let req = match parsed.to_generation_request(args) {
            Ok(req) => req,
            Err(err) => {
                let _ = write_json_line(&mut out, &JsonlError { error: err });
                continue;
            }
        };
        let prompt_suffix =
            match resolve_jsonl_prompt_suffix(&parsed, tokenizer, args.add_special_tokens) {
                Ok(suffix) => suffix,
                Err(err) => {
                    let _ = write_json_line(&mut out, &JsonlError { error: err });
                    continue;
                }
            };
        let mut effective_prompt_ids = prompt_ids.to_vec();
        effective_prompt_ids.extend_from_slice(&prompt_suffix);
        let effective_prompt_text =
            decode_ids(tokenizer, &effective_prompt_ids, args.skip_special_tokens);

        let batch = match session.generate_with_suffix(&req, &prompt_suffix) {
            Ok(batch) => batch,
            Err(err) => {
                let _ = write_json_line(&mut out, &JsonlError { error: err });
                continue;
            }
        };
        let response = build_jsonl_response(
            tokenizer,
            prompt_ids,
            &effective_prompt_ids,
            base_prompt_text.as_deref(),
            effective_prompt_text.as_deref(),
            args.skip_special_tokens,
            &req,
            &batch,
            session.prompt_len(),
        );
        if let Err(err) = write_json_line(&mut out, &response) {
            eprintln!("failed to write stdout: {err}");
            break;
        }
    }
}

fn main() {
    let args = parse_args();
    let ckpt = load_checkpoint(&args.checkpoint).expect("failed to load checkpoint");
    let cfg = ckpt.cfg;
    let weights = ckpt.weights;

    let tokenizer = args
        .tokenizer
        .as_ref()
        .map(|path| Tokenizer::from_file(path).expect("failed to load tokenizer"));
    let prompt_ids = if let Some(prompt) = args.prompt.as_ref() {
        let tokenizer = tokenizer
            .as_ref()
            .expect("tokenizer is required when using --prompt");
        tokenizer
            .encode(prompt.as_str(), args.add_special_tokens)
            .expect("failed to encode prompt")
            .get_ids()
            .to_vec()
    } else {
        args.prompt_ids.clone().expect("prompt ids must be present")
    };
    if prompt_ids.is_empty() {
        fail("prompt encoded to zero tokens");
    }

    let effective_context_len = prompt_ids.len().min(cfg.seq);
    let decode_selection =
        select_decode_mode(&cfg, args.decode_backend, effective_context_len, args.steps);
    let greedy_sampling = args.temperature <= 0.0 || args.top_k == 1;

    if !args.jsonl_session {
        println!(
            "loaded checkpoint: step={} model={}L {}D {}H {}V seq={}",
            ckpt.step, cfg.nlayers, cfg.dim, cfg.hidden, cfg.vocab, cfg.seq
        );
        println!(
            "prompt_len={} effective_context_len={} steps={} temperature={} top_k={} decode_backend_request={} decode_mode={} greedy_sampling={}",
            prompt_ids.len(),
            effective_context_len,
            args.steps,
            args.temperature,
            args.top_k,
            args.decode_backend.as_str(),
            decode_selection.mode.as_str(),
            greedy_sampling
        );
        if prompt_ids.len() > cfg.seq {
            println!(
                "prompt_truncated=true kept_last_tokens={}",
                effective_context_len
            );
        }
        if let Some(reason) = decode_selection.auto_reason {
            println!(
                "decode_backend_auto_reason={} model_params={} auto_metal_threshold={}",
                reason,
                cfg.param_count(),
                AUTO_METAL_PARAM_THRESHOLD
            );
        }
        if let Some(reason) = decode_selection.fallback_reason {
            println!("decode_backend_fallback=true reason={reason}");
        }
        println!(
            "samples={} prefix_reuse={}",
            args.samples,
            if args.samples > 1 {
                "enabled"
            } else {
                "disabled"
            }
        );
        print_decode_backend_notes(decode_selection.mode, args.temperature, args.top_k);
    }

    let compile_start = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    if args.jsonl_session {
        eprintln!(
            "jsonl_session_ready checkpoint_step={} prompt_len={} decode_backend_default={} compile_s={:.2}",
            ckpt.step,
            prompt_ids.len(),
            args.decode_backend.as_str(),
            compile_start.elapsed().as_secs_f32()
        );
    } else {
        println!(
            "compiled kernels in {:.2}s",
            compile_start.elapsed().as_secs_f32()
        );
    }

    let mut session = InferenceSession::new(&cfg, &kernels, &weights, &prompt_ids, args.softcap)
        .unwrap_or_else(|err| fail(&err));
    if args.jsonl_session {
        run_jsonl_session(&args, tokenizer.as_ref(), &prompt_ids, &mut session);
        return;
    }
    let batch = session
        .generate(&GenerationRequest {
            decode_backend: args.decode_backend,
            steps: args.steps,
            temperature: args.temperature,
            top_k: args.top_k,
            samples: args.samples,
            seed: args.seed,
        })
        .unwrap_or_else(|err| fail(&err));

    if let Some(elapsed) = batch.metal_decode_compile_elapsed {
        println!("compiled metal decode in {:.2}s", elapsed.as_secs_f32());
    }

    let prefill_tps = if batch.prefill_elapsed.as_secs_f64() > 0.0 {
        session.prompt_len() as f64 / batch.prefill_elapsed.as_secs_f64()
    } else {
        0.0
    };
    let decode_steps = args.steps.saturating_sub(1);
    let avg_decode_tps = if decode_steps > 0 && batch.avg_decode_elapsed.as_secs_f64() > 0.0 {
        decode_steps as f64 / batch.avg_decode_elapsed.as_secs_f64()
    } else {
        0.0
    };
    let total_generated_tokens = args.steps * args.samples;
    let total_tps = if batch.total_elapsed.as_secs_f64() > 0.0 {
        total_generated_tokens as f64 / batch.total_elapsed.as_secs_f64()
    } else {
        0.0
    };

    if args.samples == 1 {
        let result = &batch.results[0];
        println!("prompt_ids={}", format_ids(&prompt_ids));
        println!("generated_ids={}", format_ids(&result.sampled));
        println!("full_sequence={}", format_ids(&result.generated));
        if let Some(text) = decode_ids(tokenizer.as_ref(), &prompt_ids, args.skip_special_tokens) {
            println!("prompt_text={text}");
        }
        if let Some(text) = decode_ids(
            tokenizer.as_ref(),
            &result.sampled,
            args.skip_special_tokens,
        ) {
            println!("generated_text={text}");
        }
        if let Some(text) = decode_ids(
            tokenizer.as_ref(),
            &result.generated,
            args.skip_special_tokens,
        ) {
            println!("full_text={text}");
        }
        println!(
            "prefill_time={:.2}ms prefill_tok/s={:.2}",
            batch.prefill_elapsed.as_secs_f64() * 1000.0,
            prefill_tps
        );
        println!(
            "decode_steps={} decode_time={:.2}ms decode_tok/s={:.2}",
            decode_steps,
            result.decode_elapsed.as_secs_f64() * 1000.0,
            if decode_steps > 0 && result.decode_elapsed.as_secs_f64() > 0.0 {
                decode_steps as f64 / result.decode_elapsed.as_secs_f64()
            } else {
                0.0
            }
        );
        println!(
            "total_generation_time={:.2}ms overall_tok/s={:.2}",
            batch.total_elapsed.as_secs_f64() * 1000.0,
            total_tps
        );
    } else {
        println!("prompt_ids={}", format_ids(&prompt_ids));
        if let Some(text) = decode_ids(tokenizer.as_ref(), &prompt_ids, args.skip_special_tokens) {
            println!("prompt_text={text}");
        }
        for (sample_idx, result) in batch.results.iter().enumerate() {
            println!("sample{}_seed={}", sample_idx, result.seed);
            println!(
                "sample{}_generated_ids={}",
                sample_idx,
                format_ids(&result.sampled)
            );
            println!(
                "sample{}_full_sequence={}",
                sample_idx,
                format_ids(&result.generated)
            );
            if let Some(text) = decode_ids(
                tokenizer.as_ref(),
                &result.sampled,
                args.skip_special_tokens,
            ) {
                println!("sample{}_generated_text={}", sample_idx, text);
            }
            if let Some(text) = decode_ids(
                tokenizer.as_ref(),
                &result.generated,
                args.skip_special_tokens,
            ) {
                println!("sample{}_full_text={}", sample_idx, text);
            }
            println!(
                "sample{}_decode_steps={} sample{}_decode_time={:.2}ms sample{}_decode_tok/s={:.2}",
                sample_idx,
                decode_steps,
                sample_idx,
                result.decode_elapsed.as_secs_f64() * 1000.0,
                sample_idx,
                if decode_steps > 0 && result.decode_elapsed.as_secs_f64() > 0.0 {
                    decode_steps as f64 / result.decode_elapsed.as_secs_f64()
                } else {
                    0.0
                }
            );
        }
        println!(
            "prefill_time={:.2}ms prefill_tok/s={:.2}",
            batch.prefill_elapsed.as_secs_f64() * 1000.0,
            prefill_tps
        );
        println!(
            "samples_decode_time_total={:.2}ms samples_decode_time_avg={:.2}ms avg_decode_tok/s={:.2}",
            batch.total_decode_elapsed.as_secs_f64() * 1000.0,
            batch.avg_decode_elapsed.as_secs_f64() * 1000.0,
            avg_decode_tps
        );
        println!(
            "total_generation_time={:.2}ms overall_tok/s={:.2}",
            batch.total_elapsed.as_secs_f64() * 1000.0,
            total_tps
        );
    }
}

fn format_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Args {
        Args {
            checkpoint: PathBuf::from("/tmp/ckpt.bin"),
            prompt_ids: Some(vec![1, 2, 3]),
            prompt: None,
            tokenizer: None,
            add_special_tokens: false,
            skip_special_tokens: true,
            decode_backend: DecodeBackendRequest::Auto,
            steps: 32,
            temperature: 0.8,
            top_k: 8,
            softcap: 0.0,
            samples: 2,
            seed: 7,
            jsonl_session: false,
        }
    }

    #[test]
    fn jsonl_request_uses_cli_defaults() {
        let req = JsonlRequest {
            append_prompt_ids: None,
            append_prompt: None,
            add_special_tokens: None,
            decode_backend: None,
            steps: None,
            temperature: None,
            top_k: None,
            samples: None,
            seed: None,
        }
        .to_generation_request(&defaults())
        .expect("request");

        assert_eq!(req.decode_backend, DecodeBackendRequest::Auto);
        assert_eq!(req.steps, 32);
        assert_eq!(req.temperature, 0.8);
        assert_eq!(req.top_k, 8);
        assert_eq!(req.samples, 2);
        assert_eq!(req.seed, 7);
    }

    #[test]
    fn jsonl_request_rejects_invalid_decode_backend() {
        let err = JsonlRequest {
            append_prompt_ids: None,
            append_prompt: None,
            add_special_tokens: None,
            decode_backend: Some("bad".into()),
            steps: None,
            temperature: None,
            top_k: None,
            samples: None,
            seed: None,
        }
        .to_generation_request(&defaults())
        .expect_err("should reject invalid backend");

        assert!(err.contains("decode_backend must be one of"));
    }

    #[test]
    fn jsonl_request_rejects_mixed_append_prompt_inputs() {
        let err = resolve_jsonl_prompt_suffix(
            &JsonlRequest {
                append_prompt_ids: Some(vec![1, 2]),
                append_prompt: Some("hello".into()),
                add_special_tokens: None,
                decode_backend: None,
                steps: None,
                temperature: None,
                top_k: None,
                samples: None,
                seed: None,
            },
            None,
            false,
        )
        .expect_err("should reject mixed prompt suffix inputs");

        assert!(err.contains("either append_prompt_ids or append_prompt"));
    }
}
