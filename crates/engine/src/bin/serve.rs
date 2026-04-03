//! Minimal HTTP inference server backed by reusable prefix-cached sessions.

use engine::checkpoint::load_checkpoint;
use engine::inference::{
    DecodeBackendRequest, GenerationBatch, GenerationRequest, InferenceSession,
};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

struct Args {
    checkpoint: PathBuf,
    tokenizer: PathBuf,
    bind: String,
    model_name: String,
    decode_backend: DecodeBackendRequest,
    softcap: f32,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    max_cache_sessions: usize,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Option<String>,
    prompt_ids: Option<Vec<u32>>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    n: Option<usize>,
    stream: Option<bool>,
    decode_backend: Option<String>,
    add_special_tokens: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    n: Option<usize>,
    stream: Option<bool>,
    decode_backend: Option<String>,
    add_special_tokens: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
    model: String,
    vocab: usize,
    seq: usize,
    cache_entries: usize,
    max_cache_sessions: usize,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
    rustane: RustaneMeta,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: &'static str,
    generated_ids: Vec<u32>,
    full_sequence: Vec<u32>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
    rustane: RustaneMeta,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: AssistantMessage,
    finish_reason: &'static str,
    generated_ids: Vec<u32>,
    full_sequence: Vec<u32>,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Clone, Serialize)]
struct RustaneMeta {
    effective_prompt_tokens: usize,
    prompt_truncated: bool,
    cache_prefix_tokens: usize,
    cache_hit: bool,
    base_prefill_ms: f64,
    request_prefill_ms: f64,
    total_prefill_ms: f64,
    decode_ms_total: f64,
    decode_ms_avg: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct CompletionStreamResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rustane: Option<RustaneMeta>,
}

#[derive(Serialize)]
struct CompletionStreamChoice {
    index: usize,
    text: String,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionChunkResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rustane: Option<RustaneMeta>,
}

#[derive(Serialize)]
struct ChatCompletionChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Default, Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct ServerState<'a> {
    cfg: &'a ModelConfig,
    tokenizer: Tokenizer,
    model_name: String,
    default_decode_backend: DecodeBackendRequest,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    cache: SessionCache<'a>,
}

struct SessionCache<'a> {
    cfg: &'a ModelConfig,
    kernels: &'a CompiledKernels,
    weights: &'a engine::full_model::ModelWeights,
    softcap: f32,
    max_entries: usize,
    tick: u64,
    entries: Vec<CachedSession<'a>>,
}

struct CachedSession<'a> {
    window_prompt_ids: Vec<u32>,
    session: InferenceSession<'a>,
    last_used_tick: u64,
}

struct CacheGeneration {
    batch: GenerationBatch,
    effective_prompt_tokens: usize,
    prompt_truncated: bool,
    cache_prefix_tokens: usize,
}

enum PromptInput {
    Text(String),
    TokenIds(Vec<u32>),
}

impl SessionCache<'static> {
    fn new(
        cfg: &'static ModelConfig,
        kernels: &'static CompiledKernels,
        weights: &'static engine::full_model::ModelWeights,
        softcap: f32,
        max_entries: usize,
    ) -> Self {
        Self {
            cfg,
            kernels,
            weights,
            softcap,
            max_entries: max_entries.max(1),
            tick: 0,
            entries: Vec::new(),
        }
    }
}

impl<'a> SessionCache<'a> {
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        req: &GenerationRequest,
    ) -> Result<CacheGeneration, String> {
        let window_prompt_ids = self.normalize_prompt_ids(prompt_ids);
        let prompt_truncated = window_prompt_ids.len() != prompt_ids.len();
        let effective_prompt_tokens = window_prompt_ids.len();

        if let Some(idx) = self.find_exact(&window_prompt_ids) {
            let tick = self.bump_tick();
            let batch = {
                let entry = &mut self.entries[idx];
                entry.last_used_tick = tick;
                entry.session.generate(req)?
            };
            return Ok(CacheGeneration {
                batch,
                effective_prompt_tokens,
                prompt_truncated,
                cache_prefix_tokens: effective_prompt_tokens,
            });
        }

        if let Some(idx) = self.find_longest_prefix(&window_prompt_ids) {
            let prefix_len = self.entries[idx].window_prompt_ids.len();
            let suffix = &window_prompt_ids[prefix_len..];
            let tick = self.bump_tick();
            let batch = {
                let entry = &mut self.entries[idx];
                entry.last_used_tick = tick;
                entry.session.generate_with_suffix(req, suffix)?
            };
            self.maybe_insert(&window_prompt_ids)?;
            return Ok(CacheGeneration {
                batch,
                effective_prompt_tokens,
                prompt_truncated,
                cache_prefix_tokens: prefix_len,
            });
        }

        let idx = self.insert(&window_prompt_ids)?;
        let tick = self.bump_tick();
        let batch = {
            let entry = &mut self.entries[idx];
            entry.last_used_tick = tick;
            entry.session.generate(req)?
        };
        Ok(CacheGeneration {
            batch,
            effective_prompt_tokens,
            prompt_truncated,
            cache_prefix_tokens: 0,
        })
    }

    fn stream_generate<F>(
        &mut self,
        prompt_ids: &[u32],
        req: &GenerationRequest,
        mut on_token: F,
    ) -> Result<CacheGeneration, String>
    where
        F: FnMut(u32, &[u32], &[u32]) -> Result<(), String>,
    {
        let window_prompt_ids = self.normalize_prompt_ids(prompt_ids);
        let prompt_truncated = window_prompt_ids.len() != prompt_ids.len();
        let effective_prompt_tokens = window_prompt_ids.len();

        if let Some(idx) = self.find_exact(&window_prompt_ids) {
            let tick = self.bump_tick();
            let batch = {
                let entry = &mut self.entries[idx];
                entry.last_used_tick = tick;
                entry
                    .session
                    .generate_stream_with_suffix(req, &[], &mut on_token)?
            };
            return Ok(CacheGeneration {
                batch,
                effective_prompt_tokens,
                prompt_truncated,
                cache_prefix_tokens: effective_prompt_tokens,
            });
        }

        if let Some(idx) = self.find_longest_prefix(&window_prompt_ids) {
            let prefix_len = self.entries[idx].window_prompt_ids.len();
            let suffix = &window_prompt_ids[prefix_len..];
            let tick = self.bump_tick();
            let batch = {
                let entry = &mut self.entries[idx];
                entry.last_used_tick = tick;
                entry
                    .session
                    .generate_stream_with_suffix(req, suffix, &mut on_token)?
            };
            self.maybe_insert(&window_prompt_ids)?;
            return Ok(CacheGeneration {
                batch,
                effective_prompt_tokens,
                prompt_truncated,
                cache_prefix_tokens: prefix_len,
            });
        }

        let idx = self.insert(&window_prompt_ids)?;
        let tick = self.bump_tick();
        let batch = {
            let entry = &mut self.entries[idx];
            entry.last_used_tick = tick;
            entry
                .session
                .generate_stream_with_suffix(req, &[], &mut on_token)?
        };
        Ok(CacheGeneration {
            batch,
            effective_prompt_tokens,
            prompt_truncated,
            cache_prefix_tokens: 0,
        })
    }

    fn maybe_insert(&mut self, prompt_ids: &[u32]) -> Result<(), String> {
        if self.find_exact(prompt_ids).is_none() {
            let _ = self.insert(prompt_ids)?;
        }
        Ok(())
    }

    fn insert(&mut self, prompt_ids: &[u32]) -> Result<usize, String> {
        let session = InferenceSession::new(
            self.cfg,
            self.kernels,
            self.weights,
            prompt_ids,
            self.softcap,
        )?;
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        let tick = self.bump_tick();
        let idx = self.entries.len();
        self.entries.push(CachedSession {
            window_prompt_ids: prompt_ids.to_vec(),
            session,
            last_used_tick: tick,
        });
        Ok(idx)
    }

    fn evict_lru(&mut self) {
        if let Some((idx, _)) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.last_used_tick)
        {
            self.entries.swap_remove(idx);
        }
    }

    fn find_exact(&self, prompt_ids: &[u32]) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.window_prompt_ids == prompt_ids)
    }

    fn find_longest_prefix(&self, prompt_ids: &[u32]) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| prompt_ids.starts_with(&entry.window_prompt_ids))
            .max_by_key(|(_, entry)| entry.window_prompt_ids.len())
            .map(|(idx, _)| idx)
    }

    fn normalize_prompt_ids(&self, prompt_ids: &[u32]) -> Vec<u32> {
        if prompt_ids.len() <= self.cfg.seq {
            return prompt_ids.to_vec();
        }
        prompt_ids[prompt_ids.len() - self.cfg.seq..].to_vec()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn max_entries(&self) -> usize {
        self.max_entries
    }

    fn bump_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut checkpoint = None;
    let mut tokenizer = None;
    let mut bind = "127.0.0.1:8080".to_string();
    let mut model_name = "rustane".to_string();
    let mut decode_backend = DecodeBackendRequest::Auto;
    let mut softcap = 0.0f32;
    let mut add_special_tokens = false;
    let mut skip_special_tokens = true;
    let mut max_cache_sessions = 32usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--bind" => {
                bind = args[i + 1].clone();
                i += 2;
            }
            "--model-name" => {
                model_name = args[i + 1].clone();
                i += 2;
            }
            "--decode-backend" => {
                decode_backend = DecodeBackendRequest::parse(&args[i + 1]).unwrap_or_else(|| {
                    eprintln!("--decode-backend must be one of: auto, naive, metal");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--softcap" => {
                softcap = args[i + 1].parse().expect("--softcap must be an f32");
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
            "--max-cache-sessions" => {
                max_cache_sessions = args[i + 1]
                    .parse()
                    .expect("--max-cache-sessions must be an integer");
                i += 2;
            }
            "--help" | "-h" => print_usage_and_exit(0),
            other => {
                eprintln!("Unknown arg: {other}");
                print_usage_and_exit(1);
            }
        }
    }

    Args {
        checkpoint: checkpoint.unwrap_or_else(|| {
            eprintln!("--checkpoint required");
            print_usage_and_exit(1);
        }),
        tokenizer: tokenizer.unwrap_or_else(|| {
            eprintln!("--tokenizer required");
            print_usage_and_exit(1);
        }),
        bind,
        model_name,
        decode_backend,
        softcap,
        add_special_tokens,
        skip_special_tokens,
        max_cache_sessions,
    }
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage: serve --checkpoint PATH --tokenizer PATH [--bind HOST:PORT] [--model-name NAME] [--decode-backend auto|naive|metal] [--softcap S] [--add-special-tokens] [--no-skip-special-tokens] [--max-cache-sessions N]"
    );
    std::process::exit(code);
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

fn next_id(prefix: &str, seed: u64) -> String {
    format!("{prefix}-{:x}", seed ^ now_unix())
}

fn decode_backend_for(
    raw: Option<&str>,
    default: DecodeBackendRequest,
) -> Result<DecodeBackendRequest, String> {
    match raw {
        Some(raw) => DecodeBackendRequest::parse(raw)
            .ok_or_else(|| format!("decode_backend must be one of: auto, naive, metal; got {raw}")),
        None => Ok(default),
    }
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    input: PromptInput,
    add_special_tokens: bool,
    vocab: usize,
) -> Result<Vec<u32>, String> {
    let ids = match input {
        PromptInput::Text(text) => tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| format!("failed to encode prompt: {err}"))?
            .get_ids()
            .to_vec(),
        PromptInput::TokenIds(ids) => ids,
    };
    if ids.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    if let Some((pos, tok)) = ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, tok)| *tok as usize >= vocab)
    {
        return Err(format!(
            "prompt token id {} at position {} exceeds model vocab {}",
            tok, pos, vocab
        ));
    }
    Ok(ids)
}

fn render_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for (idx, message) in messages.iter().enumerate() {
        if idx > 0 {
            out.push_str("\n\n");
        }
        if message.role.eq_ignore_ascii_case("system") {
            out.push_str(message.content.trim());
        } else if message.role.eq_ignore_ascii_case("user") {
            out.push_str(message.content.trim());
        } else if message.role.eq_ignore_ascii_case("assistant") {
            out.push_str(message.content.trim());
        } else {
            out.push_str(message.content.trim());
        }
    }
    out
}

fn build_generation_request(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    n: Option<usize>,
    decode_backend: Option<&str>,
    default_decode_backend: DecodeBackendRequest,
) -> Result<GenerationRequest, String> {
    let samples = n.unwrap_or(1);
    if samples == 0 {
        return Err("n must be >= 1".into());
    }
    Ok(GenerationRequest {
        decode_backend: decode_backend_for(decode_backend, default_decode_backend)?,
        steps: max_tokens.unwrap_or(32),
        temperature: temperature.unwrap_or(0.0),
        top_k: top_k.unwrap_or(0),
        samples,
        seed: seed.unwrap_or(1),
    })
}

fn prepare_completion_request(
    state: &ServerState<'_>,
    req: CompletionRequest,
) -> Result<(Vec<u32>, GenerationRequest), String> {
    if let Some(model) = req.model.as_deref()
        && model != state.model_name
    {
        return Err(format!(
            "requested model {model} does not match loaded model {}",
            state.model_name
        ));
    }
    let prompt_input = match (req.prompt, req.prompt_ids) {
        (Some(_), Some(_)) => return Err("pass either prompt or prompt_ids, not both".into()),
        (Some(prompt), None) => PromptInput::Text(prompt),
        (None, Some(ids)) => PromptInput::TokenIds(ids),
        (None, None) => return Err("one of prompt or prompt_ids is required".into()),
    };
    let prompt_ids = tokenize_prompt(
        &state.tokenizer,
        prompt_input,
        req.add_special_tokens.unwrap_or(state.add_special_tokens),
        state.cfg.vocab,
    )?;
    let gen_req = build_generation_request(
        req.max_tokens,
        req.temperature,
        req.top_k,
        req.seed,
        req.n,
        req.decode_backend.as_deref(),
        state.default_decode_backend,
    )?;
    Ok((prompt_ids, gen_req))
}

fn prepare_chat_completion_request(
    state: &ServerState<'_>,
    req: ChatCompletionRequest,
) -> Result<(Vec<u32>, GenerationRequest), String> {
    if let Some(model) = req.model.as_deref()
        && model != state.model_name
    {
        return Err(format!(
            "requested model {model} does not match loaded model {}",
            state.model_name
        ));
    }
    if req.messages.is_empty() {
        return Err("messages must contain at least one item".into());
    }
    let prompt_ids = tokenize_prompt(
        &state.tokenizer,
        PromptInput::Text(render_chat_prompt(&req.messages)),
        req.add_special_tokens.unwrap_or(state.add_special_tokens),
        state.cfg.vocab,
    )?;
    let gen_req = build_generation_request(
        req.max_tokens,
        req.temperature,
        req.top_k,
        req.seed,
        req.n,
        req.decode_backend.as_deref(),
        state.default_decode_backend,
    )?;
    Ok((prompt_ids, gen_req))
}

fn usage_for(prompt_tokens: usize, batch: &GenerationBatch) -> Usage {
    let completion_tokens = batch
        .results
        .first()
        .map(|sample| sample.sampled.len())
        .unwrap_or(0);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

fn rustane_meta(
    batch: &GenerationBatch,
    effective_prompt_tokens: usize,
    prompt_truncated: bool,
    cache_prefix_tokens: usize,
) -> RustaneMeta {
    RustaneMeta {
        effective_prompt_tokens,
        prompt_truncated,
        cache_prefix_tokens,
        cache_hit: cache_prefix_tokens > 0,
        base_prefill_ms: batch.base_prefill_elapsed.as_secs_f64() * 1000.0,
        request_prefill_ms: batch.request_prefill_elapsed.as_secs_f64() * 1000.0,
        total_prefill_ms: batch.prefill_elapsed.as_secs_f64() * 1000.0,
        decode_ms_total: batch.total_decode_elapsed.as_secs_f64() * 1000.0,
        decode_ms_avg: batch.avg_decode_elapsed.as_secs_f64() * 1000.0,
        total_ms: batch.total_elapsed.as_secs_f64() * 1000.0,
    }
}

fn decode_sample_text(
    tokenizer: &Tokenizer,
    ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String, String> {
    tokenizer
        .decode(ids, skip_special_tokens)
        .map_err(|err| format!("failed to decode output ids: {err}"))
}

fn completion_response(
    state: &ServerState<'_>,
    prompt_ids: &[u32],
    outcome: CacheGeneration,
) -> Result<CompletionResponse, String> {
    let usage = usage_for(prompt_ids.len(), &outcome.batch);
    let choices = outcome
        .batch
        .results
        .iter()
        .enumerate()
        .map(|(idx, result)| {
            let text =
                decode_sample_text(&state.tokenizer, &result.sampled, state.skip_special_tokens)?;
            Ok(CompletionChoice {
                index: idx,
                text,
                finish_reason: "length",
                generated_ids: result.sampled.clone(),
                full_sequence: result.generated.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(CompletionResponse {
        id: next_id("cmpl", outcome.batch.total_elapsed.as_nanos() as u64),
        object: "text_completion",
        created: now_unix(),
        model: state.model_name.clone(),
        choices,
        usage,
        rustane: rustane_meta(
            &outcome.batch,
            outcome.effective_prompt_tokens,
            outcome.prompt_truncated,
            outcome.cache_prefix_tokens,
        ),
    })
}

fn chat_response(
    state: &ServerState<'_>,
    prompt_ids: &[u32],
    outcome: CacheGeneration,
) -> Result<ChatCompletionResponse, String> {
    let usage = usage_for(prompt_ids.len(), &outcome.batch);
    let choices = outcome
        .batch
        .results
        .iter()
        .enumerate()
        .map(|(idx, result)| {
            let text =
                decode_sample_text(&state.tokenizer, &result.sampled, state.skip_special_tokens)?;
            Ok(ChatChoice {
                index: idx,
                message: AssistantMessage {
                    role: "assistant",
                    content: text,
                },
                finish_reason: "length",
                generated_ids: result.sampled.clone(),
                full_sequence: result.generated.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(ChatCompletionResponse {
        id: next_id("chatcmpl", outcome.batch.total_elapsed.as_nanos() as u64),
        object: "chat.completion",
        created: now_unix(),
        model: state.model_name.clone(),
        choices,
        usage,
        rustane: rustane_meta(
            &outcome.batch,
            outcome.effective_prompt_tokens,
            outcome.prompt_truncated,
            outcome.cache_prefix_tokens,
        ),
    })
}

fn handle_completion(
    state: &mut ServerState<'_>,
    req: CompletionRequest,
) -> Result<CompletionResponse, String> {
    if req.stream == Some(true) {
        return Err("stream request must use SSE path".into());
    }
    let (prompt_ids, gen_req) = prepare_completion_request(state, req)?;
    let outcome = state.cache.generate(&prompt_ids, &gen_req)?;
    completion_response(state, &prompt_ids, outcome)
}

fn handle_chat_completion(
    state: &mut ServerState<'_>,
    req: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, String> {
    if req.stream == Some(true) {
        return Err("stream request must use SSE path".into());
    }
    let (prompt_ids, gen_req) = prepare_chat_completion_request(state, req)?;
    let outcome = state.cache.generate(&prompt_ids, &gen_req)?;
    chat_response(state, &prompt_ids, outcome)
}

fn parse_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .map_err(|err| format!("failed to read request line: {err}"))?;
    if request_line.trim().is_empty() {
        return Err("empty request".into());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "missing HTTP method".to_string())?;
    let path = parts
        .next()
        .ok_or_else(|| "missing HTTP path".to_string())?;

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|err| format!("failed to read header line: {err}"))?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let content_length = headers
        .get("content-length")
        .map(|raw| {
            raw.parse::<usize>()
                .map_err(|_| "invalid content-length".to_string())
        })
        .transpose()?
        .unwrap_or(0);
    let mut body = vec![0u8; content_length];
    reader
        .read_exact(&mut body)
        .map_err(|err| format!("failed to read request body: {err}"))?;

    Ok(HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        body,
    })
}

fn write_json_response<T: Serialize>(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    body: &T,
) -> io::Result<()> {
    let payload = serde_json::to_vec(body).expect("json serialize");
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_code,
        status_text,
        payload.len()
    )?;
    stream.write_all(&payload)?;
    stream.flush()
}

fn write_chunk(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    write!(stream, "{:X}\r\n", payload.len())?;
    stream.write_all(payload)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn write_sse_headers(stream: &mut TcpStream) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n"
    )?;
    stream.flush()
}

fn write_sse_json<T: Serialize>(stream: &mut TcpStream, body: &T) -> io::Result<()> {
    let payload = format!(
        "data: {}\n\n",
        serde_json::to_string(body).expect("json serialize")
    );
    write_chunk(stream, payload.as_bytes())
}

fn write_sse_done(stream: &mut TcpStream) -> io::Result<()> {
    write_chunk(stream, b"data: [DONE]\n\n")?;
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn write_sse_error(stream: &mut TcpStream, message: impl Into<String>) -> io::Result<()> {
    let body = ErrorEnvelope {
        error: ErrorBody {
            message: message.into(),
            r#type: "invalid_request_error",
        },
    };
    write_sse_json(stream, &body)?;
    write_sse_done(stream)
}

fn stream_completion_response(
    state: &mut ServerState<'_>,
    req: CompletionRequest,
    stream: &mut TcpStream,
) -> io::Result<()> {
    let (prompt_ids, gen_req) = match prepare_completion_request(state, req) {
        Ok(parts) => parts,
        Err(err) => return write_error(stream, 400, err),
    };
    if gen_req.samples != 1 {
        return write_error(stream, 400, "stream=true only supports n=1");
    }

    write_sse_headers(stream)?;
    let created = now_unix();
    let id = next_id("cmpl", gen_req.seed ^ prompt_ids.len() as u64);
    let model = state.model_name.clone();
    let mut previous_text = String::new();

    let outcome = state
        .cache
        .stream_generate(&prompt_ids, &gen_req, |_, sampled, _| {
            let current = decode_sample_text(&state.tokenizer, sampled, state.skip_special_tokens)?;
            let delta = current
                .strip_prefix(&previous_text)
                .map(str::to_owned)
                .unwrap_or_else(|| current.clone());
            previous_text = current;
            if delta.is_empty() {
                return Ok(());
            }
            write_sse_json(
                stream,
                &CompletionStreamResponse {
                    id: id.clone(),
                    object: "text_completion",
                    created,
                    model: model.clone(),
                    choices: vec![CompletionStreamChoice {
                        index: 0,
                        text: delta,
                        finish_reason: None,
                    }],
                    rustane: None,
                },
            )
            .map_err(|err| err.to_string())
        });

    match outcome {
        Ok(outcome) => {
            write_sse_json(
                stream,
                &CompletionStreamResponse {
                    id,
                    object: "text_completion",
                    created,
                    model,
                    choices: vec![CompletionStreamChoice {
                        index: 0,
                        text: String::new(),
                        finish_reason: Some("length"),
                    }],
                    rustane: Some(rustane_meta(
                        &outcome.batch,
                        outcome.effective_prompt_tokens,
                        outcome.prompt_truncated,
                        outcome.cache_prefix_tokens,
                    )),
                },
            )?;
            write_sse_done(stream)
        }
        Err(err) => write_sse_error(stream, err),
    }
}

fn stream_chat_completion_response(
    state: &mut ServerState<'_>,
    req: ChatCompletionRequest,
    stream: &mut TcpStream,
) -> io::Result<()> {
    let (prompt_ids, gen_req) = match prepare_chat_completion_request(state, req) {
        Ok(parts) => parts,
        Err(err) => return write_error(stream, 400, err),
    };
    if gen_req.samples != 1 {
        return write_error(stream, 400, "stream=true only supports n=1");
    }

    write_sse_headers(stream)?;
    let created = now_unix();
    let id = next_id("chatcmpl", gen_req.seed ^ prompt_ids.len() as u64);
    let model = state.model_name.clone();
    write_sse_json(
        stream,
        &ChatCompletionChunkResponse {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
            rustane: None,
        },
    )?;

    let mut previous_text = String::new();
    let outcome = state
        .cache
        .stream_generate(&prompt_ids, &gen_req, |_, sampled, _| {
            let current = decode_sample_text(&state.tokenizer, sampled, state.skip_special_tokens)?;
            let delta = current
                .strip_prefix(&previous_text)
                .map(str::to_owned)
                .unwrap_or_else(|| current.clone());
            previous_text = current;
            if delta.is_empty() {
                return Ok(());
            }
            write_sse_json(
                stream,
                &ChatCompletionChunkResponse {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(delta),
                        },
                        finish_reason: None,
                    }],
                    rustane: None,
                },
            )
            .map_err(|err| err.to_string())
        });

    match outcome {
        Ok(outcome) => {
            write_sse_json(
                stream,
                &ChatCompletionChunkResponse {
                    id,
                    object: "chat.completion.chunk",
                    created,
                    model,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatDelta::default(),
                        finish_reason: Some("length"),
                    }],
                    rustane: Some(rustane_meta(
                        &outcome.batch,
                        outcome.effective_prompt_tokens,
                        outcome.prompt_truncated,
                        outcome.cache_prefix_tokens,
                    )),
                },
            )?;
            write_sse_done(stream)
        }
        Err(err) => write_sse_error(stream, err),
    }
}

fn write_error(
    stream: &mut TcpStream,
    status_code: u16,
    message: impl Into<String>,
) -> io::Result<()> {
    let body = ErrorEnvelope {
        error: ErrorBody {
            message: message.into(),
            r#type: "invalid_request_error",
        },
    };
    let status_text = match status_code {
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Error",
    };
    write_json_response(stream, status_code, status_text, &body)
}

fn route_request(
    state: &mut ServerState<'_>,
    req: HttpRequest,
    stream: &mut TcpStream,
) -> io::Result<()> {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/healthz") => write_json_response(
            stream,
            200,
            "OK",
            &HealthResponse {
                ok: true,
                model: state.model_name.clone(),
                vocab: state.cfg.vocab,
                seq: state.cfg.seq,
                cache_entries: state.cache.len(),
                max_cache_sessions: state.cache.max_entries(),
            },
        ),
        ("GET", "/v1/models") => write_json_response(
            stream,
            200,
            "OK",
            &ModelsResponse {
                object: "list",
                data: vec![ModelCard {
                    id: state.model_name.clone(),
                    object: "model",
                    owned_by: "rustane",
                }],
            },
        ),
        ("POST", "/v1/completions") => {
            let request: CompletionRequest = serde_json::from_slice(&req.body).map_err(|err| {
                io::Error::new(io::ErrorKind::InvalidInput, format!("invalid JSON: {err}"))
            })?;
            if request.stream == Some(true) {
                stream_completion_response(state, request, stream)
            } else {
                match handle_completion(state, request) {
                    Ok(response) => write_json_response(stream, 200, "OK", &response),
                    Err(err) => write_error(stream, 400, err),
                }
            }
        }
        ("POST", "/v1/chat/completions") => {
            let request: ChatCompletionRequest =
                serde_json::from_slice(&req.body).map_err(|err| {
                    io::Error::new(io::ErrorKind::InvalidInput, format!("invalid JSON: {err}"))
                })?;
            if request.stream == Some(true) {
                stream_chat_completion_response(state, request, stream)
            } else {
                match handle_chat_completion(state, request) {
                    Ok(response) => write_json_response(stream, 200, "OK", &response),
                    Err(err) => write_error(stream, 400, err),
                }
            }
        }
        ("POST", _) | ("GET", _) => write_error(stream, 404, "unknown route"),
        _ => write_error(stream, 405, "unsupported HTTP method"),
    }
}

fn main() {
    let args = parse_args();
    let ckpt = load_checkpoint(&args.checkpoint).expect("failed to load checkpoint");
    let cfg = Box::leak(Box::new(ckpt.cfg));
    let weights = Box::leak(Box::new(ckpt.weights));
    let tokenizer = Tokenizer::from_file(&args.tokenizer).expect("failed to load tokenizer");
    let kernels = Box::leak(Box::new(CompiledKernels::compile(cfg)));
    let listener = TcpListener::bind(&args.bind).expect("failed to bind server");

    let mut state = ServerState {
        cfg,
        tokenizer,
        model_name: args.model_name,
        default_decode_backend: args.decode_backend,
        add_special_tokens: args.add_special_tokens,
        skip_special_tokens: args.skip_special_tokens,
        cache: SessionCache::new(cfg, kernels, weights, args.softcap, args.max_cache_sessions),
    };

    eprintln!(
        "rustane_serve_ready bind={} model={} vocab={} seq={} cache_sessions={}",
        args.bind, state.model_name, cfg.vocab, cfg.seq, args.max_cache_sessions
    );

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let response = parse_http_request(&mut stream)
                    .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))
                    .and_then(|req| route_request(&mut state, req, &mut stream));
                if let Err(err) = response {
                    let _ = write_error(&mut stream, 400, err.to_string());
                }
            }
            Err(err) => eprintln!("accept failed: {err}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::model::FfnActivation;

    #[test]
    fn longest_prefix_match_prefers_deepest_entry() {
        let cfg = Box::leak(Box::new(ModelConfig {
            dim: 256,
            hidden: 640,
            heads: 2,
            kv_heads: 2,
            hd: 128,
            seq: 64,
            nlayers: 2,
            vocab: 8192,
            q_dim: 256,
            kv_dim: 256,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        }));
        let weights = Box::leak(Box::new(engine::full_model::ModelWeights::random(cfg)));
        let kernels = Box::leak(Box::new(CompiledKernels::compile(cfg)));
        let mut cache = SessionCache::new(cfg, kernels, weights, 0.0, 8);

        cache.insert(&[1, 2]).expect("insert");
        cache.insert(&[1, 2, 3, 4]).expect("insert");
        cache.insert(&[9]).expect("insert");

        let idx = cache.find_longest_prefix(&[1, 2, 3, 4, 5]).expect("prefix");
        assert_eq!(cache.entries[idx].window_prompt_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn normalize_prompt_ids_keeps_last_seq_window() {
        let cfg = Box::leak(Box::new(ModelConfig {
            dim: 256,
            hidden: 640,
            heads: 2,
            kv_heads: 2,
            hd: 128,
            seq: 64,
            nlayers: 2,
            vocab: 8192,
            q_dim: 256,
            kv_dim: 256,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        }));
        let weights = Box::leak(Box::new(engine::full_model::ModelWeights::random(cfg)));
        let kernels = Box::leak(Box::new(CompiledKernels::compile(cfg)));
        let cache = SessionCache::new(cfg, kernels, weights, 0.0, 8);

        assert_eq!(cache.normalize_prompt_ids(&[1, 2, 3]), vec![1, 2, 3]);
        let long_prompt: Vec<u32> = (1..=70).collect();
        let expected: Vec<u32> = (7..=70).collect();
        assert_eq!(cache.normalize_prompt_ids(&long_prompt), expected);
    }

    #[test]
    fn chat_prompt_renderer_flattens_message_content() {
        let prompt = render_chat_prompt(&[
            ChatMessage {
                role: "system".into(),
                content: "Be terse".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ]);
        assert_eq!(prompt, "Be terse\n\nHello");
    }
}
