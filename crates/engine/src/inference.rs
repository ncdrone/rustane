//! Reusable inference session API for checkpoint-backed generation.

use crate::decode::{DecodeContext, MAX_GPU_TOPK};
use crate::full_model::{self, ModelForwardWorkspace, ModelWeights};
use crate::layer::{CompiledKernels, ForwardCache};
use crate::model::ModelConfig;
use std::time::{Duration, Instant};

pub const AUTO_METAL_PARAM_THRESHOLD: usize = 20_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeBackendRequest {
    Auto,
    Naive,
    Metal,
}

impl DecodeBackendRequest {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "auto" => Some(Self::Auto),
            "naive" => Some(Self::Naive),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Naive => "naive",
            Self::Metal => "metal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeMode {
    PrefillOnly,
    NaiveFullContext,
    KvCacheMetal,
}

impl DecodeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrefillOnly => "prefill_only",
            Self::NaiveFullContext => "naive_full_context",
            Self::KvCacheMetal => "kv_cache_metal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeSelection {
    pub mode: DecodeMode,
    pub fallback_reason: Option<&'static str>,
    pub auto_reason: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationRequest {
    pub decode_backend: DecodeBackendRequest,
    pub steps: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub samples: usize,
    pub seed: u64,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            decode_backend: DecodeBackendRequest::Auto,
            steps: 32,
            temperature: 0.0,
            top_k: 0,
            samples: 1,
            seed: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SampleResult {
    pub seed: u64,
    pub generated: Vec<u32>,
    pub sampled: Vec<u32>,
    pub decode_elapsed: Duration,
}

#[derive(Clone, Debug)]
pub struct GenerationBatch {
    pub decode_selection: DecodeSelection,
    pub base_prefill_elapsed: Duration,
    pub request_prefill_elapsed: Duration,
    pub prefill_elapsed: Duration,
    pub metal_decode_compile_elapsed: Option<Duration>,
    pub total_decode_elapsed: Duration,
    pub avg_decode_elapsed: Duration,
    pub total_elapsed: Duration,
    pub results: Vec<SampleResult>,
}

type StreamCallback<'a> = dyn FnMut(u32, &[u32], &[u32]) -> Result<(), String> + 'a;

pub struct InferenceSession<'a> {
    cfg: &'a ModelConfig,
    kernels: &'a CompiledKernels,
    weights: &'a ModelWeights,
    prompt_ids: Vec<u32>,
    prompt_len: usize,
    softcap: f32,
    prefill_elapsed: Duration,
    prompt_logits: Vec<f32>,
    prompt_ws: ModelForwardWorkspace,
    naive_ws: Option<ModelForwardWorkspace>,
    input: Vec<u32>,
    sampler: SamplerWorkspace,
    decode_ctx: Option<DecodeContext>,
}

impl<'a> InferenceSession<'a> {
    pub fn new(
        cfg: &'a ModelConfig,
        kernels: &'a CompiledKernels,
        weights: &'a ModelWeights,
        prompt_ids: &[u32],
        softcap: f32,
    ) -> Result<Self, String> {
        if prompt_ids.is_empty() {
            return Err("prompt must contain at least one token".into());
        }
        validate_token_ids(prompt_ids, cfg.vocab, "prompt")?;

        let mut prompt_ws = ModelForwardWorkspace::new(cfg);
        let mut input = vec![0u32; cfg.seq];
        let prompt_len = fill_input_window(prompt_ids, cfg.seq, &mut input);
        let t0 = Instant::now();
        full_model::forward_logits_ws(cfg, kernels, weights, &input, softcap, &mut prompt_ws);
        let prefill_elapsed = t0.elapsed();
        let last_pos = prompt_len - 1;
        let prompt_logits =
            prompt_ws.logits[last_pos * cfg.vocab..(last_pos + 1) * cfg.vocab].to_vec();

        Ok(Self {
            cfg,
            kernels,
            weights,
            prompt_ids: prompt_ids.to_vec(),
            prompt_len,
            softcap,
            prefill_elapsed,
            prompt_logits,
            prompt_ws,
            naive_ws: None,
            input,
            sampler: SamplerWorkspace::new(cfg.vocab),
            decode_ctx: None,
        })
    }

    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn prefill_elapsed(&self) -> Duration {
        self.prefill_elapsed
    }

    pub fn softcap(&self) -> f32 {
        self.softcap
    }

    pub fn generate(&mut self, req: &GenerationRequest) -> Result<GenerationBatch, String> {
        self.generate_with_suffix(req, &[])
    }

    pub fn generate_with_suffix(
        &mut self,
        req: &GenerationRequest,
        prompt_suffix: &[u32],
    ) -> Result<GenerationBatch, String> {
        if req.samples == 0 {
            return Err("samples must be >= 1".into());
        }
        validate_token_ids(prompt_suffix, self.cfg.vocab, "prompt suffix")?;
        let effective_prompt_len = (self.prompt_len + prompt_suffix.len()).min(self.cfg.seq);
        let decode_selection = select_decode_mode(
            self.cfg,
            req.decode_backend,
            effective_prompt_len,
            req.steps,
        );
        let use_kv_cache = decode_selection.mode == DecodeMode::KvCacheMetal;
        let mut request_prefill_elapsed = Duration::ZERO;
        let effective_prompt_ids = if prompt_suffix.is_empty() {
            self.prompt_ids.clone()
        } else {
            let mut ids = self.prompt_ids.clone();
            ids.extend_from_slice(prompt_suffix);
            ids
        };

        let mut metal_decode_compile_elapsed = None;
        if use_kv_cache && self.decode_ctx.is_none() {
            let t0 = Instant::now();
            self.decode_ctx = Some(DecodeContext::new(self.cfg, self.weights));
            metal_decode_compile_elapsed = Some(t0.elapsed());
        }

        let prompt_logits = if prompt_suffix.is_empty() {
            self.prompt_logits.clone()
        } else if use_kv_cache {
            let ctx = self
                .decode_ctx
                .as_mut()
                .expect("metal decode context must exist when kv-cache is selected");
            let t0 = Instant::now();
            let logits = prefill_prompt_suffix_cached(
                self.cfg,
                self.weights,
                &self.prompt_logits,
                &self.prompt_ws.caches,
                self.prompt_len,
                prompt_suffix,
                self.softcap,
                ctx,
            );
            request_prefill_elapsed = t0.elapsed();
            logits
        } else {
            let ws = self
                .naive_ws
                .get_or_insert_with(|| ModelForwardWorkspace::new(self.cfg));
            let t0 = Instant::now();
            let logits = prefill_prompt_suffix_naive(
                self.cfg,
                self.kernels,
                self.weights,
                ws,
                &mut self.input,
                &effective_prompt_ids,
                self.softcap,
            );
            request_prefill_elapsed = t0.elapsed();
            logits
        };

        let mut results = Vec::with_capacity(req.samples);
        for sample_idx in 0..req.samples {
            let sample_seed = req.seed.wrapping_add(sample_idx as u64);
            let result = if use_kv_cache {
                let ctx = self
                    .decode_ctx
                    .as_mut()
                    .expect("metal decode context must exist when kv-cache is selected");
                run_cached_sample(
                    self.cfg,
                    self.weights,
                    &effective_prompt_ids,
                    &prompt_logits,
                    &self.prompt_ws.caches,
                    self.prompt_len,
                    prompt_suffix,
                    self.softcap,
                    req,
                    ctx,
                    &mut self.sampler,
                    sample_seed,
                    None,
                )?
            } else {
                let ws = self
                    .naive_ws
                    .get_or_insert_with(|| ModelForwardWorkspace::new(self.cfg));
                run_naive_sample(
                    self.cfg,
                    self.kernels,
                    self.weights,
                    ws,
                    &mut self.input,
                    &effective_prompt_ids,
                    &prompt_logits,
                    self.softcap,
                    req,
                    &mut self.sampler,
                    sample_seed,
                    None,
                )?
            };
            results.push(result);
        }

        let total_decode_elapsed: Duration = results.iter().map(|r| r.decode_elapsed).sum();
        let avg_decode_elapsed = if req.samples > 0 {
            total_decode_elapsed.div_f64(req.samples as f64)
        } else {
            Duration::ZERO
        };
        let prefill_elapsed = self.prefill_elapsed + request_prefill_elapsed;
        let total_elapsed = prefill_elapsed + total_decode_elapsed;

        Ok(GenerationBatch {
            decode_selection,
            base_prefill_elapsed: self.prefill_elapsed,
            request_prefill_elapsed,
            prefill_elapsed,
            metal_decode_compile_elapsed,
            total_decode_elapsed,
            avg_decode_elapsed,
            total_elapsed,
            results,
        })
    }

    pub fn generate_stream_with_suffix<F>(
        &mut self,
        req: &GenerationRequest,
        prompt_suffix: &[u32],
        mut on_token: F,
    ) -> Result<GenerationBatch, String>
    where
        F: FnMut(u32, &[u32], &[u32]) -> Result<(), String>,
    {
        if req.samples != 1 {
            return Err("streaming only supports samples=1".into());
        }
        validate_token_ids(prompt_suffix, self.cfg.vocab, "prompt suffix")?;
        let effective_prompt_len = (self.prompt_len + prompt_suffix.len()).min(self.cfg.seq);
        let decode_selection = select_decode_mode(
            self.cfg,
            req.decode_backend,
            effective_prompt_len,
            req.steps,
        );
        let use_kv_cache = decode_selection.mode == DecodeMode::KvCacheMetal;
        let mut request_prefill_elapsed = Duration::ZERO;
        let effective_prompt_ids = if prompt_suffix.is_empty() {
            self.prompt_ids.clone()
        } else {
            let mut ids = self.prompt_ids.clone();
            ids.extend_from_slice(prompt_suffix);
            ids
        };

        let mut metal_decode_compile_elapsed = None;
        if use_kv_cache && self.decode_ctx.is_none() {
            let t0 = Instant::now();
            self.decode_ctx = Some(DecodeContext::new(self.cfg, self.weights));
            metal_decode_compile_elapsed = Some(t0.elapsed());
        }

        let prompt_logits = if prompt_suffix.is_empty() {
            self.prompt_logits.clone()
        } else if use_kv_cache {
            let ctx = self
                .decode_ctx
                .as_mut()
                .expect("metal decode context must exist when kv-cache is selected");
            let t0 = Instant::now();
            let logits = prefill_prompt_suffix_cached(
                self.cfg,
                self.weights,
                &self.prompt_logits,
                &self.prompt_ws.caches,
                self.prompt_len,
                prompt_suffix,
                self.softcap,
                ctx,
            );
            request_prefill_elapsed = t0.elapsed();
            logits
        } else {
            let ws = self
                .naive_ws
                .get_or_insert_with(|| ModelForwardWorkspace::new(self.cfg));
            let t0 = Instant::now();
            let logits = prefill_prompt_suffix_naive(
                self.cfg,
                self.kernels,
                self.weights,
                ws,
                &mut self.input,
                &effective_prompt_ids,
                self.softcap,
            );
            request_prefill_elapsed = t0.elapsed();
            logits
        };

        let mut stream_cb: &mut StreamCallback<'_> = &mut on_token;
        let sample = if use_kv_cache {
            let ctx = self
                .decode_ctx
                .as_mut()
                .expect("metal decode context must exist when kv-cache is selected");
            run_cached_sample(
                self.cfg,
                self.weights,
                &effective_prompt_ids,
                &prompt_logits,
                &self.prompt_ws.caches,
                self.prompt_len,
                prompt_suffix,
                self.softcap,
                req,
                ctx,
                &mut self.sampler,
                req.seed,
                Some(&mut stream_cb),
            )?
        } else {
            let ws = self
                .naive_ws
                .get_or_insert_with(|| ModelForwardWorkspace::new(self.cfg));
            run_naive_sample(
                self.cfg,
                self.kernels,
                self.weights,
                ws,
                &mut self.input,
                &effective_prompt_ids,
                &prompt_logits,
                self.softcap,
                req,
                &mut self.sampler,
                req.seed,
                Some(&mut stream_cb),
            )?
        };

        let total_decode_elapsed = sample.decode_elapsed;
        let prefill_elapsed = self.prefill_elapsed + request_prefill_elapsed;
        let total_elapsed = prefill_elapsed + total_decode_elapsed;
        Ok(GenerationBatch {
            decode_selection,
            base_prefill_elapsed: self.prefill_elapsed,
            request_prefill_elapsed,
            prefill_elapsed,
            metal_decode_compile_elapsed,
            total_decode_elapsed,
            avg_decode_elapsed: total_decode_elapsed,
            total_elapsed,
            results: vec![sample],
        })
    }
}

pub fn auto_prefers_metal(cfg: &ModelConfig) -> bool {
    cfg.param_count() >= AUTO_METAL_PARAM_THRESHOLD
}

pub fn select_decode_mode(
    cfg: &ModelConfig,
    requested: DecodeBackendRequest,
    effective_context_len: usize,
    steps: usize,
) -> DecodeSelection {
    if steps == 0 {
        return DecodeSelection {
            mode: DecodeMode::PrefillOnly,
            fallback_reason: None,
            auto_reason: None,
        };
    }

    let metal_window_ok = steps > 1 && effective_context_len + steps <= cfg.seq;
    match requested {
        DecodeBackendRequest::Naive => DecodeSelection {
            mode: DecodeMode::NaiveFullContext,
            fallback_reason: None,
            auto_reason: None,
        },
        DecodeBackendRequest::Metal => {
            if metal_window_ok {
                DecodeSelection {
                    mode: DecodeMode::KvCacheMetal,
                    fallback_reason: None,
                    auto_reason: None,
                }
            } else {
                DecodeSelection {
                    mode: DecodeMode::NaiveFullContext,
                    fallback_reason: Some(if steps <= 1 {
                        "single_step_prefers_naive"
                    } else {
                        "requested_length_exceeds_cfg_seq"
                    }),
                    auto_reason: None,
                }
            }
        }
        DecodeBackendRequest::Auto => {
            if !metal_window_ok {
                DecodeSelection {
                    mode: DecodeMode::NaiveFullContext,
                    fallback_reason: if effective_context_len + steps > cfg.seq && steps > 1 {
                        Some("requested_length_exceeds_cfg_seq")
                    } else {
                        None
                    },
                    auto_reason: Some(if steps <= 1 {
                        "single_step_prefers_naive"
                    } else {
                        "window_not_metal_eligible"
                    }),
                }
            } else if auto_prefers_metal(cfg) {
                DecodeSelection {
                    mode: DecodeMode::KvCacheMetal,
                    fallback_reason: None,
                    auto_reason: Some("model_param_count_ge_threshold"),
                }
            } else {
                DecodeSelection {
                    mode: DecodeMode::NaiveFullContext,
                    fallback_reason: None,
                    auto_reason: Some("model_param_count_lt_threshold"),
                }
            }
        }
    }
}

struct SamplerWorkspace {
    indices: Vec<usize>,
}

impl SamplerWorkspace {
    fn new(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
        }
    }
}

fn run_cached_sample(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_ids: &[u32],
    prompt_logits: &[f32],
    prompt_caches: &[ForwardCache],
    prompt_len: usize,
    prompt_suffix: &[u32],
    softcap: f32,
    req: &GenerationRequest,
    ctx: &mut DecodeContext,
    sampler: &mut SamplerWorkspace,
    seed: u64,
    mut on_token: Option<&mut StreamCallback<'_>>,
) -> Result<SampleResult, String> {
    let mut generated = prompt_ids.to_vec();
    let mut sampled = Vec::with_capacity(req.steps);
    let mut rng = seed;

    ctx.seed_from_forward_caches(cfg, prompt_caches, prompt_len);
    let decode_start = Instant::now();
    for &tok in prompt_suffix {
        let _ = ctx.decode_next_logits(cfg, weights, tok, softcap);
    }

    if req.steps > 0 {
        let next = sample_next(prompt_logits, req.temperature, req.top_k, &mut rng, sampler);
        generated.push(next);
        sampled.push(next);
        if let Some(cb) = on_token.as_mut() {
            cb(next, &sampled, &generated)?;
        }
    }
    if let Some(mut current) = sampled.first().copied() {
        if req.temperature <= 0.0 || req.top_k == 1 {
            for _ in 1..req.steps {
                let next = ctx.decode_next_greedy_token(current);
                generated.push(next);
                sampled.push(next);
                if let Some(cb) = on_token.as_mut() {
                    cb(next, &sampled, &generated)?;
                }
                current = next;
            }
        } else if req.top_k > 0 && req.top_k <= MAX_GPU_TOPK {
            for _ in 1..req.steps {
                let next = {
                    let (idxs, logits) =
                        ctx.decode_next_topk_candidates(cfg, current, req.top_k, softcap);
                    sample_topk_candidates(idxs, logits, req.temperature, &mut rng)
                };
                generated.push(next);
                sampled.push(next);
                if let Some(cb) = on_token.as_mut() {
                    cb(next, &sampled, &generated)?;
                }
                current = next;
            }
        } else if req.top_k == 0 {
            for _ in 1..req.steps {
                let next = {
                    let (block_maxes, block_sums) = ctx.decode_next_fullvocab_block_stats(
                        cfg,
                        current,
                        req.temperature,
                        softcap,
                    );
                    let block_idx = sample_block_from_stats(block_maxes, block_sums, &mut rng);
                    let block_logits =
                        ctx.materialize_current_logits_block(cfg, block_idx, softcap);
                    sample_block_logits(
                        (block_idx * 256) as u32,
                        block_logits,
                        req.temperature,
                        &mut rng,
                    )
                };
                generated.push(next);
                sampled.push(next);
                if let Some(cb) = on_token.as_mut() {
                    cb(next, &sampled, &generated)?;
                }
                current = next;
            }
        } else {
            for _ in 1..req.steps {
                let next = {
                    let logits = ctx.decode_next_logits(cfg, weights, current, softcap);
                    sample_next(logits, req.temperature, req.top_k, &mut rng, sampler)
                };
                generated.push(next);
                sampled.push(next);
                if let Some(cb) = on_token.as_mut() {
                    cb(next, &sampled, &generated)?;
                }
                current = next;
            }
        }
    }

    Ok(SampleResult {
        seed,
        generated,
        sampled,
        decode_elapsed: decode_start.elapsed(),
    })
}

fn prefill_prompt_suffix_cached(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_logits: &[f32],
    prompt_caches: &[ForwardCache],
    prompt_len: usize,
    prompt_suffix: &[u32],
    softcap: f32,
    ctx: &mut DecodeContext,
) -> Vec<f32> {
    if prompt_suffix.is_empty() {
        return prompt_logits.to_vec();
    }

    ctx.seed_from_forward_caches(cfg, prompt_caches, prompt_len);
    let mut logits = prompt_logits.to_vec();
    for &tok in prompt_suffix {
        logits = ctx.decode_next_logits(cfg, weights, tok, softcap).to_vec();
    }
    logits
}

fn prefill_prompt_suffix_naive(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    ws: &mut ModelForwardWorkspace,
    input: &mut [u32],
    prompt_ids: &[u32],
    softcap: f32,
) -> Vec<f32> {
    let context_len = fill_input_window(prompt_ids, cfg.seq, input);
    full_model::forward_logits_ws(cfg, kernels, weights, input, softcap, ws);
    let last_pos = context_len - 1;
    ws.logits[last_pos * cfg.vocab..(last_pos + 1) * cfg.vocab].to_vec()
}

fn run_naive_sample(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    ws: &mut ModelForwardWorkspace,
    input: &mut [u32],
    prompt_ids: &[u32],
    prompt_logits: &[f32],
    softcap: f32,
    req: &GenerationRequest,
    sampler: &mut SamplerWorkspace,
    seed: u64,
    mut on_token: Option<&mut StreamCallback<'_>>,
) -> Result<SampleResult, String> {
    let mut generated = prompt_ids.to_vec();
    let mut sampled = Vec::with_capacity(req.steps);
    let mut rng = seed;

    if req.steps > 0 {
        let next = sample_next(prompt_logits, req.temperature, req.top_k, &mut rng, sampler);
        generated.push(next);
        sampled.push(next);
        if let Some(cb) = on_token.as_mut() {
            cb(next, &sampled, &generated)?;
        }
    }

    let decode_start = Instant::now();
    for _ in 1..req.steps {
        let context_len = fill_input_window(&generated, cfg.seq, input);
        full_model::forward_logits_ws(cfg, kernels, weights, input, softcap, ws);
        let last_pos = context_len - 1;
        let logits = &ws.logits[last_pos * cfg.vocab..(last_pos + 1) * cfg.vocab];
        let next = sample_next(logits, req.temperature, req.top_k, &mut rng, sampler);
        generated.push(next);
        sampled.push(next);
        if let Some(cb) = on_token.as_mut() {
            cb(next, &sampled, &generated)?;
        }
    }

    Ok(SampleResult {
        seed,
        generated,
        sampled,
        decode_elapsed: decode_start.elapsed(),
    })
}

fn validate_token_ids(token_ids: &[u32], vocab: usize, source: &str) -> Result<(), String> {
    if let Some((pos, tok)) = token_ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, tok)| *tok as usize >= vocab)
    {
        return Err(format!(
            "{source} token id {} at position {} exceeds model vocab {}",
            tok, pos, vocab
        ));
    }
    Ok(())
}

fn fill_input_window(generated: &[u32], seq: usize, input: &mut [u32]) -> usize {
    input.fill(0);
    let context_start = generated.len().saturating_sub(seq);
    let context = &generated[context_start..];
    input[..context.len()].copy_from_slice(context);
    context.len()
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn next_f32(state: &mut u64) -> f32 {
    ((next_u64(state) >> 40) as f32) / ((1u64 << 24) as f32)
}

fn sample_next(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut u64,
    sampler: &mut SamplerWorkspace,
) -> u32 {
    if temperature <= 0.0 || top_k == 1 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
    }

    if top_k == 0 || top_k >= logits.len() {
        return sample_softmax_index(logits, temperature, rng) as u32;
    }

    sampler.indices.clear();
    sampler.indices.extend(0..logits.len());
    sampler
        .indices
        .sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if top_k < sampler.indices.len() {
        sampler.indices.truncate(top_k);
    }

    let max_logit = sampler
        .indices
        .iter()
        .map(|&idx| logits[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &idx in &sampler.indices {
        sum += ((logits[idx] - max_logit) / temperature).exp();
    }

    if !sum.is_finite() || sum <= 0.0 {
        return sampler.indices[0] as u32;
    }

    let mut draw = next_f32(rng) * sum;
    for &idx in &sampler.indices {
        draw -= ((logits[idx] - max_logit) / temperature).exp();
        if draw <= 0.0 {
            return idx as u32;
        }
    }
    *sampler.indices.last().unwrap_or(&0) as u32
}

fn sample_topk_candidates(idxs: &[u32], logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    if idxs.is_empty() {
        return 0;
    }
    if temperature <= 0.0 || idxs.len() == 1 {
        return idxs[0];
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &logit in logits {
        sum += ((logit - max_logit) / temperature).exp();
    }

    if !sum.is_finite() || sum <= 0.0 {
        return idxs[0];
    }

    let mut draw = next_f32(rng) * sum;
    for (i, &logit) in logits.iter().enumerate() {
        draw -= ((logit - max_logit) / temperature).exp();
        if draw <= 0.0 {
            return idxs[i];
        }
    }
    *idxs.last().unwrap_or(&0)
}

fn sample_block_from_stats(block_maxes: &[f32], block_sums: &[f32], rng: &mut u64) -> usize {
    if block_maxes.is_empty() || block_sums.is_empty() {
        return 0;
    }

    let global_max = block_maxes
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !global_max.is_finite() {
        return 0;
    }

    let mut total = 0.0f32;
    for (&mx, &sum) in block_maxes.iter().zip(block_sums.iter()) {
        let weight = if mx.is_finite() && sum.is_finite() && sum > 0.0 {
            (mx - global_max).exp() * sum
        } else {
            0.0
        };
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        return 0;
    }

    let mut draw = next_f32(rng) * total;
    for (idx, (&mx, &sum)) in block_maxes.iter().zip(block_sums.iter()).enumerate() {
        let weight = if mx.is_finite() && sum.is_finite() && sum > 0.0 {
            (mx - global_max).exp() * sum
        } else {
            0.0
        };
        draw -= weight;
        if draw <= 0.0 {
            return idx;
        }
    }
    block_maxes.len().saturating_sub(1)
}

fn sample_softmax_index(logits: &[f32], temperature: f32, rng: &mut u64) -> usize {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &logit in logits {
        sum += ((logit - max_logit) / temperature).exp();
    }

    if !sum.is_finite() || sum <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    let mut draw = next_f32(rng) * sum;
    for (idx, &logit) in logits.iter().enumerate() {
        draw -= ((logit - max_logit) / temperature).exp();
        if draw <= 0.0 {
            return idx;
        }
    }
    logits.len().saturating_sub(1)
}

fn sample_block_logits(base: u32, logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    base + sample_softmax_index(logits, temperature, rng) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_model::ModelWeights;
    use crate::model::FfnActivation;

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
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
        }
    }

    #[test]
    fn auto_prefers_naive_on_tiny_model() {
        let selected = select_decode_mode(&tiny_cfg(), DecodeBackendRequest::Auto, 4, 8);
        assert_eq!(selected.mode, DecodeMode::NaiveFullContext);
        assert_eq!(selected.auto_reason, Some("model_param_count_lt_threshold"));
    }

    #[test]
    fn auto_prefers_metal_on_gpt_1024() {
        let selected =
            select_decode_mode(&ModelConfig::gpt_1024(), DecodeBackendRequest::Auto, 4, 32);
        assert_eq!(selected.mode, DecodeMode::KvCacheMetal);
        assert_eq!(selected.auto_reason, Some("model_param_count_ge_threshold"));
    }

    #[test]
    fn forced_metal_falls_back_when_window_exceeds_seq() {
        let selected = select_decode_mode(
            &ModelConfig::gpt_1024(),
            DecodeBackendRequest::Metal,
            500,
            20,
        );
        assert_eq!(selected.mode, DecodeMode::NaiveFullContext);
        assert_eq!(
            selected.fallback_reason,
            Some("requested_length_exceeds_cfg_seq")
        );
    }

    #[test]
    fn session_reuses_prompt_across_calls_with_stable_results() {
        let cfg = tiny_cfg();
        let weights = ModelWeights::random(&cfg);
        let kernels = CompiledKernels::compile(&cfg);
        let prompt = vec![1u32, 2, 3, 4];
        let mut session =
            InferenceSession::new(&cfg, &kernels, &weights, &prompt, 0.0).expect("session");
        let req = GenerationRequest {
            decode_backend: DecodeBackendRequest::Metal,
            steps: 8,
            temperature: 0.8,
            top_k: 4,
            samples: 2,
            seed: 7,
        };

        let first = session.generate(&req).expect("first run");
        let second = session.generate(&req).expect("second run");

        assert_eq!(first.results.len(), 2);
        assert_eq!(first.results[0].generated, second.results[0].generated);
        assert_eq!(first.results[1].generated, second.results[1].generated);
    }

    #[test]
    fn session_suffix_matches_fresh_full_prompt_generation() {
        let cfg = tiny_cfg();
        let weights = ModelWeights::random(&cfg);
        let kernels = CompiledKernels::compile(&cfg);
        let base_prompt = vec![1u32, 2, 3];
        let suffix = vec![4u32, 5];
        let full_prompt = [base_prompt.clone(), suffix.clone()].concat();
        let req = GenerationRequest {
            decode_backend: DecodeBackendRequest::Metal,
            steps: 6,
            temperature: 0.8,
            top_k: 4,
            samples: 2,
            seed: 17,
        };

        let mut base_session =
            InferenceSession::new(&cfg, &kernels, &weights, &base_prompt, 0.0).expect("session");
        let mut full_session =
            InferenceSession::new(&cfg, &kernels, &weights, &full_prompt, 0.0).expect("session");

        let extended = base_session
            .generate_with_suffix(&req, &suffix)
            .expect("extended run");
        let fresh = full_session.generate(&req).expect("fresh run");

        assert_eq!(extended.results.len(), fresh.results.len());
        for (lhs, rhs) in extended.results.iter().zip(fresh.results.iter()) {
            assert_eq!(lhs.generated, rhs.generated);
            assert_eq!(lhs.sampled, rhs.sampled);
        }
    }
}
