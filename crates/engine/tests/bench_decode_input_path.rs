use engine::cpu::embedding;
use engine::decode::DecodeContext;
use engine::full_model::ModelWeights;
use engine::model::{FfnActivation, ModelConfig};
use metal_decode::{Config as MetalConfig, FinalHeadInit, LayerInit, Model as MetalDecodeModel};
use std::time::Instant;

fn medium_cfg() -> ModelConfig {
    ModelConfig {
        dim: 768,
        hidden: 2048,
        heads: 6,
        kv_heads: 6,
        hd: 128,
        seq: 128,
        nlayers: 6,
        vocab: 8192,
        q_dim: 768,
        kv_dim: 768,
        gqa_ratio: 1,
        ffn_activation: FfnActivation::SwiGlu,
    }
}

fn metal_cfg(cfg: &ModelConfig) -> MetalConfig {
    MetalConfig {
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        hd: cfg.hd,
        seq: cfg.seq,
        nlayers: cfg.nlayers,
        vocab: cfg.vocab,
        q_dim: cfg.q_dim,
        kv_dim: cfg.kv_dim,
        gqa_ratio: cfg.gqa_ratio,
    }
}

fn build_model(cfg: &ModelConfig, weights: &ModelWeights) -> MetalDecodeModel {
    MetalDecodeModel::new(
        &metal_cfg(cfg),
        weights.layers.iter().map(|layer| LayerInit {
            wq: &layer.wq,
            wk: &layer.wk,
            wv: &layer.wv,
            wo: &layer.wo,
            w1: &layer.w1,
            w3: &layer.w3,
            w2: &layer.w2,
            gamma1: &layer.gamma1,
            gamma2: &layer.gamma2,
        }),
        FinalHeadInit {
            gamma_final: &weights.gamma_final,
            embed: &weights.embed,
        },
    )
    .expect("failed to build metal decode model")
}

fn topk_indices(xs: &[f32], k: usize) -> Vec<u32> {
    let mut pairs: Vec<(u32, f32)> = xs
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, val)| (idx as u32, val))
        .collect();
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(k);
    pairs.into_iter().map(|(idx, _)| idx).collect()
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

fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &logit in logits {
        let p = ((logit - max_logit) / temperature).exp();
        probs.push(p);
        sum += p;
    }
    let mut draw = next_f32(rng) * sum;
    for (i, &p) in probs.iter().enumerate() {
        draw -= p;
        if draw <= 0.0 {
            return i as u32;
        }
    }
    (logits.len().saturating_sub(1)) as u32
}

fn sample_block_from_stats(block_maxes: &[f32], block_sums: &[f32], rng: &mut u64) -> usize {
    let global_max = block_maxes
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    let mut weights = Vec::with_capacity(block_maxes.len());
    for (&mx, &sum) in block_maxes.iter().zip(block_sums.iter()) {
        let weight = if mx.is_finite() && sum.is_finite() && sum > 0.0 {
            (mx - global_max).exp() * sum
        } else {
            0.0
        };
        weights.push(weight);
        total += weight;
    }
    let mut draw = next_f32(rng) * total;
    for (idx, &weight) in weights.iter().enumerate() {
        draw -= weight;
        if draw <= 0.0 {
            return idx;
        }
    }
    weights.len().saturating_sub(1)
}

#[test]
#[ignore]
fn bench_greedy_decode_cpu_vs_gpu_embedding() {
    let cfg = medium_cfg();
    let weights = ModelWeights::random(&cfg);
    let steps = 48usize;

    let mut cpu_embed_model = build_model(&cfg, &weights);
    let mut gpu_embed_model = build_model(&cfg, &weights);

    let mut token_x = vec![0.0f32; cfg.dim];

    let mut current = 1u32;
    let t0 = Instant::now();
    for _ in 0..steps {
        embedding::forward(&weights.embed, cfg.dim, &[current], &mut token_x);
        current = cpu_embed_model.decode_token_argmax(&token_x);
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    let mut current = 1u32;
    let t1 = Instant::now();
    for _ in 0..steps {
        current = gpu_embed_model.decode_token_argmax_from_id(current);
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    println!(
        "cpu_embed_ms_per_token={:.3} gpu_embed_ms_per_token={:.3} speedup={:.3}x",
        cpu_ms,
        gpu_ms,
        cpu_ms / gpu_ms
    );
}

#[test]
#[ignore]
fn bench_topk_decode_full_logits_vs_gpu_candidates() {
    let cfg = medium_cfg();
    let weights = ModelWeights::random(&cfg);
    let steps = 48usize;
    let top_k = 8usize;

    let mut full_logits_ctx = DecodeContext::new(&cfg, &weights);
    let mut gpu_topk_ctx = DecodeContext::new(&cfg, &weights);

    let t0 = Instant::now();
    for _ in 0..steps {
        let logits = full_logits_ctx.decode_next_logits(&cfg, &weights, 1, 0.0);
        let _ = topk_indices(logits, top_k);
    }
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    let t1 = Instant::now();
    for _ in 0..steps {
        let _ = gpu_topk_ctx.decode_next_topk_candidates(&cfg, 1, top_k, 0.0);
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    println!(
        "full_logits_ms_per_token={:.3} gpu_topk_ms_per_token={:.3} speedup={:.3}x",
        full_ms,
        gpu_ms,
        full_ms / gpu_ms
    );
}

#[test]
#[ignore]
fn bench_fullvocab_decode_full_logits_vs_gpu_block_sampling() {
    let cfg = medium_cfg();
    let weights = ModelWeights::random(&cfg);
    let steps = 48usize;
    let temperature = 0.8f32;
    let mut rng_a = 1u64;
    let mut rng_b = 1u64;

    let mut full_logits_ctx = DecodeContext::new(&cfg, &weights);
    let mut gpu_block_ctx = DecodeContext::new(&cfg, &weights);

    let t0 = Instant::now();
    for _ in 0..steps {
        let logits = full_logits_ctx.decode_next_logits(&cfg, &weights, 1, 0.0);
        let _ = sample_from_logits(logits, temperature, &mut rng_a);
    }
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    let t1 = Instant::now();
    for _ in 0..steps {
        let (block_maxes, block_sums) =
            gpu_block_ctx.decode_next_fullvocab_block_stats(&cfg, 1, temperature, 0.0);
        let block_idx = sample_block_from_stats(block_maxes, block_sums, &mut rng_b);
        let block_logits = gpu_block_ctx.materialize_current_logits_block(&cfg, block_idx, 0.0);
        let _ = sample_from_logits(block_logits, temperature, &mut rng_b);
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    println!(
        "full_logits_ms_per_token={:.3} gpu_block_ms_per_token={:.3} speedup={:.3}x",
        full_ms,
        gpu_ms,
        full_ms / gpu_ms
    );
}
