use engine::decode::DecodeContext;
use engine::full_model::{self, ModelForwardWorkspace, ModelWeights};
use engine::layer::CompiledKernels;
use engine::model::{FfnActivation, ModelConfig};

fn tiny_cfg() -> ModelConfig {
    ModelConfig {
        dim: 256,
        hidden: 640,
        heads: 2,
        kv_heads: 2,
        hd: 128,
        seq: 64,
        nlayers: 2,
        vocab: 64,
        q_dim: 256,
        kv_dim: 256,
        gqa_ratio: 1,
        ffn_activation: FfnActivation::SwiGlu,
    }
}

fn multiblock_vocab_cfg() -> ModelConfig {
    ModelConfig {
        vocab: 320,
        ..tiny_cfg()
    }
}

fn argmax(xs: &[f32]) -> u32 {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn topk(xs: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(u32, f32)> = xs
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, val)| (idx as u32, val))
        .collect();
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(k);
    pairs
}

fn block_stats(xs: &[f32], temperature: f32, block: usize, block_size: usize) -> (f32, f32) {
    let start = block * block_size;
    let end = (start + block_size).min(xs.len());
    let slice = &xs[start..end];
    let inv_temp = 1.0 / temperature;
    let scaled: Vec<f32> = slice.iter().map(|&x| x * inv_temp).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = scaled.iter().map(|&x| (x - max).exp()).sum();
    (max, sum)
}

fn padded_input(cfg: &ModelConfig, tokens: &[u32]) -> Vec<u32> {
    let mut input = vec![0u32; cfg.seq];
    input[..tokens.len()].copy_from_slice(tokens);
    input
}

#[test]
fn kv_decode_matches_naive_greedy_tokens_within_context_limit() {
    let cfg = tiny_cfg();
    let weights = ModelWeights::random(&cfg);
    let kernels = CompiledKernels::compile(&cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut decode = DecodeContext::new(&cfg, &weights);

    let mut generated = vec![1u32, 2, 3];
    assert!(generated.len() + 4 <= cfg.seq);

    let input = padded_input(&cfg, &generated);
    full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
    decode.seed_from_forward_caches(&cfg, &fwd_ws.caches, generated.len());

    let mut next =
        argmax(&fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab]);
    generated.push(next);

    for _ in 0..3 {
        let cached_next = {
            let logits = decode.decode_next_logits(&cfg, &weights, next, 0.0);
            argmax(logits)
        };

        let input = padded_input(&cfg, &generated);
        full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
        let naive_next =
            argmax(&fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab]);

        assert_eq!(cached_next, naive_next);
        next = cached_next;
        generated.push(next);
    }
}

#[test]
fn kv_decode_gpu_argmax_matches_naive_greedy_tokens_within_context_limit() {
    let cfg = tiny_cfg();
    let weights = ModelWeights::random(&cfg);
    let kernels = CompiledKernels::compile(&cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut decode = DecodeContext::new(&cfg, &weights);

    let mut generated = vec![1u32, 2, 3];
    assert!(generated.len() + 4 <= cfg.seq);

    let input = padded_input(&cfg, &generated);
    full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
    decode.seed_from_forward_caches(&cfg, &fwd_ws.caches, generated.len());

    let mut next =
        argmax(&fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab]);
    generated.push(next);

    for _ in 0..3 {
        let cached_next = decode.decode_next_greedy_token(next);

        let input = padded_input(&cfg, &generated);
        full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
        let naive_next =
            argmax(&fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab]);

        assert_eq!(cached_next, naive_next);
        next = cached_next;
        generated.push(next);
    }
}

#[test]
fn kv_decode_gpu_topk_matches_naive_topk_within_context_limit() {
    let cfg = multiblock_vocab_cfg();
    let weights = ModelWeights::random(&cfg);
    let kernels = CompiledKernels::compile(&cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut decode = DecodeContext::new(&cfg, &weights);

    let mut generated = vec![1u32, 2, 3];
    assert!(generated.len() + 2 <= cfg.seq);

    let input = padded_input(&cfg, &generated);
    full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
    decode.seed_from_forward_caches(&cfg, &fwd_ws.caches, generated.len());

    let next =
        argmax(&fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab]);
    generated.push(next);

    let (cached_idxs, cached_logits) = decode.decode_next_topk_candidates(&cfg, next, 4, 0.0);

    let input = padded_input(&cfg, &generated);
    full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
    let naive_topk = topk(
        &fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab],
        4,
    );

    assert_eq!(cached_idxs.len(), naive_topk.len());
    for ((&cached_idx, &cached_logit), (naive_idx, naive_logit)) in cached_idxs
        .iter()
        .zip(cached_logits.iter())
        .zip(naive_topk.iter().copied())
    {
        assert_eq!(cached_idx, naive_idx);
        assert!((cached_logit - naive_logit).abs() < 1e-4);
    }
}

#[test]
fn kv_decode_gpu_block_stats_match_naive_fullvocab_sampling_inputs() {
    let cfg = multiblock_vocab_cfg();
    let weights = ModelWeights::random(&cfg);
    let kernels = CompiledKernels::compile(&cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut decode = DecodeContext::new(&cfg, &weights);

    let generated = vec![1u32, 2, 3, 4];
    let input = padded_input(&cfg, &generated);
    full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
    decode.seed_from_forward_caches(&cfg, &fwd_ws.caches, generated.len());

    let next =
        argmax(&fwd_ws.logits[(generated.len() - 1) * cfg.vocab..generated.len() * cfg.vocab]);
    let temperature = 0.8;
    let (cached_maxes, cached_sums) =
        decode.decode_next_fullvocab_block_stats(&cfg, next, temperature, 0.0);

    let input = padded_input(
        &cfg,
        &[generated.as_slice(), std::slice::from_ref(&next)].concat(),
    );
    full_model::forward_logits_ws(&cfg, &kernels, &weights, &input, 0.0, &mut fwd_ws);
    let naive_logits =
        &fwd_ws.logits[generated.len() * cfg.vocab..(generated.len() + 1) * cfg.vocab];

    for block in 0..cached_maxes.len() {
        let (naive_max, naive_sum) = block_stats(naive_logits, temperature, block, 256);
        assert!((cached_maxes[block] - naive_max).abs() < 1e-4);
        assert!((cached_sums[block] - naive_sum).abs() < 1e-3);
    }

    let block_logits = decode.materialize_current_logits_block(&cfg, 1, 0.0);
    let start = 256;
    let end = (start + block_logits.len()).min(cfg.vocab);
    let naive_block = &naive_logits[start..end];
    for (&a, &b) in block_logits.iter().zip(naive_block.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}
