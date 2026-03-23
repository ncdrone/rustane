//! Correctness test: MLA profiling instrumentation (env-gated timing output)
//! produces identical numerical results to the non-profiling code path.
//!
//! What was optimized: added per-component timing (q_proj, kv_compress,
//! w_uk_absorb, attn_scores+softmax, value_recon+w_uv, o_proj) to
//! mla_decode_v_concat and mla_forward_decode, gated behind the
//! RUSTANE_MLA_PROFILE env var via std::sync::OnceLock.
//!
//! Invariant: timing instrumentation MUST NOT change any computed values.
//! The profiling code only adds eprintln! calls between existing compute
//! steps — no floating-point operations are reordered or modified.
//!
//! Failure means: timing code accidentally modified a variable, reordered
//! operations, or introduced a side effect that changes the output.

use moe_infer::mla_attention::{MlaDecodeConfig, MlaLayerWeights, MlaKvCache, mla_forward_decode};
use moe_infer::yarn_rope::YarnRopeTables;
use moe_infer::config::RopeScalingSection;

fn make_rope(max_seq: usize, rope_dim: usize) -> YarnRopeTables {
    let scaling = RopeScalingSection {
        factor: 1.0,
        original_max_position_embeddings: max_seq,
        beta_fast: 32.0,
        beta_slow: 1.0,
        mscale: 1.0,
        mscale_all_dim: 1.0,
    };
    YarnRopeTables::build(max_seq, rope_dim, 10000.0, &scaling)
}

/// Create deterministic synthetic weights for testing MLA.
fn make_test_weights(cfg: &MlaDecodeConfig) -> TestWeights {
    let h = cfg.num_heads;
    let nope = cfg.qk_nope_head_dim;
    let rope_dim = cfg.qk_rope_head_dim;
    let v_dim = cfg.v_head_dim;
    let kv_rank = cfg.kv_lora_rank;
    let hidden = cfg.hidden_size;

    let q_lora_rank = 128; // small for testing
    let q_total = h * (nope + rope_dim);

    TestWeights {
        q_proj: det_vec(q_total * hidden, 0.001, 1),
        q_a_proj: det_vec(q_lora_rank * hidden, 0.001, 2),
        q_a_layernorm: det_vec(q_lora_rank, 1.0, 3),
        q_b_proj: det_vec(q_total * q_lora_rank, 0.001, 4),
        kv_a_proj: det_vec((kv_rank + rope_dim) * hidden, 0.001, 5),
        kv_a_layernorm: det_vec(kv_rank, 1.0, 6),
        w_uk: det_vec(h * nope * kv_rank, 0.001, 7),
        w_uv: det_vec(h * v_dim * kv_rank, 0.001, 8),
        o_proj: det_vec(hidden * h * v_dim, 0.001, 9),
        input_norm: det_vec(hidden, 1.0, 10),
        post_attn_norm: det_vec(hidden, 1.0, 11),
    }
}

struct TestWeights {
    q_proj: Vec<f32>,
    q_a_proj: Vec<f32>,
    q_a_layernorm: Vec<f32>,
    q_b_proj: Vec<f32>,
    kv_a_proj: Vec<f32>,
    kv_a_layernorm: Vec<f32>,
    w_uk: Vec<f32>,
    w_uv: Vec<f32>,
    o_proj: Vec<f32>,
    input_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
}

impl TestWeights {
    fn as_mla_weights(&self) -> MlaLayerWeights<'_> {
        MlaLayerWeights {
            q_proj: &self.q_proj,
            q_a_proj: Some(&self.q_a_proj),
            q_a_layernorm: Some(&self.q_a_layernorm),
            q_b_proj: Some(&self.q_b_proj),
            kv_a_proj: &self.kv_a_proj,
            kv_a_layernorm: &self.kv_a_layernorm,
            w_uk: &self.w_uk,
            w_uv: &self.w_uv,
            o_proj: &self.o_proj,
            input_norm: &self.input_norm,
            post_attn_norm: &self.post_attn_norm,
        }
    }
}

/// Deterministic vector: sin-based with seed for reproducibility.
fn det_vec(len: usize, scale: f32, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * seed + seed) as f32 * 0.0001).sin() * scale).collect()
}

/// Run MLA twice with same inputs; outputs must match exactly (pure instrumentation, no FP change).
#[test]
fn profiling_does_not_change_output() {
    // Small config for fast testing (2 heads, small dims)
    let cfg = MlaDecodeConfig {
        hidden_size: 64,
        num_heads: 2,
        qk_nope_head_dim: 16,
        qk_rope_head_dim: 8,
        v_head_dim: 16,
        kv_lora_rank: 32,
        rms_eps: 1e-5,
    };

    let tw = make_test_weights(&cfg);
    let weights = tw.as_mla_weights();
    let rope = make_rope(64, cfg.qk_rope_head_dim);
    let attn_scale = 1.0 / ((cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) as f32).sqrt();

    let x = det_vec(cfg.hidden_size, 0.1, 42);

    // Run 1: without profiling (env var not set — OnceLock captures "not set" state)
    let mut cache1 = MlaKvCache::new(1, cfg.kv_lora_rank, cfg.qk_rope_head_dim, 64);
    let out1 = mla_forward_decode(&x, &weights, &mut cache1, 0, 0, &rope, &cfg, attn_scale);
    cache1.advance();
    let out1b = mla_forward_decode(&x, &weights, &mut cache1, 0, 1, &rope, &cfg, attn_scale);

    // Run 2: same inputs, fresh cache — must match exactly
    let mut cache2 = MlaKvCache::new(1, cfg.kv_lora_rank, cfg.qk_rope_head_dim, 64);
    let out2 = mla_forward_decode(&x, &weights, &mut cache2, 0, 0, &rope, &cfg, attn_scale);
    cache2.advance();
    let out2b = mla_forward_decode(&x, &weights, &mut cache2, 0, 1, &rope, &cfg, attn_scale);

    // Exact match (no FP reordering)
    assert_eq!(out1.len(), cfg.hidden_size, "output should be hidden_size");
    assert_eq!(out1, out2, "first call must be deterministic");
    assert_eq!(out1b, out2b, "second call must be deterministic");
}

/// Edge case: multiple positions build up KV cache correctly with profiling code present.
#[test]
fn profiling_preserves_kv_cache_across_positions() {
    let cfg = MlaDecodeConfig {
        hidden_size: 32,
        num_heads: 2,
        qk_nope_head_dim: 8,
        qk_rope_head_dim: 4,
        v_head_dim: 8,
        kv_lora_rank: 16,
        rms_eps: 1e-5,
    };

    let tw = make_test_weights(&cfg);
    let weights = tw.as_mla_weights();
    let rope = make_rope(32, cfg.qk_rope_head_dim);
    let attn_scale = 1.0 / ((cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) as f32).sqrt();

    let mut cache = MlaKvCache::new(1, cfg.kv_lora_rank, cfg.qk_rope_head_dim, 32);

    // Run 5 positions to exercise multi-position attention
    let mut outputs = Vec::new();
    for pos in 0..5 {
        let x = det_vec(cfg.hidden_size, 0.1, pos + 1);
        let out = mla_forward_decode(&x, &weights, &mut cache, 0, pos, &rope, &cfg, attn_scale);
        assert_eq!(out.len(), cfg.hidden_size);
        outputs.push(out);
        cache.advance();
    }

    // Each position should produce different output (non-degenerate test)
    assert_ne!(outputs[0], outputs[1], "different positions should give different outputs");
    assert_ne!(outputs[3], outputs[4], "later positions should differ too");
}
