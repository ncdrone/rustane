//! Correctness test: sgemm_nt_add (beta=1.0) + in-place softmax matches
//! the original two-buffer implementation (sgemm_nt + separate add + separate softmax).
//!
//! What was optimized: eliminated scores_rope_buf allocation and attn_weights
//! allocation by using sgemm_nt_add to accumulate rope scores in-place, and
//! computing softmax in-place in the scores buffer.
//!
//! Invariant: fused scores+softmax match original within 1e-5 tolerance.
//! The sgemm_nt_add path uses identical BLAS operations (just beta=1.0 vs beta=0.0
//! followed by element-wise add), so results should be nearly identical (same
//! accumulation order within BLAS). In-place softmax is mathematically identical.
//!
//! Failure means: sgemm_nt_add has wrong beta semantics, or in-place softmax
//! has a buffer aliasing bug.

/// Original implementation: two separate buffers + addition loop + separate softmax.
fn scores_original(
    q_absorbed: &[f32], q_pe: &[f32],
    latent_cache: &[f32], rope_cache: &[f32],
    h: usize, seq_len: usize, kv_rank: usize, rope_dim: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; h * seq_len];
    let mut scores_rope_buf = vec![0.0f32; h * seq_len];
    moe_infer::blas::sgemm_nt(q_absorbed, latent_cache, &mut scores, h, seq_len, kv_rank);
    moe_infer::blas::sgemm_nt(q_pe, rope_cache, &mut scores_rope_buf, h, seq_len, rope_dim);
    for i in 0..h * seq_len {
        scores[i] = (scores[i] + scores_rope_buf[i]) * attn_scale;
    }

    // Softmax into separate buffer
    let mut attn_weights = vec![0.0f32; h * seq_len];
    for head in 0..h {
        let s = &scores[head * seq_len..(head + 1) * seq_len];
        let w = &mut attn_weights[head * seq_len..(head + 1) * seq_len];
        let max_s = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for t in 0..seq_len { w[t] = (s[t] - max_s).exp(); sum += w[t]; }
        for t in 0..seq_len { w[t] /= sum; }
    }
    attn_weights
}

/// Optimized: sgemm_nt_add (beta=1.0) + in-place softmax.
fn scores_fused(
    q_absorbed: &[f32], q_pe: &[f32],
    latent_cache: &[f32], rope_cache: &[f32],
    h: usize, seq_len: usize, kv_rank: usize, rope_dim: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; h * seq_len];
    moe_infer::blas::sgemm_nt(q_absorbed, latent_cache, &mut scores, h, seq_len, kv_rank);
    moe_infer::blas::sgemm_nt_add(q_pe, rope_cache, &mut scores, h, seq_len, rope_dim);

    // Scale + softmax in-place
    for head in 0..h {
        let s = &mut scores[head * seq_len..(head + 1) * seq_len];
        for t in 0..seq_len { s[t] *= attn_scale; }
        let max_s = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for t in 0..seq_len { s[t] = (s[t] - max_s).exp(); sum += s[t]; }
        for t in 0..seq_len { s[t] /= sum; }
    }
    scores
}

#[test]
fn fused_scores_match_original_k2_dims() {
    // K2 dimensions: 64 heads, kv_rank=512, rope_dim=64
    let h = 64;
    let seq_len = 10;
    let kv_rank = 512;
    let rope_dim = 64;
    let attn_scale = 0.006944f32; // typical K2 scale

    // Generate deterministic test data
    let q_absorbed: Vec<f32> = (0..h * kv_rank).map(|i| (i as f32 * 0.001).sin()).collect();
    let q_pe: Vec<f32> = (0..h * rope_dim).map(|i| (i as f32 * 0.01).cos()).collect();
    let latent_cache: Vec<f32> = (0..seq_len * kv_rank).map(|i| (i as f32 * 0.002).sin()).collect();
    let rope_cache: Vec<f32> = (0..seq_len * rope_dim).map(|i| (i as f32 * 0.02).cos()).collect();

    let original = scores_original(&q_absorbed, &q_pe, &latent_cache, &rope_cache,
        h, seq_len, kv_rank, rope_dim, attn_scale);
    let fused = scores_fused(&q_absorbed, &q_pe, &latent_cache, &rope_cache,
        h, seq_len, kv_rank, rope_dim, attn_scale);

    assert_eq!(original.len(), fused.len());
    let max_diff = original.iter().zip(fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 1e-5,
        "fused vs original max_diff={max_diff} (expected < 1e-5)");
    eprintln!("K2 dims (h=64, seq=10): max_diff={max_diff:.2e}");
}

#[test]
fn fused_scores_match_original_seq1() {
    // Edge case: seq_len=1 (first decode token)
    let h = 64;
    let seq_len = 1;
    let kv_rank = 512;
    let rope_dim = 64;
    let attn_scale = 0.006944f32;

    let q_absorbed: Vec<f32> = (0..h * kv_rank).map(|i| (i as f32 * 0.003).sin()).collect();
    let q_pe: Vec<f32> = (0..h * rope_dim).map(|i| (i as f32 * 0.03).cos()).collect();
    let latent_cache: Vec<f32> = (0..seq_len * kv_rank).map(|i| (i as f32 * 0.004).sin()).collect();
    let rope_cache: Vec<f32> = (0..seq_len * rope_dim).map(|i| (i as f32 * 0.04).cos()).collect();

    let original = scores_original(&q_absorbed, &q_pe, &latent_cache, &rope_cache,
        h, seq_len, kv_rank, rope_dim, attn_scale);
    let fused = scores_fused(&q_absorbed, &q_pe, &latent_cache, &rope_cache,
        h, seq_len, kv_rank, rope_dim, attn_scale);

    let max_diff = original.iter().zip(fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // At seq_len=1, softmax output is always 1.0 for both paths
    assert!(max_diff < 1e-6,
        "seq=1 fused vs original max_diff={max_diff} (expected < 1e-6)");
    eprintln!("seq_len=1: max_diff={max_diff:.2e}, weights[0]={:.6}", fused[0]);
}

#[test]
fn sgemm_nt_add_accumulates_correctly() {
    // Direct test of sgemm_nt_add: verify beta=1.0 semantics
    let m = 4;
    let n = 3;
    let k = 5;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.2).cos()).collect();

    // Compute A @ B^T with beta=0
    let mut c_fresh = vec![0.0f32; m * n];
    moe_infer::blas::sgemm_nt(&a, &b, &mut c_fresh, m, n, k);

    // Compute into pre-existing values with beta=1
    let bias: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.5).collect();
    let mut c_accum = bias.clone();
    moe_infer::blas::sgemm_nt_add(&a, &b, &mut c_accum, m, n, k);

    // Verify: c_accum[i] = c_fresh[i] + bias[i]
    let max_diff = c_accum.iter().zip(c_fresh.iter()).zip(bias.iter())
        .map(|((acc, fresh), b)| (acc - fresh - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 1e-6,
        "sgemm_nt_add accumulation error: max_diff={max_diff}");
    eprintln!("sgemm_nt_add: max_diff={max_diff:.2e}");
}

#[test]
fn fused_scores_large_seq() {
    // Larger seq_len to test scaling behavior
    let h = 8; // smaller head count for speed
    let seq_len = 100;
    let kv_rank = 128;
    let rope_dim = 64;
    let attn_scale = 0.01f32;

    let q_absorbed: Vec<f32> = (0..h * kv_rank).map(|i| (i as f32 * 0.005).sin()).collect();
    let q_pe: Vec<f32> = (0..h * rope_dim).map(|i| (i as f32 * 0.05).cos()).collect();
    let latent_cache: Vec<f32> = (0..seq_len * kv_rank).map(|i| (i as f32 * 0.001).sin()).collect();
    let rope_cache: Vec<f32> = (0..seq_len * rope_dim).map(|i| (i as f32 * 0.01).cos()).collect();

    let original = scores_original(&q_absorbed, &q_pe, &latent_cache, &rope_cache,
        h, seq_len, kv_rank, rope_dim, attn_scale);
    let fused = scores_fused(&q_absorbed, &q_pe, &latent_cache, &rope_cache,
        h, seq_len, kv_rank, rope_dim, attn_scale);

    let max_diff = original.iter().zip(fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 1e-5,
        "large seq fused vs original max_diff={max_diff}");
    eprintln!("Large seq (h=8, seq=100): max_diff={max_diff:.2e}");
}
