//! Correctness test: sgemm_nt attention scores match scalar f64 reference.
//!
//! What was optimized: replaced scalar f64 dot-product loops for attention scores
//! and value reconstruction with batched sgemm_nt/sgemm (AMX-optimized).
//! At seq_len≈16 (average during decode), scalar loops take ~1ms/layer (measured
//! by s1-mla-profiling). sgemm on L2-resident data: ~10µs.
//!
//! Invariant: sgemm f32 scores match f64 reference within tolerance 1e-3.
//! The switch from f64 to f32 accumulation introduces per-element error of
//! ~sqrt(k)*eps ≈ sqrt(512)*6e-8 ≈ 1.4e-6. After scaling+softmax, output
//! precision is dominated by exp() precision, not dot product precision.
//!
//! Failure means: sgemm_nt transposition is wrong, dimension mismatch, or
//! accumulation order causes error beyond tolerance.

/// Reference: scalar f64 dot product (original implementation).
fn scores_f64_ref(
    q_absorbed: &[f32], q_pe: &[f32],
    latent_cache: &[f32], rope_cache: &[f32],
    h: usize, seq_len: usize, kv_rank: usize, rope_dim: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; h * seq_len];
    for head in 0..h {
        let q_abs = &q_absorbed[head * kv_rank..(head + 1) * kv_rank];
        let q_rope = &q_pe[head * rope_dim..(head + 1) * rope_dim];
        for t in 0..seq_len {
            let lat_t = &latent_cache[t * kv_rank..(t + 1) * kv_rank];
            let rope_t = &rope_cache[t * rope_dim..(t + 1) * rope_dim];
            let mut dot_nope = 0.0f64;
            for d in 0..kv_rank { dot_nope += q_abs[d] as f64 * lat_t[d] as f64; }
            let mut dot_rope = 0.0f64;
            for d in 0..rope_dim { dot_rope += q_rope[d] as f64 * rope_t[d] as f64; }
            scores[head * seq_len + t] = (dot_nope + dot_rope) as f32 * attn_scale;
        }
    }
    scores
}

/// Optimized: batched sgemm_nt.
fn scores_sgemm(
    q_absorbed: &[f32], q_pe: &[f32],
    latent_cache: &[f32], rope_cache: &[f32],
    h: usize, seq_len: usize, kv_rank: usize, rope_dim: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; h * seq_len];
    let mut scores_rope = vec![0.0f32; h * seq_len];
    moe_infer::blas::sgemm_nt(q_absorbed, latent_cache, &mut scores, h, seq_len, kv_rank);
    moe_infer::blas::sgemm_nt(q_pe, rope_cache, &mut scores_rope, h, seq_len, rope_dim);
    for i in 0..h * seq_len {
        scores[i] = (scores[i] + scores_rope[i]) * attn_scale;
    }
    scores
}

/// Reference: per-head scalar value reconstruction.
fn value_recon_ref(
    attn_weights: &[f32], latent_cache: &[f32],
    h: usize, seq_len: usize, kv_rank: usize,
) -> Vec<f32> {
    let mut v_latents = vec![0.0f32; h * kv_rank];
    for head in 0..h {
        let w = &attn_weights[head * seq_len..(head + 1) * seq_len];
        let v_lat = &mut v_latents[head * kv_rank..(head + 1) * kv_rank];
        for t in 0..seq_len {
            let lat_t = &latent_cache[t * kv_rank..(t + 1) * kv_rank];
            let wt = w[t];
            for d in 0..kv_rank { v_lat[d] += wt * lat_t[d]; }
        }
    }
    v_latents
}

/// Optimized: batched sgemm for value reconstruction.
fn value_recon_sgemm(
    attn_weights: &[f32], latent_cache: &[f32],
    h: usize, seq_len: usize, kv_rank: usize,
) -> Vec<f32> {
    let mut v_latents = vec![0.0f32; h * kv_rank];
    moe_infer::blas::sgemm(attn_weights, latent_cache, &mut v_latents, h, kv_rank, seq_len);
    v_latents
}

fn make_data(seed: u64, len: usize) -> Vec<f32> {
    (0..len).map(|i| ((seed as f64 * 0.37 + i as f64 * 0.001).sin() * 0.5) as f32).collect()
}

/// V3-like dimensions: 128 heads, kv_rank=512, rope_dim=64.
#[test]
fn sgemm_scores_match_f64_v3() {
    let h = 128;
    let seq_len = 20;
    let kv_rank = 512;
    let rope_dim = 64;
    let attn_scale = 0.135f32;

    let q_abs = make_data(1, h * kv_rank);
    let q_pe = make_data(2, h * rope_dim);
    let lat_cache = make_data(3, seq_len * kv_rank);
    let rope_cache = make_data(4, seq_len * rope_dim);

    let ref_scores = scores_f64_ref(&q_abs, &q_pe, &lat_cache, &rope_cache, h, seq_len, kv_rank, rope_dim, attn_scale);
    let opt_scores = scores_sgemm(&q_abs, &q_pe, &lat_cache, &rope_cache, h, seq_len, kv_rank, rope_dim, attn_scale);

    let max_diff = ref_scores.iter().zip(opt_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // f64→f32 precision: sqrt(512)*eps ≈ 1.4e-6 per dot, ×attn_scale ≈ 2e-7
    assert!(max_diff < 1e-3, "Scores max diff {max_diff} exceeds 1e-3");
    eprintln!("V3 scores: max_diff={max_diff:.2e} (h={h}, seq={seq_len})");
}

/// Value reconstruction sgemm matches scalar reference.
#[test]
fn sgemm_value_recon_matches_scalar() {
    let h = 128;
    let seq_len = 20;
    let kv_rank = 512;

    let attn_w = make_data(5, h * seq_len);
    let lat_cache = make_data(3, seq_len * kv_rank);

    let ref_vals = value_recon_ref(&attn_w, &lat_cache, h, seq_len, kv_rank);
    let opt_vals = value_recon_sgemm(&attn_w, &lat_cache, h, seq_len, kv_rank);

    let max_diff = ref_vals.iter().zip(opt_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "Value recon max diff {max_diff} exceeds 1e-3");
    eprintln!("Value recon: max_diff={max_diff:.2e} (h={h}, seq={seq_len}, kv_rank={kv_rank})");
}

/// Edge case: seq_len=1 (single token, common at start of decode).
#[test]
fn sgemm_scores_seq_len_1() {
    let h = 128;
    let seq_len = 1;
    let kv_rank = 512;
    let rope_dim = 64;
    let attn_scale = 0.135f32;

    let q_abs = make_data(10, h * kv_rank);
    let q_pe = make_data(11, h * rope_dim);
    let lat_cache = make_data(12, seq_len * kv_rank);
    let rope_cache = make_data(13, seq_len * rope_dim);

    let ref_scores = scores_f64_ref(&q_abs, &q_pe, &lat_cache, &rope_cache, h, seq_len, kv_rank, rope_dim, attn_scale);
    let opt_scores = scores_sgemm(&q_abs, &q_pe, &lat_cache, &rope_cache, h, seq_len, kv_rank, rope_dim, attn_scale);

    let max_diff = ref_scores.iter().zip(opt_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "seq_len=1 max diff {max_diff} exceeds 1e-4");
}

/// Edge case: V2-Lite-like dimensions (smaller model).
#[test]
fn sgemm_scores_v2lite_dims() {
    let h = 16;
    let seq_len = 10;
    let kv_rank = 512;
    let rope_dim = 64;
    let attn_scale = 0.08f32;

    let q_abs = make_data(20, h * kv_rank);
    let q_pe = make_data(21, h * rope_dim);
    let lat_cache = make_data(22, seq_len * kv_rank);
    let rope_cache = make_data(23, seq_len * rope_dim);

    let ref_scores = scores_f64_ref(&q_abs, &q_pe, &lat_cache, &rope_cache, h, seq_len, kv_rank, rope_dim, attn_scale);
    let opt_scores = scores_sgemm(&q_abs, &q_pe, &lat_cache, &rope_cache, h, seq_len, kv_rank, rope_dim, attn_scale);

    let max_diff = ref_scores.iter().zip(opt_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "V2-Lite dims max diff {max_diff}");
}
