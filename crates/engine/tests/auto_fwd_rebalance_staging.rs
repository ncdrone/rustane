//! Test: fwd-rebalance-staging — move ffnFused weight staging from woFwd overlap to sdpaFwd overlap.
//!
//! What was optimized: In forward_into, ffnFused weight staging was moved from step 5
//! (overlapped with woFwd ANE, CPU-bound at ~1.2ms > woFwd ~0.5ms) to step 3 (overlapped
//! with sdpaFwd ANE, which has ~2ms providing ample CPU headroom). This eliminates the
//! CPU bottleneck in step 5 without increasing step 3 time.
//!
//! Invariant: forward_into must produce identical output regardless of WHEN weights are
//! staged to IOSurface. The staging order is a scheduling optimization that must not affect
//! numerical results.
//!
//! What a failure means: The IOSurface buffer for ffnFused is being written to at the wrong
//! time, or the woFwd/ffnFused kernel inputs are corrupted by concurrent IOSurface access.
//!
//! Run: cargo test -p engine --test auto_fwd_rebalance_staging --release -- --nocapture

use engine::layer::{self, CompiledKernels, ForwardCache, LayerWeights};
use engine::model::ModelConfig;

/// Forward pass output must be deterministic across multiple calls with same input.
/// This catches any race conditions or stale-data issues from the staging reorder.
#[test]
fn forward_deterministic() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let dim = cfg.dim;
    let seq = cfg.seq;
    let x: Vec<f32> = (0..dim * seq)
        .map(|i| ((i * 17 + 3) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    // Run forward twice with same input
    let mut cache1 = ForwardCache::new(&cfg);
    let mut x_next1 = vec![0.0f32; dim * seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache1, &mut x_next1);

    let mut cache2 = ForwardCache::new(&cfg);
    let mut x_next2 = vec![0.0f32; dim * seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache2, &mut x_next2);

    // Must be bit-exact (same staging, same ANE, same math)
    let max_diff: f32 = x_next1
        .iter()
        .zip(x_next2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Determinism: max_diff = {:.2e}", max_diff);
    assert!(
        max_diff == 0.0,
        "Forward pass not deterministic: max_diff = {:.2e}",
        max_diff
    );

    // Also verify cache intermediates match
    let cache_max_diff: f32 = cache1
        .h1
        .iter()
        .zip(cache2.h1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Cache h1 determinism: max_diff = {:.2e}", cache_max_diff);
    assert!(cache_max_diff == 0.0, "Cache h1 not deterministic");
}

/// Forward output matches between forward_into (workspace) and forward (allocating).
/// This confirms the staging reorder didn't change numerical results.
#[test]
fn forward_into_matches_forward() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let dim = cfg.dim;
    let seq = cfg.seq;
    let x: Vec<f32> = (0..dim * seq)
        .map(|i| ((i * 31 + 7) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    // Allocating path (reference)
    let (x_next_ref, _cache_ref) = layer::forward(&cfg, &kernels, &weights, &x);

    // Workspace path (under test)
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; dim * seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    // Should match within ANE fp16 round-trip tolerance
    let max_diff: f32 = x_next
        .iter()
        .zip(x_next_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("forward_into vs forward: max_diff = {:.2e}", max_diff);
    assert!(
        max_diff < 1e-5,
        "Outputs diverge: max_diff = {:.2e} (expected < 1e-5)",
        max_diff
    );
}

/// Back-to-back forward calls reusing the same ForwardCache and KernelBuffers.
/// Catches stale IOSurface data from weight pre-staging written at wrong offset.
#[test]
fn back_to_back_with_different_weights() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);

    let dim = cfg.dim;
    let seq = cfg.seq;
    let x: Vec<f32> = (0..dim * seq)
        .map(|i| ((i * 17 + 3) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    // Two different weight sets (simulates different layers sharing kernel buffers)
    let weights_a = LayerWeights::random(&cfg);
    let mut weights_b = LayerWeights::random(&cfg);
    // Make weights_b different — must change wo/w2 (not zero-init'd copies)
    // because DeepNet zero-init makes output independent of wq/wk/wv/w1/w3 when wo=w2=0
    for (i, x) in weights_b.wo.iter_mut().enumerate() {
        *x = 0.01 * ((i * 13 + 7) % 100) as f32 * 0.01;
    }
    for (i, x) in weights_b.w2.iter_mut().enumerate() {
        *x = 0.01 * ((i * 17 + 3) % 100) as f32 * 0.01;
    }

    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; dim * seq];

    // Run with weights_a
    layer::forward_into(&cfg, &kernels, &weights_a, &x, &mut cache, &mut x_next);
    let x_next_a = x_next.clone();

    // Run with weights_b (different weights, same IOSurface buffers)
    layer::forward_into(&cfg, &kernels, &weights_b, &x, &mut cache, &mut x_next);
    let x_next_b = x_next.clone();

    // Run with weights_a again — must match first run
    layer::forward_into(&cfg, &kernels, &weights_a, &x, &mut cache, &mut x_next);
    let x_next_a2 = x_next.clone();

    // Verify weights_a results match across both runs
    let max_diff: f32 = x_next_a
        .iter()
        .zip(x_next_a2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("weights_a run1 vs run3: max_diff = {:.2e}", max_diff);
    assert!(
        max_diff == 0.0,
        "Stale IOSurface data: weights_a results differ after weights_b run (max_diff={:.2e})",
        max_diff
    );

    // Verify weights_a and weights_b produce different results
    let ab_diff: f32 = x_next_a
        .iter()
        .zip(x_next_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("weights_a vs weights_b: max_diff = {:.2e}", ab_diff);
    assert!(
        ab_diff > 0.01,
        "Different weights should produce different outputs"
    );
}
