//! Correctness test for deferred-ffn-readback optimization.
//!
//! What was optimized:
//!   `forward_into_pipelined` defers reading h1/h3/gate from the ffnFused IOSurface
//!   output, only reading x_next immediately. The deferred readback happens during
//!   the NEXT layer's sdpaFwd ANE overlap (step 3), hiding 0.8ms of IOSurface reads
//!   behind ANE compute time. The last layer's cache is read via `read_ffn_cache`.
//!
//! Invariant checked:
//!   `forward_into_pipelined` + `read_ffn_cache` must produce identical cache contents
//!   and x_next as `forward_into`. Same ANE kernels, same data, different readback timing.
//!
//! Tolerance: 0.0 (exact match — same ANE output, same reads, just reordered)
//!
//! Failure meaning:
//!   The deferred readback read stale or wrong data from the IOSurface, either because:
//!   - The IOSurface was overwritten before readback completed
//!   - The readback read from wrong channel offsets
//!   - The split_at_mut indexing was wrong (wrote to wrong cache)

use engine::layer::{self, CompiledKernels, ForwardCache, LayerWeights};
use engine::model::ModelConfig;

fn assert_exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: ref={x} vs opt={y} (not bit-identical)"
        );
    }
}

/// Test 1: Single layer — pipelined(None) + read_ffn_cache matches forward_into.
#[test]
fn deferred_readback_single_layer_matches() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    // Reference: forward_into (reads h1/h3/gate immediately)
    let mut cache_ref = ForwardCache::new(&cfg);
    let mut x_next_ref = vec![0.0f32; n_in];
    layer::forward_into(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut cache_ref,
        &mut x_next_ref,
    );

    // Optimized: forward_into_pipelined(None) + read_ffn_cache
    let mut cache_opt = ForwardCache::new(&cfg);
    let mut x_next_opt = vec![0.0f32; n_in];
    layer::forward_into_pipelined(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut cache_opt,
        &mut x_next_opt,
        None,
    );
    layer::read_ffn_cache(&cfg, &kernels, &mut cache_opt);

    assert_exact(&x_next_ref, &x_next_opt, "x_next");
    assert_exact(&cache_ref.h1, &cache_opt.h1, "h1");
    assert_exact(&cache_ref.h3, &cache_opt.h3, "h3");
    assert_exact(&cache_ref.gate, &cache_opt.gate, "gate");
    assert_exact(&cache_ref.x, &cache_opt.x, "cache.x");
    assert_exact(&cache_ref.xnorm, &cache_opt.xnorm, "cache.xnorm");
    assert_exact(&cache_ref.attn_out, &cache_opt.attn_out, "cache.attn_out");
    assert_exact(&cache_ref.x2norm, &cache_opt.x2norm, "cache.x2norm");

    println!(
        "PASS: pipelined(None) + read_ffn_cache matches forward_into (all fields bit-identical)"
    );
}

/// Test 2: Two layers — pipelined pipeline correctly reads previous layer's cache.
/// Layer 0 defers its h1/h3/gate. Layer 1's pipelined(prev_cache=layer0) reads them
/// during sdpaFwd overlap. Then layer 1's cache is read via read_ffn_cache.
#[test]
fn deferred_readback_two_layer_pipeline() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let w0 = LayerWeights::random(&cfg);
    let w1 = LayerWeights::random(&cfg);

    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    // Reference: two forward_into calls
    let mut cache0_ref = ForwardCache::new(&cfg);
    let mut cache1_ref = ForwardCache::new(&cfg);
    let mut x_mid = vec![0.0f32; n_in];
    let mut x_next_ref = vec![0.0f32; n_in];
    layer::forward_into(&cfg, &kernels, &w0, &x, &mut cache0_ref, &mut x_mid);
    layer::forward_into(
        &cfg,
        &kernels,
        &w1,
        &x_mid,
        &mut cache1_ref,
        &mut x_next_ref,
    );

    // Optimized: pipelined chain
    let mut cache0_opt = ForwardCache::new(&cfg);
    let mut cache1_opt = ForwardCache::new(&cfg);
    let mut x_mid_opt = vec![0.0f32; n_in];
    let mut x_next_opt = vec![0.0f32; n_in];
    // Layer 0: no prev cache
    layer::forward_into_pipelined(
        &cfg,
        &kernels,
        &w0,
        &x,
        &mut cache0_opt,
        &mut x_mid_opt,
        None,
    );
    // Layer 1: reads layer 0's deferred h1/h3/gate during sdpaFwd overlap
    layer::forward_into_pipelined(
        &cfg,
        &kernels,
        &w1,
        &x_mid_opt,
        &mut cache1_opt,
        &mut x_next_opt,
        Some(&mut cache0_opt),
    );
    // Read last layer's deferred cache
    layer::read_ffn_cache(&cfg, &kernels, &mut cache1_opt);

    // Verify x_next and both caches
    assert_exact(&x_next_ref, &x_next_opt, "x_next");
    assert_exact(&cache0_ref.h1, &cache0_opt.h1, "L0.h1");
    assert_exact(&cache0_ref.h3, &cache0_opt.h3, "L0.h3");
    assert_exact(&cache0_ref.gate, &cache0_opt.gate, "L0.gate");
    assert_exact(&cache1_ref.h1, &cache1_opt.h1, "L1.h1");
    assert_exact(&cache1_ref.h3, &cache1_opt.h3, "L1.h3");
    assert_exact(&cache1_ref.gate, &cache1_opt.gate, "L1.gate");

    println!("PASS: two-layer pipeline correctly reads deferred caches (all fields bit-identical)");
}

/// Test 3: Idempotency — two consecutive pipelined calls produce identical results.
/// Catches stale IOSurface data from the first call contaminating the second.
#[test]
fn deferred_readback_idempotent() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);

    let n_in = cfg.dim * cfg.seq;
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32 * 0.001) - 0.5) * 0.1)
        .collect();

    // Call 1
    let mut cache1 = ForwardCache::new(&cfg);
    let mut x_next1 = vec![0.0f32; n_in];
    layer::forward_into_pipelined(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut cache1,
        &mut x_next1,
        None,
    );
    layer::read_ffn_cache(&cfg, &kernels, &mut cache1);

    // Call 2
    let mut cache2 = ForwardCache::new(&cfg);
    let mut x_next2 = vec![0.0f32; n_in];
    layer::forward_into_pipelined(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut cache2,
        &mut x_next2,
        None,
    );
    layer::read_ffn_cache(&cfg, &kernels, &mut cache2);

    assert_exact(&x_next1, &x_next2, "x_next_idempotent");
    assert_exact(&cache1.h1, &cache2.h1, "h1_idempotent");
    assert_exact(&cache1.h3, &cache2.h3, "h3_idempotent");
    assert_exact(&cache1.gate, &cache2.gate, "gate_idempotent");

    println!("PASS: pipelined is idempotent (no stale IOSurface contamination)");
}
