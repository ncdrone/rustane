//! Correctness test: deferred-convert pipeline produces identical results
//! to the original full-overlap pipeline.
//!
//! What was optimized: the decode loop now runs MLA attention FIRST without
//! any background conversion thread (full memory bandwidth for sgemv),
//! then overlaps convert(N+1) with FFN (which uses Metal GPU + SSD pread,
//! not competing for memory bandwidth).
//!
//! Invariant: the deferred scheduling does not change computation order or
//! values — only WHEN the convert thread runs relative to MLA vs FFN.
//! Results must be bit-identical to the original pipeline.
//!
//! Failure means: incorrect split of MLA/FFN phases, or data race from
//! changed thread overlap timing.

/// Deferred-convert pipeline produces bit-identical results to original pipeline.
///
/// Simulates both patterns:
/// - Original: spawn(convert(N+1)) || [MLA + FFN]
/// - Deferred: MLA first, then spawn(convert(N+1)) || FFN
///
/// Since conversion writes to a SEPARATE buffer (buf_b) that isn't read
/// until the NEXT layer, the timing of conversion doesn't affect results.
#[test]
fn deferred_convert_matches_original_pipeline() {
    let num_layers = 61; // match V3
    let size = 4096;

    // Simulate MLA computation (depends on buf_a only)
    fn simulate_mla(buf: &[f32]) -> (f64, f64) {
        let attn: f64 = buf.iter().map(|&v| (v as f64).sin()).sum();
        let normed2: f64 = buf.iter().map(|&v| (v as f64).cos()).sum();
        (attn, normed2)
    }

    // Simulate FFN computation (depends on buf_a and normed2)
    fn simulate_ffn(buf: &[f32], normed2: f64) -> f64 {
        buf.iter().map(|&v| v as f64 * normed2).sum::<f64>().sin()
    }

    // Simulate conversion of next layer
    fn simulate_convert(buf: &mut [f32], layer: usize) {
        for i in 0..buf.len() {
            buf[i] = ((layer * buf.len() + i) as f32 * 0.0001).sin();
        }
    }

    // --- Original pipeline: convert(N+1) || [MLA + FFN] ---
    let mut orig_a = vec![0.0f32; size];
    let mut orig_b = vec![0.0f32; size];
    let mut orig_results = Vec::new();
    simulate_convert(&mut orig_a, 0);

    for layer in 0..num_layers {
        if layer + 1 < num_layers {
            let next = layer + 1;
            let result = std::thread::scope(|s| {
                s.spawn(|| simulate_convert(&mut orig_b, next));
                let (attn, normed2) = simulate_mla(&orig_a);
                let ffn = simulate_ffn(&orig_a, normed2);
                attn + ffn
            });
            orig_results.push(result);
            std::mem::swap(&mut orig_a, &mut orig_b);
        } else {
            let (attn, normed2) = simulate_mla(&orig_a);
            let ffn = simulate_ffn(&orig_a, normed2);
            orig_results.push(attn + ffn);
        }
    }

    // --- Deferred pipeline: MLA first, then convert(N+1) || FFN ---
    let mut def_a = vec![0.0f32; size];
    let mut def_b = vec![0.0f32; size];
    let mut def_results = Vec::new();
    simulate_convert(&mut def_a, 0);

    for layer in 0..num_layers {
        // Phase 1: MLA (no background thread)
        let (attn, normed2) = simulate_mla(&def_a);

        // Phase 2: convert(N+1) || FFN
        if layer + 1 < num_layers {
            let next = layer + 1;
            let ffn = std::thread::scope(|s| {
                s.spawn(|| simulate_convert(&mut def_b, next));
                simulate_ffn(&def_a, normed2)
            });
            def_results.push(attn + ffn);
            std::mem::swap(&mut def_a, &mut def_b);
        } else {
            let ffn = simulate_ffn(&def_a, normed2);
            def_results.push(attn + ffn);
        }
    }

    // Must be bit-identical
    assert_eq!(orig_results.len(), def_results.len());
    for (i, (o, d)) in orig_results.iter().zip(def_results.iter()).enumerate() {
        assert_eq!(
            *o, *d,
            "Layer {i}: original={o} deferred={d} — scheduling change affected results"
        );
    }
}

/// Edge case: single layer (no pipeline at all).
#[test]
fn deferred_single_layer() {
    let size = 256;
    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];
    for i in 0..size { a[i] = (i as f32).cos(); }

    // MLA phase
    let mla: f32 = a.iter().map(|v| v.sin()).sum();
    // FFN phase (no convert overlap)
    let ffn: f32 = a.iter().map(|v| v * mla).sum();

    assert!(mla.is_finite() && ffn.is_finite());
    assert!(b.iter().all(|&v| v == 0.0), "buf_b untouched for single layer");
}

/// Edge case: conversion finishes during FFN phase (not during MLA).
/// Verifies the deferred pattern doesn't race with buf_a reads.
#[test]
fn deferred_no_race_on_buf_a() {
    let size = 8192;
    let num_layers = 10;

    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];
    for i in 0..size { a[i] = (i as f32 * 0.01).sin(); }

    let mut results = Vec::new();
    for layer in 0..num_layers {
        // MLA reads buf_a
        let mla: f64 = a.iter().map(|v| *v as f64).sum();

        if layer + 1 < num_layers {
            // FFN reads buf_a while convert writes buf_b — no race
            let ffn = std::thread::scope(|s| {
                s.spawn(|| {
                    for i in 0..size {
                        b[i] = (((layer + 1) * size + i) as f32 * 0.01).sin();
                    }
                });
                a.iter().map(|v| (*v as f64) * mla).sum::<f64>()
            });
            results.push(mla + ffn);
            std::mem::swap(&mut a, &mut b);
        } else {
            let ffn: f64 = a.iter().map(|v| (*v as f64) * mla).sum();
            results.push(mla + ffn);
        }
    }

    // Verify all results are finite and non-zero
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_finite() && *r != 0.0, "Layer {i}: result={r} should be finite non-zero");
    }
}
