//! Correctness test: caching dense layers (0..num_dense) in the decode pipeline
//! produces identical results to the original serial-convert + full-pipeline path.
//!
//! What was optimized: dense layers have large FFN weights ([18432,7168] × 3),
//! making their f16→f32 conversion ~10ms — longer than FFN compute (~7ms).
//! This caused a serial convert(0) stall (~10ms) and pipeline overflow for
//! dense-to-dense transitions (~3ms × 2). Caching eliminates both.
//!
//! Invariant: same layer weights → same compute results, regardless of whether
//! weights come from a cached buffer or a dynamically converted buffer.
//!
//! Failure means: wrong buffer used for a layer, pipeline shift error after
//! skipping cached layers, or buffer swap bug in the transition from cached
//! to dynamic layers.

/// Simulate the layer compute: deterministic function of layer weights.
fn simulate_compute(weights: &[f32]) -> f64 {
    weights.iter().enumerate()
        .map(|(i, &v)| v as f64 * ((i + 1) as f64))
        .sum()
}

/// Fill buffer with deterministic layer-specific data (simulates f16→f32 conversion).
fn fill_layer(buf: &mut [f32], layer: usize) {
    let size = buf.len();
    for i in 0..size {
        buf[i] = ((layer * size + i) as f32 * 0.001).sin();
    }
}

/// Original pipeline: serial convert(0), then convert(L+1) || compute(L) for all layers.
#[test]
fn cached_dense_matches_original_pipeline() {
    let num_layers = 20;
    let num_dense = 3; // matches V3 first_k_dense_replace
    let size = 4096;

    // --- Original pipeline: serial convert(0) + full pipeline ---
    let mut buf_a = vec![0.0f32; size];
    let mut buf_b = vec![0.0f32; size];
    let mut orig_results = Vec::new();

    fill_layer(&mut buf_a, 0); // serial convert(0)
    for layer in 0..num_layers {
        let result = simulate_compute(&buf_a);
        orig_results.push(result);
        if layer + 1 < num_layers {
            let next = layer + 1;
            std::thread::scope(|s| {
                s.spawn(|| fill_layer(&mut buf_b, next));
                std::hint::black_box(&buf_a); // simulate FFN work
            });
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
    }

    // --- Cached dense pipeline: cache layers 0..num_dense, skip convert for those ---
    let mut cached: Vec<Vec<f32>> = (0..num_dense).map(|layer| {
        let mut buf = vec![0.0f32; size];
        fill_layer(&mut buf, layer);
        buf
    }).collect();
    let mut buf_a2 = vec![0.0f32; size];
    let mut buf_b2 = vec![0.0f32; size];
    let mut cached_results = Vec::new();

    for layer in 0..num_layers {
        let lf = if layer < num_dense { &cached[layer] } else { &buf_a2 };
        let result = simulate_compute(lf);
        cached_results.push(result);

        if layer + 1 < num_layers {
            let next = layer + 1;
            if next < num_dense {
                // Next is cached — no background convert
                std::hint::black_box(lf); // simulate FFN
            } else {
                // Pipeline: convert(next) || FFN
                std::thread::scope(|s| {
                    s.spawn(|| fill_layer(&mut buf_b2, next));
                    std::hint::black_box(lf);
                });
                std::mem::swap(&mut buf_a2, &mut buf_b2);
            }
        }
    }

    assert_eq!(orig_results.len(), cached_results.len());
    for (i, (o, c)) in orig_results.iter().zip(cached_results.iter()).enumerate() {
        assert_eq!(*o, *c,
            "Layer {i}: original={o} cached={c} — buffer mismatch");
    }
}

/// Edge case: num_dense=0 (no caching, pure pipeline — should match original exactly).
#[test]
fn cached_dense_zero_cached_matches() {
    let num_layers = 10;
    let num_dense = 0;
    let size = 1024;

    let mut buf_a = vec![0.0f32; size];
    let mut buf_b = vec![0.0f32; size];
    let mut orig = Vec::new();

    fill_layer(&mut buf_a, 0);
    for layer in 0..num_layers {
        orig.push(simulate_compute(&buf_a));
        if layer + 1 < num_layers {
            std::thread::scope(|s| {
                s.spawn(|| fill_layer(&mut buf_b, layer + 1));
            });
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
    }

    // Same code path with num_dense=0 (no cached branch ever taken)
    let mut buf_a2 = vec![0.0f32; size];
    let mut buf_b2 = vec![0.0f32; size];
    let mut cached = Vec::new();

    // No serial convert needed — pipeline starts with convert(0) on first iteration
    // But to match original, we need buf_a2 initialized with layer 0
    fill_layer(&mut buf_a2, 0);
    for layer in 0..num_layers {
        let lf = &buf_a2; // num_dense=0, always dynamic
        cached.push(simulate_compute(lf));
        if layer + 1 < num_layers {
            std::thread::scope(|s| {
                s.spawn(|| fill_layer(&mut buf_b2, layer + 1));
            });
            std::mem::swap(&mut buf_a2, &mut buf_b2);
        }
    }

    for (i, (o, c)) in orig.iter().zip(cached.iter()).enumerate() {
        assert_eq!(*o, *c, "Layer {i}: mismatch with num_dense=0");
    }
}

/// Edge case: all layers cached (num_dense == num_layers).
#[test]
fn cached_dense_all_cached() {
    let num_layers = 5;
    let num_dense = 5;
    let size = 512;

    // Reference: sequential
    let mut buf = vec![0.0f32; size];
    let mut seq = Vec::new();
    for layer in 0..num_layers {
        fill_layer(&mut buf, layer);
        seq.push(simulate_compute(&buf));
    }

    // Cached: all layers pre-cached, no pipeline at all
    let cached: Vec<Vec<f32>> = (0..num_dense).map(|layer| {
        let mut b = vec![0.0f32; size];
        fill_layer(&mut b, layer);
        b
    }).collect();

    for (i, s) in seq.iter().enumerate() {
        let c = simulate_compute(&cached[i]);
        assert_eq!(*s, c, "Layer {i}: all-cached mismatch");
    }
}

/// Verify the transition from cached to dynamic layers uses correct buffer.
/// This is the critical edge: layer num_dense-1 (last cached) must pipeline
/// convert(num_dense) into buf_b, and layer num_dense must read from buf_a
/// (swapped from buf_b).
#[test]
fn cached_dense_transition_correctness() {
    let num_layers = 10;
    let num_dense = 3;
    let size = 2048;

    // Build reference results sequentially
    let mut ref_results = Vec::new();
    let mut buf = vec![0.0f32; size];
    for layer in 0..num_layers {
        fill_layer(&mut buf, layer);
        ref_results.push(simulate_compute(&buf));
    }

    // Cached dense pipeline
    let cached: Vec<Vec<f32>> = (0..num_dense).map(|layer| {
        let mut b = vec![0.0f32; size];
        fill_layer(&mut b, layer);
        b
    }).collect();
    let mut buf_a = vec![0.0f32; size];
    let mut buf_b = vec![0.0f32; size];
    let mut results = Vec::new();

    for layer in 0..num_layers {
        let lf = if layer < num_dense { &cached[layer] } else { &buf_a };
        results.push(simulate_compute(lf));

        if layer + 1 < num_layers {
            let next = layer + 1;
            if next < num_dense {
                // cached, no convert
            } else {
                std::thread::scope(|s| {
                    s.spawn(|| fill_layer(&mut buf_b, next));
                });
                std::mem::swap(&mut buf_a, &mut buf_b);
            }
        }
    }

    // Check transition layers specifically
    for layer in 0..num_layers {
        assert_eq!(ref_results[layer], results[layer],
            "Layer {layer} mismatch (transition at layer {num_dense})");
    }
    eprintln!("Transition from cached (layer {}) to dynamic (layer {num_dense}) verified.",
        num_dense - 1);
}
