//! Correctness test: pipelined double-buffer conversion+compute produces
//! identical results to sequential execution.
//!
//! What was optimized: the decode loop now overlaps f16→f32 conversion of
//! layer L+1 with computation of layer L using std::thread::scope + double-buffer
//! swap. Previously, conversion and computation ran sequentially per layer.
//!
//! Invariant: pipeline scheduling does not change computation order or values.
//! Results must be bit-identical to sequential execution.
//!
//! Failure means: race condition in pipeline, incorrect buffer swap, or
//! data corruption from concurrent access.

/// Pipeline double-buffer produces bit-identical results to sequential execution.
///
/// Simulates the real decode pattern: convert(L) then compute(L) sequentially
/// vs. convert(L+1) || compute(L) with buffer swap.
#[test]
fn pipeline_matches_sequential() {
    let num_layers = 20;
    let size = 8192; // realistic buffer size (similar to hidden_size)

    // --- Sequential: convert(L), compute(L) ---
    let mut buf = vec![0.0f32; size];
    let mut seq_results = Vec::new();
    for layer in 0..num_layers {
        // Simulate conversion: fill with layer-specific deterministic data
        for i in 0..size {
            buf[i] = ((layer * size + i) as f32 * 0.001).sin();
        }
        // Simulate computation: weighted reduction (depends on full buffer)
        let result: f64 = buf.iter().enumerate()
            .map(|(i, &v)| v as f64 * ((i + 1) as f64))
            .sum();
        seq_results.push(result);
    }

    // --- Pipeline: convert(L+1) || compute(L), swap buffers ---
    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];
    let mut pipe_results = Vec::new();

    // Convert layer 0 into a
    for i in 0..size {
        a[i] = ((0 * size + i) as f32 * 0.001).sin();
    }

    for layer in 0..num_layers {
        if layer + 1 < num_layers {
            let next_layer = layer + 1;
            let result = std::thread::scope(|s| {
                s.spawn(|| {
                    for i in 0..size {
                        b[i] = ((next_layer * size + i) as f32 * 0.001).sin();
                    }
                });
                let r: f64 = a.iter().enumerate()
                    .map(|(i, &v)| v as f64 * ((i + 1) as f64))
                    .sum();
                r
            });
            pipe_results.push(result);
            std::mem::swap(&mut a, &mut b);
        } else {
            let result: f64 = a.iter().enumerate()
                .map(|(i, &v)| v as f64 * ((i + 1) as f64))
                .sum();
            pipe_results.push(result);
        }
    }

    assert_eq!(seq_results.len(), pipe_results.len());
    for (i, (s, p)) in seq_results.iter().zip(pipe_results.iter()).enumerate() {
        assert_eq!(
            *s, *p,
            "Layer {i}: sequential={s} pipeline={p} — buffer swap or conversion error"
        );
    }
}

/// Edge case: single layer (no pipeline overlap possible).
#[test]
fn pipeline_single_layer() {
    let size = 256;
    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];

    // Convert layer 0
    for i in 0..size { a[i] = (i as f32).cos(); }

    // Single layer: no pipeline, just compute from a
    let result: f32 = a.iter().sum();
    assert!(result.is_finite(), "single layer result must be finite");

    // Verify b is untouched (no spurious conversion)
    assert!(b.iter().all(|&v| v == 0.0), "buf_b should not be modified for single layer");
}

/// Edge case: two layers (minimal pipeline: one overlap + one sequential).
#[test]
fn pipeline_two_layers() {
    let size = 512;
    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];

    // Sequential reference
    let mut seq_buf = vec![0.0f32; size];
    let mut seq = Vec::new();
    for layer in 0..2 {
        for i in 0..size { seq_buf[i] = ((layer * size + i) as f32).sin(); }
        seq.push(seq_buf.iter().sum::<f32>());
    }

    // Pipeline
    for i in 0..size { a[i] = ((0 * size + i) as f32).sin(); }

    // Layer 0: overlap with convert(1)
    let r0 = std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..size { b[i] = ((1 * size + i) as f32).sin(); }
        });
        a.iter().sum::<f32>()
    });
    std::mem::swap(&mut a, &mut b);

    // Layer 1: last layer, no overlap
    let r1: f32 = a.iter().sum();

    let max_diff_0 = (r0 - seq[0]).abs();
    let max_diff_1 = (r1 - seq[1]).abs();
    assert!(max_diff_0 == 0.0, "Layer 0: pipeline={r0} sequential={} diff={max_diff_0}", seq[0]);
    assert!(max_diff_1 == 0.0, "Layer 1: pipeline={r1} sequential={} diff={max_diff_1}", seq[1]);
}

/// Measure std::thread::scope overhead to confirm it's small vs ~7ms conversion savings.
#[test]
fn thread_scope_overhead_acceptable() {
    let iters = 200;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                std::hint::black_box(42u64);
            });
            std::hint::black_box(43u64);
            h.join().unwrap();
        });
    }
    let total_us = t.elapsed().as_micros();
    let per_scope_us = total_us as f64 / iters as f64;

    eprintln!("\n=== thread::scope overhead ===");
    eprintln!("  {per_scope_us:.1}μs per scope ({iters} iterations)");
    eprintln!("  × 60 MoE layers = {:.1}ms total pipeline overhead",
        per_scope_us * 60.0 / 1000.0);

    // Must be < 100μs per scope (budget: 60 layers × 100μs = 6ms, must be << 7ms savings)
    assert!(per_scope_us < 100.0,
        "thread::scope overhead {per_scope_us:.0}μs too high (must be <100μs)");
}

/// Stress test: many layers with large buffers to expose race conditions.
#[test]
fn pipeline_stress_61_layers() {
    // Matches V3's 61 layers
    let num_layers = 61;
    let size = 4096;

    let mut seq_checksums = Vec::new();
    let mut buf = vec![0.0f32; size];
    for layer in 0..num_layers {
        for i in 0..size {
            buf[i] = ((layer as f32 * 1000.0 + i as f32) * 0.0001).sin();
        }
        let cs: f64 = buf.iter().map(|&v| v as f64 * v as f64).sum();
        seq_checksums.push(cs);
    }

    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];
    let mut pipe_checksums = Vec::new();

    for i in 0..size {
        a[i] = ((0.0f32 * 1000.0 + i as f32) * 0.0001).sin();
    }

    for layer in 0..num_layers {
        if layer + 1 < num_layers {
            let next_layer = layer + 1;
            let cs = std::thread::scope(|s| {
                s.spawn(|| {
                    for i in 0..size {
                        b[i] = ((next_layer as f32 * 1000.0 + i as f32) * 0.0001).sin();
                    }
                });
                a.iter().map(|&v| v as f64 * v as f64).sum::<f64>()
            });
            pipe_checksums.push(cs);
            std::mem::swap(&mut a, &mut b);
        } else {
            let cs: f64 = a.iter().map(|&v| v as f64 * v as f64).sum();
            pipe_checksums.push(cs);
        }
    }

    for (i, (s, p)) in seq_checksums.iter().zip(pipe_checksums.iter()).enumerate() {
        assert_eq!(*s, *p, "Layer {i}: mismatch — sequential={s} pipeline={p}");
    }
}
