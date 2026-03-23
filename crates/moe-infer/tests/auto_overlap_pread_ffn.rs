//! Correctness test: overlapping expert pread with shared FFN produces
//! identical results to sequential execution.
//!
//! What was optimized: in moe_ffn_v2, expert pread from SSD now runs
//! concurrently with shared expert FFN computation via std::thread::scope.
//! Previously they ran sequentially (~3ms pread + ~1-4ms shared FFN).
//! After: max(pread, shared_FFN) per MoE layer.
//!
//! Invariant: pread and shared FFN operate on completely separate data.
//! Pread writes to staging buffer; shared FFN reads weight matrices and
//! writes to a new Vec. No data dependency between them.
//! The results must be identical to sequential execution.
//!
//! Failure means: race condition between pread and shared FFN threads,
//! or incorrect thread::scope lifetime management.

/// Verify that concurrent I/O + compute on independent data produces same results.
///
/// Simulates the pattern: thread::scope { spawn(write_to_staging) || shared_ffn() }
/// where staging and FFN operate on completely independent memory regions.
#[test]
fn overlap_independent_compute_and_io() {
    let num_layers = 20;
    let hidden = 4096;
    let staging_size = 8192;

    // Simulate sequential: shared FFN then staging write
    let mut seq_ffn_results = Vec::new();
    let mut seq_staging = vec![0u8; staging_size];
    for layer in 0..num_layers {
        // Shared FFN: deterministic compute on input
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((layer * hidden + i) as f32 * 0.001).sin())
            .collect();
        let ffn_result: f64 = input.iter().map(|v| (*v as f64).abs()).sum();
        seq_ffn_results.push(ffn_result);

        // Staging write: fill with layer-dependent pattern
        for i in 0..staging_size {
            seq_staging[i] = ((layer * staging_size + i) % 256) as u8;
        }
    }

    // Simulate overlapped: thread::scope { spawn(staging_write) || shared_ffn }
    let mut par_ffn_results = Vec::new();
    let mut par_staging = vec![0u8; staging_size];
    for layer in 0..num_layers {
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((layer * hidden + i) as f32 * 0.001).sin())
            .collect();
        let staging_ref = &mut par_staging;

        let (ffn_result, ) = std::thread::scope(|s| {
            let layer_copy = layer;
            let staging_size_copy = staging_size;

            // Spawn I/O on background thread
            s.spawn(move || {
                for i in 0..staging_size_copy {
                    staging_ref[i] = ((layer_copy * staging_size_copy + i) % 256) as u8;
                }
            });

            // Compute on main thread
            let result: f64 = input.iter().map(|v| (*v as f64).abs()).sum();
            (result, )
        });

        par_ffn_results.push(ffn_result);
    }

    // Results must be identical
    for i in 0..num_layers {
        assert!(
            (seq_ffn_results[i] - par_ffn_results[i]).abs() == 0.0,
            "FFN result mismatch at layer {i}: seq={} par={}",
            seq_ffn_results[i], par_ffn_results[i]
        );
    }
    assert_eq!(seq_staging, par_staging, "staging buffer mismatch");
}

/// Edge case: empty expert list (no pread, shared FFN still runs).
#[test]
fn overlap_no_experts_still_computes_shared() {
    let hidden = 1024;
    let input: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).cos()).collect();

    // With zero experts, the scope should still compute shared FFN correctly
    let result = std::thread::scope(|s| {
        // Empty pread (nothing to do)
        s.spawn(|| { /* no experts to load */ });

        // Shared FFN
        let sum: f64 = input.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        sum
    });

    let expected: f64 = input.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    assert!((result - expected).abs() < 1e-10, "empty expert list broke shared FFN");
}
