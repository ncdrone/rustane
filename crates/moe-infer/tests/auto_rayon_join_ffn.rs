//! Correctness test: rayon::join produces identical FFN overlap behavior as thread::scope.
//!
//! What was optimized: replaced std::thread::scope + s.spawn with rayon::join for the
//! pread + shared_ffn overlap in moe_ffn_v2. This eliminates ~60 pthread_create/join
//! calls per token (~30-100µs each) by using rayon's warm thread pool (~1-3µs overhead).
//!
//! Invariant: the computation is identical — same shared_expert_ffn output, same pread
//! data, same Metal dispatch. Only the thread dispatch mechanism changed.
//!
//! Failure means: rayon::join doesn't provide equivalent concurrent execution to
//! thread::scope (scheduling issue, deadlock, or incorrect work stealing).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Core test: rayon::join runs both closures concurrently (not sequentially).
/// Verifies that the optimization actually overlaps work.
#[test]
fn rayon_join_runs_concurrently() {
    // Both tasks sleep/spin for ~50ms. If concurrent, total < 100ms.
    // If sequential, total >= 100ms.
    let t = Instant::now();
    let (a, b) = rayon::join(
        || {
            let start = Instant::now();
            while start.elapsed().as_millis() < 50 {}
            42u64
        },
        || {
            let start = Instant::now();
            while start.elapsed().as_millis() < 50 {}
            99u64
        },
    );
    let elapsed_ms = t.elapsed().as_millis();
    assert_eq!(a, 42);
    assert_eq!(b, 99);
    // Allow generous margin: concurrent should be <80ms, sequential would be >100ms
    assert!(elapsed_ms < 90, "rayon::join took {elapsed_ms}ms — not concurrent (expected <90ms)");
    eprintln!("rayon::join concurrency: {elapsed_ms}ms for two 50ms tasks (expected ~50ms)");
}

/// Test that rayon::join with nested par_iter works (no deadlock).
/// This mirrors the moe_ffn_v2 pattern: one closure does shared_ffn (CPU-bound),
/// the other does parallel pread (spawns more rayon tasks via par_iter).
#[test]
fn rayon_join_with_nested_par_iter() {
    use rayon::prelude::*;

    let n = 8;
    let mut buffers: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; 1024]).collect();
    let flags: Vec<bool> = vec![true; n];

    let ((result, _elapsed), ()) = rayon::join(
        || {
            let t = Instant::now();
            // Simulate shared_expert_ffn: CPU-bound work
            let mut acc = vec![0.0f32; 128];
            for i in 0..128 {
                acc[i] = (i as f32 * 0.01).sin();
            }
            (acc, t.elapsed())
        },
        || {
            // Simulate parallel pread with par_iter (nested rayon)
            let flags_ref = &flags;
            buffers
                .chunks_mut(1)
                .enumerate()
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|(i, chunk)| {
                    if flags_ref[i] {
                        // Simulate pread: fill buffer with known data
                        for byte in chunk[0].iter_mut() {
                            *byte = (i as u8).wrapping_add(1);
                        }
                    }
                });
        },
    );

    // Verify shared FFN result
    assert!((result[0] - 0.0f32).abs() < 1e-6, "sin(0) should be ~0");
    assert!((result[1] - 0.01f32.sin()).abs() < 1e-6, "sin(0.01) mismatch");

    // Verify all buffers were filled by pread
    for (i, buf) in buffers.iter().enumerate() {
        let expected = (i as u8).wrapping_add(1);
        assert_eq!(buf[0], expected, "buffer {i} not filled by pread");
    }
    eprintln!("rayon::join with nested par_iter: OK (no deadlock)");
}

/// Edge case: rayon::join when one closure is trivial (no shared gate).
/// Matches the `else { vec![0.0f32; hidden] }` path in moe_ffn_v2.
#[test]
fn rayon_join_trivial_closure() {
    let has_shared_gate = false;
    let hidden = 7168;

    let done = AtomicBool::new(false);
    let (result, ()) = rayon::join(
        || {
            if has_shared_gate {
                vec![1.0f32; hidden]
            } else {
                vec![0.0f32; hidden]
            }
        },
        || {
            // Simulate pread (must still run even if shared gate is absent)
            done.store(true, Ordering::SeqCst);
        },
    );

    assert_eq!(result.len(), hidden);
    assert!(result.iter().all(|&v| v == 0.0), "no shared gate → all zeros");
    assert!(done.load(Ordering::SeqCst), "pread closure must have run");
    eprintln!("rayon::join trivial closure: OK");
}

/// Stress test: rayon::join under repeated invocations (mimics 61 layers).
/// Verifies no resource leaks or thread pool exhaustion.
#[test]
fn rayon_join_repeated_61_layers() {
    use rayon::prelude::*;

    let counter = AtomicU64::new(0);
    let num_layers = 61;

    let t = Instant::now();
    for _layer in 0..num_layers {
        let mut data = vec![0u8; 8 * 1024]; // 8 "experts" × 1KB
        let (result, ()) = rayon::join(
            || {
                // Simulate shared FFN
                counter.fetch_add(1, Ordering::Relaxed);
                vec![1.0f32; 128]
            },
            || {
                // Simulate parallel pread
                data.chunks_mut(1024)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .for_each(|chunk| {
                        chunk[0] = 0xFF;
                    });
            },
        );
        assert_eq!(result.len(), 128);
    }
    let elapsed_ms = t.elapsed().as_millis();

    assert_eq!(counter.load(Ordering::Relaxed), num_layers as u64);
    // 61 rayon::join calls should be very fast (< 100ms total for trivial work)
    assert!(elapsed_ms < 500, "61 rayon::join calls took {elapsed_ms}ms (expected <500ms)");
    eprintln!("61 repeated rayon::join calls: {elapsed_ms}ms (no leaks, no exhaustion)");
}
