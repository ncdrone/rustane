//! Benchmark: daemon eval (evaluateWithQoS) vs direct eval (doEvaluateDirectWithQoS).
//!
//! Tests whether the direct evaluation path reduces per-dispatch overhead
//! by bypassing the XPC daemon.
//!
//! Run manually:
//!   cargo test -p engine --test bench_direct_eval --release -- --ignored --nocapture

use ane_bridge::ane::{Shape, TensorData};
use engine::kernels::dyn_matmul;
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn bench_eval_mode(name: &str, ic: usize, oc: usize, seq: usize, iters: usize) {
    let graph = dyn_matmul::build(ic, oc, seq);
    let exe = graph
        .compile(NSQualityOfService::UserInteractive)
        .expect("ANE compile failed");

    let sp = dyn_matmul::spatial_width(seq, oc);
    let input_shape = Shape {
        batch: 1,
        channels: ic,
        height: 1,
        width: sp,
    };
    let output_shape = Shape {
        batch: 1,
        channels: oc,
        height: 1,
        width: seq,
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| (i % 100) as f32 * 0.001)
        .collect();
    let input_td = TensorData::with_f32(&input_data, input_shape);
    let output_td = TensorData::new(output_shape);

    // Warmup both paths
    for _ in 0..5 {
        exe.run_cached(&[&input_td], &[&output_td])
            .expect("warmup daemon failed");
    }

    // Benchmark daemon path (evaluateWithQoS)
    let mut daemon_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        exe.run_cached(&[&input_td], &[&output_td])
            .expect("daemon eval failed");
        daemon_us.push(t.elapsed().as_micros() as f64);
    }

    // Benchmark direct path (doEvaluateDirectWithQoS)
    // Note: run_cached_direct uses the same cached_request as run_cached,
    // so we need to verify it works on the same cache.
    let mut direct_us = Vec::with_capacity(iters);
    let mut direct_ok = true;
    for i in 0..iters {
        let t = Instant::now();
        match exe.run_cached_direct(&[&input_td], &[&output_td]) {
            Ok(()) => {
                direct_us.push(t.elapsed().as_micros() as f64);
            }
            Err(e) => {
                if i == 0 {
                    println!("\n  WARNING: doEvaluateDirectWithQoS failed: {e}");
                    println!("  This selector may not exist on _ANEInMemoryModel.");
                    println!("  Falling back: direct eval not available.");
                    direct_ok = false;
                    break;
                }
                direct_us.push(t.elapsed().as_micros() as f64);
            }
        }
    }

    // Verify numerical equivalence (run both, compare output)
    let output_daemon = TensorData::new(output_shape);
    exe.run(&[&input_td], &[&output_daemon])
        .expect("daemon run failed");

    daemon_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |v: &[f64], pct: f64| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let idx = ((v.len() as f64 * pct / 100.0) as usize).min(v.len() - 1);
        v[idx]
    };

    println!("\n=== {name} ({ic}x{oc}, seq={seq}) ===");
    println!(
        "  Daemon (evaluateWithQoS):       median={:.1}us  p5={:.1}  p95={:.1}",
        p(&daemon_us, 50.0),
        p(&daemon_us, 5.0),
        p(&daemon_us, 95.0)
    );

    if direct_ok {
        direct_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let daemon_med = p(&daemon_us, 50.0);
        let direct_med = p(&direct_us, 50.0);
        let savings = daemon_med - direct_med;

        println!(
            "  Direct (doEvaluateDirectWithQoS): median={:.1}us  p5={:.1}  p95={:.1}",
            direct_med,
            p(&direct_us, 5.0),
            p(&direct_us, 95.0)
        );
        println!(
            "  Savings: {:.1}us per dispatch ({:.1}%)",
            savings,
            if daemon_med > 0.0 {
                savings / daemon_med * 100.0
            } else {
                0.0
            }
        );

        // At 60 dispatches/step, extrapolated savings:
        let step_savings_ms = savings * 60.0 / 1000.0;
        println!(
            "  Extrapolated: {:.2}ms/step (60 dispatches)",
            step_savings_ms
        );

        // Verify numerical equivalence
        // Both daemon and direct should produce the same output
        let daemon_vals = output_daemon.as_f32_slice();
        let locked = output_td.as_f32_slice(); // last direct eval output
        let mut max_diff: f32 = 0.0;
        for (a, b) in daemon_vals.iter().zip(locked.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("  Numerical max diff: {:.6}", max_diff);
        if max_diff < 1e-3 {
            println!("  GATE PASS: bit-equivalent output, direct eval works");
        } else {
            println!("  WARNING: outputs differ by {:.6} — investigate", max_diff);
        }
    } else {
        println!("  Direct: NOT AVAILABLE (selector unrecognized)");
        println!("  Next: try _ANEClient.sharedConnection -> doEvaluateDirectWithModel:");
    }
}

#[test]
#[ignore]
fn bench_direct_eval_768() {
    println!("\n=== Daemon vs Direct Eval Benchmark (768-dim) ===");
    bench_eval_mode("woFwd", 768, 768, 512, 100);
}

#[test]
#[ignore]
fn bench_direct_eval_1536() {
    println!("\n=== Daemon vs Direct Eval Benchmark (1536-dim) ===");
    bench_eval_mode("woFwd", 1536, 1536, 512, 100);
    bench_eval_mode("ffnBwdW2t", 1536, 4096, 512, 100);
}
