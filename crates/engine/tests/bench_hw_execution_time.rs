//! Benchmark: ANE hardware execution time vs wall clock time.
//!
//! Uses _ANEPerformanceStats to measure actual hardware nanoseconds,
//! separating XPC/dispatch overhead from real ANE compute.
//!
//! Run manually:
//!   cargo test -p engine --test bench_hw_execution_time --release -- --ignored --nocapture

use ane_bridge::ane::{Shape, TensorData};
use engine::kernels::dyn_matmul;
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn bench_kernel(name: &str, ic: usize, oc: usize, seq: usize, iters: usize) {
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

    // Warmup: 5 iterations (using run_cached to prime the cached request)
    for _ in 0..5 {
        exe.run_cached(&[&input_td], &[&output_td])
            .expect("warmup failed");
    }

    // Collect wall clock times using run_cached (no stats)
    let mut wall_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        exe.run_cached(&[&input_td], &[&output_td])
            .expect("eval failed");
        wall_us.push(t.elapsed().as_micros() as f64);
    }

    // Now collect hw times using run_cached_with_stats
    // (uses separate cached request, so no conflict)
    let mut hw_ns = Vec::with_capacity(iters);
    let mut wall_stats_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let hw = exe
            .run_cached_with_stats(&[&input_td], &[&output_td])
            .expect("stats eval failed");
        wall_stats_us.push(t.elapsed().as_micros() as f64);
        hw_ns.push(hw as f64);
    }

    // Sort for percentile computation
    wall_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    wall_stats_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hw_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p = |v: &[f64], pct: f64| -> f64 {
        let idx = ((v.len() as f64 * pct / 100.0) as usize).min(v.len() - 1);
        v[idx]
    };

    let median_wall = p(&wall_us, 50.0);
    let median_wall_stats = p(&wall_stats_us, 50.0);
    let median_hw = p(&hw_ns, 50.0);
    let overhead_us = median_wall - (median_hw / 1000.0);
    let overhead_pct = if median_wall > 0.0 {
        overhead_us / median_wall * 100.0
    } else {
        0.0
    };

    println!("\n=== {name} ({ic}x{oc}, seq={seq}) ===");
    println!(
        "  Wall clock (no stats):   median={:.1}us  p5={:.1}  p95={:.1}",
        median_wall,
        p(&wall_us, 5.0),
        p(&wall_us, 95.0)
    );
    println!(
        "  Wall clock (with stats): median={:.1}us  p5={:.1}  p95={:.1}",
        median_wall_stats,
        p(&wall_stats_us, 5.0),
        p(&wall_stats_us, 95.0)
    );
    println!(
        "  HW execution time:       median={:.0}ns  p5={:.0}  p95={:.0}",
        median_hw,
        p(&hw_ns, 5.0),
        p(&hw_ns, 95.0)
    );
    println!(
        "  Overhead (wall - hw):     {:.1}us  ({:.1}%)",
        overhead_us, overhead_pct
    );

    if median_hw == 0.0 {
        println!("  WARNING: hwExecutionTime returned 0 — perfStats may not be populated");
        println!("  Try setting perfStatsMask on the model (0x1, 0xFF, 0xFFFFFFFF)");
    } else {
        assert!(
            median_hw < median_wall * 1000.0,
            "hw time ({:.0}ns) should be less than wall clock ({:.1}us = {:.0}ns)",
            median_hw,
            median_wall,
            median_wall * 1000.0
        );
        println!("  GATE PASS: hw_time > 0 and hw_time < wall_clock");
    }
}

#[test]
#[ignore]
fn bench_hw_time_768() {
    let cfg = ModelConfig::gpt_karpathy();
    println!("\n=== ANE HW Execution Time Benchmark (48.8M / 768-dim) ===");
    bench_kernel("woFwd", cfg.q_dim, cfg.dim, cfg.seq, 100);
    bench_kernel("ffnBwdW2t", cfg.dim, cfg.hidden, cfg.seq, 100);
}

#[test]
#[ignore]
fn bench_hw_time_1536() {
    println!("\n=== ANE HW Execution Time Benchmark (579M / 1536-dim) ===");
    // 579M model dimensions
    let dim = 1536;
    let hidden = 4096;
    let q_dim = 1536;
    let seq = 512;
    bench_kernel("woFwd", q_dim, dim, seq, 100);
    bench_kernel("ffnBwdW2t", dim, hidden, seq, 100);
}
