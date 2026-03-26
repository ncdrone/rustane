//! Benchmark: matmul-based dyn_matmul vs conv1x1-based dyn_matmul.
//!
//! Verifies numerical equivalence and measures hw throughput difference.
//!
//! Run manually:
//!   cargo test -p engine --test bench_conv_vs_matmul --release -- --ignored --nocapture

use ane_bridge::ane::{Shape, TensorData};
use engine::kernels::dyn_matmul;
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn bench_matmul_vs_conv(name: &str, ic: usize, oc: usize, seq: usize, iters: usize) {
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

    // Compile both variants
    let graph_mm = dyn_matmul::build(ic, oc, seq);
    let exe_mm = graph_mm
        .compile(NSQualityOfService::UserInteractive)
        .expect("matmul compile failed");

    let graph_conv = dyn_matmul::build_conv(ic, oc, seq);
    let exe_conv = match graph_conv.compile(NSQualityOfService::UserInteractive) {
        Ok(exe) => exe,
        Err(e) => {
            println!("\n=== {name} ({ic}x{oc}, seq={seq}) ===");
            println!("  Conv1x1 compile FAILED: {e}");
            println!("  ANE compiler may not support dynamic-weight conv.");
            return;
        }
    };

    // Create test data with reproducible small values
    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.001 - 0.1)
        .collect();
    let input_td = TensorData::with_f32(&input_data, input_shape);

    let out_mm = TensorData::new(output_shape);
    let out_conv = TensorData::new(output_shape);

    // Verify numerical equivalence
    exe_mm
        .run(&[&input_td], &[&out_mm])
        .expect("matmul eval failed");
    exe_conv
        .run(&[&input_td], &[&out_conv])
        .expect("conv eval failed");

    let mm_vals = out_mm.as_f32_slice();
    let conv_vals = out_conv.as_f32_slice();

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for (a, b) in mm_vals.iter().zip(conv_vals.iter()) {
        let d = (a - b).abs();
        max_diff = max_diff.max(d);
        sum_diff += d;
    }
    let avg_diff = sum_diff / mm_vals.len() as f32;

    // Warmup
    for _ in 0..10 {
        exe_mm
            .run_cached(&[&input_td], &[&out_mm])
            .expect("warmup mm");
        exe_conv
            .run_cached(&[&input_td], &[&out_conv])
            .expect("warmup conv");
    }

    // Benchmark matmul path
    let mut mm_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        exe_mm
            .run_cached(&[&input_td], &[&out_mm])
            .expect("mm eval");
        mm_us.push(t.elapsed().as_micros() as f64);
    }

    // Benchmark conv1x1 path
    let mut conv_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        exe_conv
            .run_cached(&[&input_td], &[&out_conv])
            .expect("conv eval");
        conv_us.push(t.elapsed().as_micros() as f64);
    }

    mm_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    conv_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p = |v: &[f64], pct: f64| -> f64 {
        let idx = ((v.len() as f64 * pct / 100.0) as usize).min(v.len() - 1);
        v[idx]
    };

    let mm_med = p(&mm_us, 50.0);
    let conv_med = p(&conv_us, 50.0);
    let speedup = mm_med / conv_med;

    println!("\n=== {name} ({ic}x{oc}, seq={seq}) ===");
    println!(
        "  Numerical: max_diff={:.6}, avg_diff={:.8}",
        max_diff, avg_diff
    );
    if max_diff < 1e-2 {
        println!("  GATE PASS: conv output matches matmul within tolerance");
    } else {
        println!("  WARNING: diff too large, investigate");
    }
    println!(
        "  Matmul:  median={:.1}us  p5={:.1}  p95={:.1}",
        mm_med,
        p(&mm_us, 5.0),
        p(&mm_us, 95.0)
    );
    println!(
        "  Conv1x1: median={:.1}us  p5={:.1}  p95={:.1}",
        conv_med,
        p(&conv_us, 5.0),
        p(&conv_us, 95.0)
    );
    println!(
        "  Speedup: {:.2}x (conv is {})",
        speedup,
        if speedup > 1.0 { "FASTER" } else { "SLOWER" }
    );
}

fn bench_dual(name: &str, ic: usize, oc: usize, seq: usize, iters: usize) {
    let sp = dyn_matmul::dual_spatial_width(seq, oc);
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

    let graph_mm = dyn_matmul::build_dual(ic, oc, seq);
    let exe_mm = graph_mm
        .compile(NSQualityOfService::UserInteractive)
        .expect("dual matmul compile failed");

    let graph_conv = dyn_matmul::build_dual_conv(ic, oc, seq);
    let exe_conv = match graph_conv.compile(NSQualityOfService::UserInteractive) {
        Ok(exe) => exe,
        Err(e) => {
            println!("\n=== {name} dual ({ic}x{oc}, seq={seq}) ===");
            println!("  Dual conv1x1 compile FAILED: {e}");
            return;
        }
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.001 - 0.1)
        .collect();
    let input_td = TensorData::with_f32(&input_data, input_shape);
    let out_mm = TensorData::new(output_shape);
    let out_conv = TensorData::new(output_shape);

    exe_mm.run(&[&input_td], &[&out_mm]).expect("dual mm eval");
    exe_conv
        .run(&[&input_td], &[&out_conv])
        .expect("dual conv eval");

    let mm_vals = out_mm.as_f32_slice();
    let conv_vals = out_conv.as_f32_slice();
    let mut max_diff: f32 = 0.0;
    for (a, b) in mm_vals.iter().zip(conv_vals.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }

    // Warmup
    for _ in 0..10 {
        exe_mm.run_cached(&[&input_td], &[&out_mm]).unwrap();
        exe_conv.run_cached(&[&input_td], &[&out_conv]).unwrap();
    }

    let mut mm_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        exe_mm.run_cached(&[&input_td], &[&out_mm]).unwrap();
        mm_us.push(t.elapsed().as_micros() as f64);
    }
    let mut conv_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        exe_conv.run_cached(&[&input_td], &[&out_conv]).unwrap();
        conv_us.push(t.elapsed().as_micros() as f64);
    }

    mm_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    conv_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |v: &[f64], pct: f64| -> f64 {
        let idx = ((v.len() as f64 * pct / 100.0) as usize).min(v.len() - 1);
        v[idx]
    };
    let mm_med = p(&mm_us, 50.0);
    let conv_med = p(&conv_us, 50.0);

    println!("\n=== {name} dual ({ic}x{oc}, seq={seq}) ===");
    println!("  Numerical: max_diff={:.6}", max_diff);
    println!("  Matmul:  median={:.1}us", mm_med);
    println!("  Conv1x1: median={:.1}us", conv_med);
    println!("  Speedup: {:.2}x", mm_med / conv_med);
}

#[test]
#[ignore]
fn bench_conv_vs_matmul_768() {
    println!("\n=== Conv1x1 vs Matmul Benchmark (768-dim) ===");
    bench_matmul_vs_conv("woFwd", 768, 768, 512, 100);
    bench_matmul_vs_conv("ffnBwdW2t", 768, 2048, 512, 100);
    bench_dual("ffnBwdW13t", 2048, 768, 512, 100);
}

#[test]
#[ignore]
fn bench_conv_vs_matmul_1536() {
    println!("\n=== Conv1x1 vs Matmul Benchmark (1536-dim) ===");
    bench_matmul_vs_conv("woFwd", 1536, 1536, 512, 100);
    bench_matmul_vs_conv("ffnBwdW2t", 1536, 4096, 512, 100);
    bench_dual("ffnBwdW13t", 4096, 1536, 512, 100);
}
