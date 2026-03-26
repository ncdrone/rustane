//! Phase 0.5.2: Measure actual ANE TFLOPS via conv1x1 kernel (≈ matmul).
//!
//! Run: cargo test -p ane-bridge --test bench_ane_tflops -- --include-ignored --nocapture

use ane::{Graph, NSQualityOfService, Shape, TensorData};
use std::time::Instant;

#[test]
#[ignore = "requires ANE hardware"]
fn ane_conv1x1_tflops() {
    println!("\n=== Phase 0.5.2: ANE TFLOPS Benchmark (conv1x1 ≈ matmul) ===\n");

    // Configs: (in_channels, out_channels, spatial_width, label)
    // ANE conv1x1: [1, IC, 1, W] @ [OC, IC, 1, 1] → [1, OC, 1, W]
    // FLOPs = 2 * IC * OC * W
    let configs: Vec<(usize, usize, usize, &str)> = vec![
        (64, 64, 64, "64→64, w=64 (minimal)"),
        (128, 128, 128, "128→128, w=128"),
        (256, 256, 256, "256→256, w=256"),
        (768, 768, 512, "768→768, w=512 (gpt_karpathy Wo)"),
        (768, 3072, 512, "768→3072, w=512 (gpt_karpathy FFN up)"),
        (3072, 768, 512, "3072→768, w=512 (gpt_karpathy FFN down)"),
    ];

    println!(
        "{:<45} {:>10} {:>10} {:>10}",
        "Config", "µs", "TFLOPS", "iters"
    );
    println!("{}", "-".repeat(80));

    for (ic, oc, w, label) in configs {
        match bench_conv1x1(ic, oc, w) {
            Ok((median_us, tflops, iters)) => {
                println!("{label:<45} {median_us:>10.1} {tflops:>10.3} {iters:>10}");
            }
            Err(e) => {
                println!("{label:<45} FAILED: {e}");
            }
        }
    }
    println!();
}

fn bench_conv1x1(ic: usize, oc: usize, w: usize) -> Result<(f64, f64, usize), String> {
    let w = w.max(64); // ANE minimum spatial width

    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(ic, 1, w));

    // Conv1x1 weight: Shape::spatial(out_channels, 1, 1), data has oc*ic elements
    let w_data: Vec<f32> = (0..oc * ic)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let weight = g.constant(&w_data, Shape::spatial(oc, 1, 1));
    let _out = g.convolution_2d_1x1(x, weight, None);

    let executable = g
        .compile(NSQualityOfService::Default)
        .map_err(|e| format!("compile: {e}"))?;

    let input_data: Vec<f32> = (0..ic * w).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let input = TensorData::with_f32(&input_data, Shape::spatial(ic, 1, w));
    let output = TensorData::new(Shape::spatial(oc, 1, w));

    // Warmup
    for _ in 0..10 {
        executable
            .run(&[&input], &[&output])
            .map_err(|e| format!("eval: {e}"))?;
    }

    // Timed iterations
    let iters = 1000;
    let mut times_us = Vec::with_capacity(iters);

    for _ in 0..iters {
        let t0 = Instant::now();
        executable
            .run(&[&input], &[&output])
            .map_err(|e| format!("eval: {e}"))?;
        let elapsed = t0.elapsed();
        times_us.push(elapsed.as_secs_f64() * 1e6);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[iters / 2];

    // FLOPs for conv1x1: 2 * IC * OC * W
    let flops = 2.0 * ic as f64 * oc as f64 * w as f64;
    let tflops = flops / (median_us * 1e-6) / 1e12;

    Ok((median_us, tflops, iters))
}
