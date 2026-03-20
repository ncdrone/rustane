//! Benchmark: Metal 4-bit dequant GEMV throughput.
//!
//! Measures effective bandwidth (GiB/s) for fused dequant + GEMV.
//! Target: >400 GiB/s on M4 Max.
//!
//! Run with: cargo test -p moe-infer --test bench_dequant --release -- --ignored --nocapture

use moe_kernels::MetalDequantGemv;
use quantize::PackedWeights4Bit;

fn make_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_input(cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..cols)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Benchmark dequant GEMV at various matrix sizes.
/// Reports effective GiB/s (bytes read / time).
#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_dequant_gemv_bandwidth() {
    let metal = MetalDequantGemv::new().expect("Metal device required");

    let configs: Vec<(usize, usize, &str)> = vec![
        (768, 768, "Qwen3-MoE expert (small)"),
        (2048, 2048, "2K x 2K"),
        (4096, 4096, "4K x 4K (typical FFN)"),
        (7168, 2048, "DeepSeek-V3 expert down"),
        (2048, 7168, "DeepSeek-V3 expert up"),
        (8192, 8192, "8K x 8K (large)"),
    ];
    let gs = 128usize;
    let iters = 20usize;

    println!("\n{}", "=".repeat(70));
    println!("  Metal 4-bit Dequant GEMV Bandwidth Benchmark");
    println!("  group_size={gs}, {iters} iterations (best of {iters})");
    println!("{}", "=".repeat(70));
    println!(
        "  {:30} {:>10} {:>10} {:>10}",
        "Config", "Size (MB)", "Time (us)", "GiB/s"
    );
    println!("{}", "-".repeat(70));

    for (rows, cols, label) in configs {
        let weights = make_weights(rows, cols, 42);
        let x = make_input(cols, 99);
        let packed = PackedWeights4Bit::pack(&weights, rows, cols, gs);

        // Total bytes read by GPU:
        // packed weights: rows * cols / 2 bytes (4-bit)
        // scales + zeros: rows * (cols/gs) * 2 * 2 bytes (f16 each)
        // input vector: cols * 4 bytes (f32, cached)
        // output: rows * 4 bytes
        let weight_bytes = rows * cols / 2;
        let meta_bytes = rows * (cols / gs) * 4;
        let total_bytes = weight_bytes + meta_bytes;
        let size_mb = total_bytes as f64 / (1024.0 * 1024.0);

        let mut best_time = f64::MAX;
        for _ in 0..iters {
            let (_, elapsed) = metal.gemv_timed(&packed, &x);
            if elapsed < best_time {
                best_time = elapsed;
            }
        }

        let gib_per_sec = total_bytes as f64 / best_time / (1024.0 * 1024.0 * 1024.0);
        let time_us = best_time * 1e6;

        println!(
            "  {:30} {:>10.2} {:>10.1} {:>10.1}",
            format!("{label} ({rows}x{cols})"),
            size_mb,
            time_us,
            gib_per_sec,
        );
    }
    println!("{}", "-".repeat(70));
    println!("  Target: >400 GiB/s");
    println!();
}
