//! Phase 0.5.5: Benchmark f32↔f16 conversion methods.
//!
//! Compares:
//! 1. `half` crate software conversion
//! 2. ane crate's built-in f32_to_fp16_bytes
//! 3. (Future) NEON intrinsics
//!
//! Run: cargo test -p ane-bridge --test bench_f16_convert -- --include-ignored --nocapture

use std::hint::black_box;
use std::time::Instant;

#[test]
fn bench_f32_to_f16_conversion() {
    println!("\n=== Phase 0.5.5: f32↔f16 Conversion Benchmark ===\n");

    // Training-relevant buffer sizes
    let sizes = [
        (768 * 512, "DIM×SEQ (768×512 = 393K)"),
        (3072 * 512, "HIDDEN×SEQ (3072×512 = 1.6M)"),
        (768 * 768, "DIM² (768² = 590K)"),
        (768 * 3072, "DIM×HIDDEN (768×3072 = 2.4M)"),
    ];

    println!(
        "{:<45} {:>10} {:>10} {:>10}",
        "Buffer", "half (ms)", "ane (ms)", "ratio"
    );
    println!("{}", "-".repeat(85));

    for (count, label) in sizes {
        let data: Vec<f32> = (0..count).map(|i| (i as f32 * 0.001) - 500.0).collect();

        // Method 1: half crate
        let half_ms = bench_half_crate_f32_to_f16(&data);

        // Method 2: ane crate's f32_to_fp16_bytes
        let ane_ms = bench_ane_f32_to_f16(&data);

        let ratio = half_ms / ane_ms;
        println!("{label:<45} {half_ms:>10.3} {ane_ms:>10.3} {ratio:>10.2}x");
    }

    println!();

    // Also benchmark f16 → f32 (read path)
    println!("{:<45} {:>10}", "f16→f32 Buffer", "half (ms)");
    println!("{}", "-".repeat(55));

    for (count, label) in sizes {
        let f16_data: Vec<half::f16> = (0..count)
            .map(|i| half::f16::from_f32((i as f32 * 0.001) - 500.0))
            .collect();

        let ms = bench_half_crate_f16_to_f32(&f16_data);
        println!("{label:<45} {ms:>10.3}");
    }

    println!();

    // Throughput summary
    let big_count = 768 * 3072; // 2.4M elements
    let big_data: Vec<f32> = (0..big_count).map(|i| i as f32 * 0.001).collect();
    let half_ms = bench_half_crate_f32_to_f16(&big_data);
    let bytes = big_count * 4; // f32 input bytes
    let gb_per_sec = (bytes as f64 / 1e9) / (half_ms / 1000.0);
    println!(
        "Peak throughput (half crate, f32→f16): {gb_per_sec:.2} GB/s ({big_count} elements in {half_ms:.3}ms)"
    );
    println!("Memory bandwidth: 546 GB/s (M4 Max shared)");
    println!(
        "Conversion overhead: {:.1}% of bandwidth",
        (1.0 - gb_per_sec / 546.0) * 100.0
    );
}

fn bench_half_crate_f32_to_f16(data: &[f32]) -> f64 {
    // Warmup
    for _ in 0..5 {
        let v: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        black_box(&v);
    }

    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let v: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        black_box(&v);
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

fn bench_ane_f32_to_f16(data: &[f32]) -> f64 {
    // Warmup
    for _ in 0..5 {
        let v = ane::f32_to_fp16_bytes(data);
        black_box(&v);
    }

    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let v = ane::f32_to_fp16_bytes(data);
        black_box(&v);
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

fn bench_half_crate_f16_to_f32(data: &[half::f16]) -> f64 {
    // Warmup
    for _ in 0..5 {
        let v: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
        black_box(&v);
    }

    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let v: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
        black_box(&v);
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}
