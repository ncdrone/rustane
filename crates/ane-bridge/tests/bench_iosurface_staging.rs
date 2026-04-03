//! Phase 0.5.3: IOSurface staging bandwidth benchmark.
//!
//! Measures:
//! 1. IOSurface lock/unlock overhead (empty cycle)
//! 2. f32 memcpy to IOSurface for training-relevant buffer sizes
//! 3. Channel-interleaved write (DynMatmul weight staging pattern)
//!
//! This is ~30% of training step time — know the actual number.
//!
//! Run: cargo test -p ane-bridge --test bench_iosurface_staging --release -- --include-ignored --nocapture

use ane::{Shape, TensorData};
use std::hint::black_box;
use std::time::Instant;

#[test]
#[ignore = "requires IOSurface (macOS only)"]
fn iosurface_staging_bandwidth() {
    println!("\n=== Phase 0.5.3: IOSurface Staging Bandwidth ===\n");

    // Training-relevant buffer sizes: (channels, spatial_width, label)
    let configs = [
        (768, 128, "768×128 (DynMatmul small, 393K B)"),
        (768, 576, "768×576 (DynMatmul Wo, 1.8M B)"),
        (768, 3584, "768×3584 (DynMatmul FFN, 11M B)"),
        (1024, 4096, "1024×4096 (Qwen3 FFN, 16.8M B)"),
    ];

    // ── 1. Lock/unlock overhead ────────────────────────────────────
    println!("--- Lock/Unlock Overhead ---");
    println!("{:<45} {:>10}", "Config", "µs/cycle");
    println!("{}", "-".repeat(60));

    for &(ch, w, label) in &configs {
        let tensor = TensorData::new(Shape::spatial(ch, 1, w));
        let iters = 10000;

        // Warmup
        for _ in 0..100 {
            let guard = tensor.as_f32_slice();
            black_box(&*guard);
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let guard = tensor.as_f32_slice();
            black_box(&*guard);
            // guard dropped here → unlock
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{label:<45} {us:>10.2}");
    }
    println!();

    // ── 2. Full f32 write via copy_from_f32 ────────────────────────
    println!("--- Full Write (copy_from_f32) ---");
    println!("{:<45} {:>10} {:>10} {:>10}", "Config", "µs", "MB", "GB/s");
    println!("{}", "-".repeat(80));

    for &(ch, w, label) in &configs {
        let tensor = TensorData::new(Shape::spatial(ch, 1, w));
        let count = ch * w;
        let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.001).collect();
        let mb = count as f64 * 4.0 / 1e6;

        // Warmup
        for _ in 0..10 {
            tensor.copy_from_f32(&data);
        }

        let iters = 1000;
        let t0 = Instant::now();
        for _ in 0..iters {
            tensor.copy_from_f32(&data);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gbps = (count as f64 * 4.0) / 1e9 / (us * 1e-6);
        println!("{label:<45} {us:>10.1} {mb:>10.2} {gbps:>10.2}");
    }
    println!();

    // ── 3. Channel-interleaved weight-only write ───────────────────
    println!("--- Weight-Only Staging (channel-interleaved) ---");
    println!(
        "{:<45} {:>10} {:>10} {:>10}",
        "Config (IC×OC)", "µs", "Weight MB", "GB/s"
    );
    println!("{}", "-".repeat(80));

    let staging_configs = [
        (768, 64, 64, "768: SEQ=64, OC=64 (probe)"),
        (768, 64, 512, "768: SEQ=64, OC=512 (Wo)"),
        (768, 64, 3072, "768: SEQ=64, OC=3072 (FFN up)"),
        (1024, 64, 3072, "1024: SEQ=64, OC=3072 (Qwen3 FFN)"),
    ];

    for &(ic, seq, oc, label) in &staging_configs {
        let sp = seq + oc;
        let tensor = TensorData::new(Shape::spatial(ic, 1, sp));
        let weight_bytes = ic * oc * 4;
        let weight_mb = weight_bytes as f64 / 1e6;

        // Warmup
        for _ in 0..10 {
            write_weights_interleaved(&tensor, ic, seq, oc, 0);
        }

        let iters = 1000;
        let t0 = Instant::now();
        for _ in 0..iters {
            write_weights_interleaved(&tensor, ic, seq, oc, 0);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gbps = weight_bytes as f64 / 1e9 / (us * 1e-6);
        println!("{label:<45} {us:>10.1} {weight_mb:>10.2} {gbps:>10.2}");
    }
    println!();

    // ── 4. Summary ─────────────────────────────────────────────────
    // Compare staging time vs ANE compute time for the FFN-sized case
    let ffn_staging_us = {
        let ic = 768;
        let seq = 64;
        let oc = 3072;
        let sp = seq + oc;
        let tensor = TensorData::new(Shape::spatial(ic, 1, sp));
        for _ in 0..10 {
            write_weights_interleaved(&tensor, ic, seq, oc, 0);
        }
        let iters = 1000;
        let t0 = Instant::now();
        for _ in 0..iters {
            write_weights_interleaved(&tensor, ic, seq, oc, 0);
        }
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    println!("=== Summary ===");
    println!("FFN weight staging (768×3072): {ffn_staging_us:.1} µs");
    println!("ANE conv1x1 compute (768→3072): ~330 µs (from bench_ane_tflops)");
    println!(
        "Staging as % of compute: {:.1}%",
        ffn_staging_us / 330.0 * 100.0
    );
    println!("M4 Max memory bandwidth: 546 GB/s");
    println!();
}

fn write_weights_interleaved(tensor: &TensorData, ic: usize, seq: usize, oc: usize, seed: usize) {
    let sp = seq + oc;
    let mut guard = tensor.as_f32_slice_mut();
    for c in 0..ic {
        for j in 0..oc {
            guard[c * sp + seq + j] = ((c * oc + j + seed) % 19) as f32 * 0.005;
        }
    }
}
