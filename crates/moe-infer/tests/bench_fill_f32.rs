//! Micro-benchmark: bulk convert_to_f32_slice vs per-element to_f32.
//! Measures f16→f32 conversion throughput for V3-sized tensors.

use half::f16;
use half::slice::HalfFloatSliceExt;

/// Benchmark conversion for a given size, returns (per_element_ms, bulk_ms).
fn bench_conversion(n: usize, iters: usize) -> (f64, f64) {
    // Create f16 source data
    let src: Vec<f16> = (0..n).map(|i| f16::from_f32((i as f32 * 0.001).sin())).collect();
    let mut dst = vec![0.0f32; n];

    // Per-element: scalar fcvt + runtime detection per call
    let t = std::time::Instant::now();
    for _ in 0..iters {
        for i in 0..n {
            dst[i] = src[i].to_f32();
        }
        std::hint::black_box(&dst);
    }
    let per_elem_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Bulk: vectorized FCVTL (4 at a time)
    let t = std::time::Instant::now();
    for _ in 0..iters {
        src.convert_to_f32_slice(&mut dst);
        std::hint::black_box(&dst);
    }
    let bulk_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    (per_elem_ms, bulk_ms)
}

#[test]
fn bench_fill_f32_v3_sizes() {
    // V3 tensor sizes (single layer, typical fields)
    let sizes = [
        ("q_proj (7168×7168)", 7168 * 7168),      // 51M elements
        ("w_uk (128×512×128)", 128 * 512 * 128),   // 8.4M elements
        ("o_proj (7168×7168)", 7168 * 7168),        // 51M elements
        ("shared_gate (18432×7168)", 18432 * 7168), // 132M elements
    ];

    eprintln!("\n=== f16→f32 conversion benchmark ===");
    eprintln!("{:<30} {:>10} {:>10} {:>8} {:>10}", "tensor", "per-elem", "bulk", "speedup", "GB/s(bulk)");
    for (name, n) in &sizes {
        let iters = if *n > 50_000_000 { 3 } else { 10 };
        let (pe, bulk) = bench_conversion(*n, iters);
        let speedup = pe / bulk;
        let gbps = (*n as f64 * 2.0) / (bulk / 1000.0) / 1e9; // f16 = 2 bytes
        eprintln!("{:<30} {:>8.2}ms {:>8.2}ms {:>7.1}x {:>8.1} GB/s", name, pe, bulk, speedup, gbps);
    }

    // Verify correctness: both paths produce identical results
    let n = 10_000;
    let src: Vec<f16> = (0..n).map(|i| f16::from_f32((i as f32 * 0.1).cos())).collect();
    let mut dst_elem = vec![0.0f32; n];
    let mut dst_bulk = vec![0.0f32; n];
    for i in 0..n { dst_elem[i] = src[i].to_f32(); }
    src.convert_to_f32_slice(&mut dst_bulk);
    assert_eq!(dst_elem, dst_bulk, "Bulk and per-element must produce identical results");
}
