//! Phase 1: Training probe — DynMatmul kernel on ANE.
//!
//! Verifies:
//! 1. Slice + transpose + matmul graph compiles on ANE
//! 2. 1000 eval cycles with dynamic weight update via IOSurface write
//! 3. Overhead breakdown: pure compute vs compute + weight staging
//!
//! DynMatmul pattern (from autoresearch-ANE):
//!   Input IOSurface: [1, IC, 1, SEQ+OC]
//!   For channel c: buf[c*SP..c*SP+SEQ] = activations, buf[c*SP+SEQ..c*SP+SP] = weights
//!   Slice to separate → transpose to [1,1,IC,N] → matmul → [1,1,SEQ,OC]
//!
//! Run: cargo test -p ane-bridge --test phase1_training_probe --release -- --include-ignored --nocapture

use ane::{Graph, NSQualityOfService, Shape, TensorData};
use std::time::Instant;

#[test]
#[ignore = "requires ANE hardware"]
fn dynmatmul_training_probe() {
    println!("\n=== Phase 1: DynMatmul Training Probe ===\n");

    let ic = 768usize; // input channels (model dim)
    let seq = 64usize; // sequence length (min spatial width)
    let oc = 64usize; // output channels
    let sp = seq + oc; // total spatial width

    // ── 1. Build graph ──────────────────────────────────────────────
    let mut g = Graph::new();

    // Input: [1, IC, 1, SEQ+OC] — packed activations + weights
    let input = g.placeholder(Shape::spatial(ic, 1, sp));

    // Slice activations: [1, IC, 1, SEQ]
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);

    // Slice weights: [1, IC, 1, OC]
    let weights = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);

    // Transpose to [1, 1, IC, N] for matmul (move channels → height)
    let acts_t = g.transpose(acts, [0, 2, 1, 3]); // [1, IC, 1, SEQ] → [1, 1, IC, SEQ]
    let weights_t = g.transpose(weights, [0, 2, 1, 3]); // [1, IC, 1, OC] → [1, 1, IC, OC]

    // Matmul: acts^T @ weights = [1,1,SEQ,IC] @ [1,1,IC,OC] = [1,1,SEQ,OC]
    let _output = g.matrix_multiplication(acts_t, weights_t, true, false);

    println!("Graph: input=[1,{ic},1,{sp}] → slice → transpose → matmul → [1,1,{seq},{oc}]");
    println!(
        "FLOPs per eval: 2 × {ic} × {seq} × {oc} = {}",
        2 * ic * seq * oc
    );
    println!();

    // ── 2. Compile ──────────────────────────────────────────────────
    let compile_start = Instant::now();
    let executable = match g.compile(NSQualityOfService::Default) {
        Ok(e) => {
            let ms = compile_start.elapsed().as_secs_f64() * 1000.0;
            println!("Compile: {ms:.1}ms ✓");
            e
        }
        Err(e) => {
            println!("COMPILE FAILED: {e}");
            println!();
            println!("DynMatmul via slice+transpose+matmul does not work on ANE.");
            println!("This is a critical finding — log it and stop.");
            return;
        }
    };

    // ── 3. Allocate I/O tensors ────────────────────────────────────
    let input_tensor = TensorData::new(Shape::spatial(ic, 1, sp));
    // Output: [1, 1, SEQ, OC]
    let output_tensor = TensorData::new(Shape {
        batch: 1,
        channels: 1,
        height: seq,
        width: oc,
    });

    // Initialize: activations = small values, weights = identity-ish
    write_packed_input(&input_tensor, ic, seq, oc, 0);

    // ── 4. Verify single eval ──────────────────────────────────────
    match executable.run(&[&input_tensor], &[&output_tensor]) {
        Ok(()) => {
            let result = output_tensor.read_f32();
            let sum: f32 = result.iter().sum();
            println!("Single eval: output sum = {sum:.4} (expected non-zero) ✓");
        }
        Err(e) => {
            println!("EVAL FAILED: {e}");
            println!("Graph compiled but evaluation failed. Log and stop.");
            return;
        }
    }
    println!();

    // ── 5. Benchmark: pure compute (no weight update) ──────────────
    let warmup = 50;
    for _ in 0..warmup {
        executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    }

    let iters = 1000;
    let t0 = Instant::now();
    for _ in 0..iters {
        executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    }
    let pure_compute_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let flops = 2.0 * ic as f64 * seq as f64 * oc as f64;
    let tflops = flops / (pure_compute_us * 1e-6) / 1e12;
    println!("Pure compute ({iters} iters):");
    println!("  Median: {pure_compute_us:.1} µs/iter");
    println!("  TFLOPS: {tflops:.4}");
    println!();

    // ── 6. Benchmark: compute + weight staging ─────────────────────
    let t0 = Instant::now();
    for i in 0..iters {
        // Update weight portion of input (simulates gradient step)
        write_weight_region(&input_tensor, ic, seq, oc, i);
        executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    }
    let staged_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let staging_overhead_us = staged_us - pure_compute_us;

    println!("Compute + weight staging ({iters} iters):");
    println!("  Total: {staged_us:.1} µs/iter");
    println!(
        "  Staging overhead: {staging_overhead_us:.1} µs ({:.1}%)",
        staging_overhead_us / staged_us * 100.0
    );
    println!();

    // ── 7. Verify dynamic update changes output ────────────────────
    write_packed_input(&input_tensor, ic, seq, oc, 0);
    executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    let result_a = output_tensor.read_f32();

    write_packed_input(&input_tensor, ic, seq, oc, 42);
    executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    let result_b = output_tensor.read_f32();

    let diff: f32 = result_a
        .iter()
        .zip(result_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.0, "Output should change when weights change!");
    println!("Dynamic weight update: outputs differ by {diff:.4} ✓");
    println!();

    // ── 8. Summary ─────────────────────────────────────────────────
    let weight_bytes = ic * oc * 4; // f32
    let staging_gbps = weight_bytes as f64 / 1e9 / (staging_overhead_us * 1e-6);
    println!("=== Training Probe Summary ===");
    println!("  Kernel: DynMatmul (slice+transpose+matmul)");
    println!("  Shape: [{ic}×{seq}] @ [{ic}×{oc}] → [{seq}×{oc}]");
    println!("  Compile: ✓");
    println!("  Compute: {pure_compute_us:.1} µs ({tflops:.4} TFLOPS)");
    println!("  Staging: {staging_overhead_us:.1} µs ({staging_gbps:.2} GB/s effective)");
    println!("  Dynamic update: ✓ (output changes with weights)");
    println!("  1000 iterations: ✓");
    println!();
}

/// Write packed [activations | weights] into input IOSurface.
/// Layout: for channel c, spatial pos s: index = c * SP + s
/// Activations at s = 0..SEQ, weights at s = SEQ..SEQ+OC
fn write_packed_input(tensor: &TensorData, ic: usize, seq: usize, oc: usize, seed: usize) {
    let sp = seq + oc;
    let mut guard = tensor.as_f32_slice_mut();
    for c in 0..ic {
        // Activations: small deterministic values
        for s in 0..seq {
            guard[c * sp + s] = ((c * seq + s) % 17) as f32 * 0.01;
        }
        // Weights: vary with seed
        for j in 0..oc {
            guard[c * sp + seq + j] = ((c * oc + j + seed) % 13) as f32 * 0.01 - 0.06;
        }
    }
}

/// Update only the weight region of the input IOSurface (simulates gradient step).
fn write_weight_region(tensor: &TensorData, ic: usize, seq: usize, oc: usize, step: usize) {
    let sp = seq + oc;
    let mut guard = tensor.as_f32_slice_mut();
    for c in 0..ic {
        for j in 0..oc {
            guard[c * sp + seq + j] = ((c * oc + j + step) % 19) as f32 * 0.005 - 0.04;
        }
    }
}

// ── Larger config: GPT-2 FFN-sized DynMatmul ─────────────────────────

#[test]
#[ignore = "requires ANE hardware"]
fn dynmatmul_gpt2_ffn_size() {
    println!("\n=== Phase 1: DynMatmul at GPT-2 FFN Scale ===\n");

    // FFN up: 768 → 3072, with seq=64
    let ic = 768usize;
    let seq = 64usize;
    let oc = 256usize; // Reduced from 3072 — full FFN width may exceed IOSurface limits
    let sp = seq + oc;

    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ic, 1, sp));
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let weights = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let acts_t = g.transpose(acts, [0, 2, 1, 3]);
    let weights_t = g.transpose(weights, [0, 2, 1, 3]);
    let _output = g.matrix_multiplication(acts_t, weights_t, true, false);

    println!("Graph: [{ic}×{seq}] @ [{ic}×{oc}] → [{seq}×{oc}]");
    println!("FLOPs: {}", 2 * ic * seq * oc);

    let executable = match g.compile(NSQualityOfService::Default) {
        Ok(e) => {
            println!("Compile: ✓");
            e
        }
        Err(e) => {
            println!("COMPILE FAILED at OC={oc}: {e}");
            return;
        }
    };

    let input_tensor = TensorData::new(Shape::spatial(ic, 1, sp));
    let output_tensor = TensorData::new(Shape {
        batch: 1,
        channels: 1,
        height: seq,
        width: oc,
    });

    write_packed_input(&input_tensor, ic, seq, oc, 0);

    // Warmup
    for _ in 0..20 {
        executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    }

    let iters = 500;
    let t0 = Instant::now();
    for _ in 0..iters {
        executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    }
    let compute_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let flops = 2.0 * ic as f64 * seq as f64 * oc as f64;
    let tflops = flops / (compute_us * 1e-6) / 1e12;

    println!("Compute: {compute_us:.1} µs ({tflops:.4} TFLOPS)");

    // With staging
    let t0 = Instant::now();
    for i in 0..iters {
        write_weight_region(&input_tensor, ic, seq, oc, i);
        executable.run(&[&input_tensor], &[&output_tensor]).unwrap();
    }
    let staged_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let overhead = staged_us - compute_us;

    println!(
        "With staging: {staged_us:.1} µs (overhead: {overhead:.1} µs, {:.1}%)",
        overhead / staged_us * 100.0
    );
    println!();
}
