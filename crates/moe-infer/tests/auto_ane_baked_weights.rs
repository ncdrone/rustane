//! CRITICAL EXPERIMENT: ANE baked-weight Conv1x1 vs CPU sgemv.
//!
//! This is the test that determines whether ANE can beat CPU for decode.
//! Baked weights = compile-time constants, zero staging overhead.
//! Per-dispatch: activation fp16 write (14 KB) + ANE compute (0.095ms) + output read.
//!
//! If baked Conv1x1 < CPU sgemv: ANE decode is viable.
//! If baked Conv1x1 > CPU sgemv: ANE is only for prefill/multi-stream.

use std::time::Instant;
use moe_kernels::mla::{build_mla_conv1x1_baked, pad16, MlaConfig};
use ane_bridge::ane::{Shape, TensorData};
use objc2_foundation::NSQualityOfService;

fn cpu_sgemv(w: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    moe_infer::blas::sgemv_f32(w, x, y, rows, cols);
}

#[test]
#[ignore = "requires ANE hardware"]
fn baked_vs_cpu_k2_q_lora_compress() {
    eprintln!("\n=== BAKED WEIGHT Conv1x1 vs CPU sgemv ===\n");
    let qos = NSQualityOfService::UserInteractive;

    let ic = 7168;  // K2 hidden
    let oc = 1536;  // q_lora_rank

    // Random weights
    let weights: Vec<f32> = (0..oc * ic).map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();
    let x: Vec<f32> = (0..ic).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

    // === CPU baseline ===
    let mut cpu_out = vec![0.0f32; oc];
    // Warmup
    for _ in 0..5 { cpu_sgemv(&weights, &x, &mut cpu_out, oc, ic); }
    let iters = 100;
    let t = Instant::now();
    for _ in 0..iters { cpu_sgemv(&weights, &x, &mut cpu_out, oc, ic); }
    let cpu_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("CPU sgemv [{}→{}]: {:.0}µs", ic, oc, cpu_us);

    // === ANE baked weight ===
    let t_compile = Instant::now();
    let graph = build_mla_conv1x1_baked(ic, oc, 1, &weights);
    let exec = graph.compile(qos).expect("baked compile");
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;
    eprintln!("ANE compile (baked [{}→{}]): {:.1}ms", ic, oc, compile_ms);

    let padded_seq = pad16(1);
    let input = TensorData::new(Shape { batch: 1, channels: ic, height: 1, width: padded_seq });
    let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: padded_seq });

    // Warmup
    for _ in 0..5 {
        unsafe { ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq); }
        exec.run_cached_direct(&[&input], &[&output]).unwrap();
    }

    // Measure
    let ane_iters = 50;
    let t = Instant::now();
    for _ in 0..ane_iters {
        unsafe { ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq); }
        exec.run_cached_direct(&[&input], &[&output]).unwrap();
        let mut ane_out = vec![0.0f32; oc];
        unsafe { ane_bridge::read_output_fp16(&output, &mut ane_out, oc, padded_seq); }
    }
    let ane_us = t.elapsed().as_micros() as f64 / ane_iters as f64;
    eprintln!("ANE baked [{}→{}]: {:.0}µs", ic, oc, ane_us);

    // Compare
    let ratio = ane_us / cpu_us;
    eprintln!("\nRatio: ANE/CPU = {:.2}x", ratio);
    if ratio < 1.0 {
        eprintln!("★★★ ANE WINS by {:.0}% ★★★", (1.0 - ratio) * 100.0);
    } else {
        eprintln!("CPU wins by {:.0}%", (ratio - 1.0) * 100.0);
    }

    // Correctness check
    let mut ane_out = vec![0.0f32; oc];
    unsafe {
        ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq);
    }
    exec.run_cached_direct(&[&input], &[&output]).unwrap();
    unsafe {
        ane_bridge::read_output_fp16(&output, &mut ane_out, oc, padded_seq);
    }

    let max_diff = cpu_out.iter().zip(&ane_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Correctness: max_diff = {:.6}", max_diff);

    let dot: f64 = cpu_out.iter().zip(&ane_out).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = cpu_out.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = ane_out.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb + 1e-10);
    eprintln!("Cosine similarity: {:.6}", cosine);
}

#[test]
#[ignore = "requires ANE hardware"]
fn baked_vs_cpu_all_k2_projections() {
    eprintln!("\n=== ALL K2 PROJECTIONS: BAKED vs CPU ===\n");
    let qos = NSQualityOfService::UserInteractive;

    let projs = [
        ("Q LoRA compress", 7168, 1536),
        ("Q LoRA expand", 1536, 12288),
        ("KV compress", 7168, 576),
        ("O projection", 8192, 7168),
        ("shared gate", 7168, 2048),
        ("shared up", 7168, 2048),
        ("shared down", 2048, 7168),
    ];

    eprintln!("{:<25} {:>10} {:>10} {:>8}", "Projection", "CPU µs", "ANE µs", "Ratio");
    eprintln!("{}", "-".repeat(57));

    let mut total_cpu = 0.0f64;
    let mut total_ane = 0.0f64;

    for (name, ic, oc) in projs {
        let weights: Vec<f32> = (0..oc * ic).map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();
        let x: Vec<f32> = (0..ic).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

        // CPU
        let mut cpu_out = vec![0.0f32; oc];
        for _ in 0..3 { cpu_sgemv(&weights, &x, &mut cpu_out, oc, ic); }
        let iters = if oc * ic > 10_000_000 { 20 } else { 50 };
        let t = Instant::now();
        for _ in 0..iters { cpu_sgemv(&weights, &x, &mut cpu_out, oc, ic); }
        let cpu_us = t.elapsed().as_micros() as f64 / iters as f64;

        // ANE baked
        let graph = build_mla_conv1x1_baked(ic, oc, 1, &weights);
        let exec = match graph.compile(qos) {
            Ok(e) => e,
            Err(e) => { eprintln!("{name}: compile failed: {e}"); continue; }
        };
        let padded_seq = pad16(1);
        let input = TensorData::new(Shape { batch: 1, channels: ic, height: 1, width: padded_seq });
        let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: padded_seq });

        // Warmup
        for _ in 0..3 {
            unsafe { ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq); }
            let _ = exec.run_cached_direct(&[&input], &[&output]);
        }

        let ane_iters = 30;
        let t = Instant::now();
        for _ in 0..ane_iters {
            unsafe { ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq); }
            exec.run_cached_direct(&[&input], &[&output]).unwrap();
            let mut out = vec![0.0f32; oc];
            unsafe { ane_bridge::read_output_fp16(&output, &mut out, oc, padded_seq); }
        }
        let ane_us = t.elapsed().as_micros() as f64 / ane_iters as f64;

        let ratio = ane_us / cpu_us;
        let marker = if ratio < 1.0 { "★ ANE" } else { "" };
        eprintln!("{:<25} {:>9.0} {:>9.0} {:>7.2}x {}", name, cpu_us, ane_us, ratio, marker);

        total_cpu += cpu_us;
        total_ane += ane_us;
    }

    eprintln!("{}", "-".repeat(57));
    eprintln!("{:<25} {:>9.0} {:>9.0} {:>7.2}x",
        "TOTAL (per layer)", total_cpu, total_ane, total_ane / total_cpu);
    eprintln!("\n× 61 layers:");
    eprintln!("  CPU total:  {:.1}ms", total_cpu * 61.0 / 1000.0);
    eprintln!("  ANE total:  {:.1}ms", total_ane * 61.0 / 1000.0);
}
