//! Deep ANE dispatch timing: break down EXACTLY where time goes in baked matmul.
//!
//! The baked matmul benchmark showed ANE shared gate beats CPU (412µs vs 719µs).
//! But Q LoRA compress lost (362µs vs 305µs). Why? What's the per-component cost?
//!
//! This test isolates: IOSurface lock, fp16 write, ANE compute, fp16 read, unlock.

use std::time::Instant;
use moe_kernels::mla::{build_mla_matmul_baked, pad16};
use ane_bridge::ane::{Shape, TensorData};
use objc2_foundation::NSQualityOfService;

#[test]
#[ignore = "requires ANE hardware"]
fn ane_baked_dispatch_breakdown() {
    eprintln!("\n=== ANE Baked Dispatch Breakdown (µs per component) ===\n");
    let qos = NSQualityOfService::UserInteractive;

    let cases = [
        ("Q LoRA compress", 7168, 1536),
        ("Q LoRA expand", 1536, 12288),
        ("KV compress", 7168, 576),
        ("O projection", 8192, 7168),
        ("shared gate", 7168, 2048),
        ("shared down", 2048, 7168),
    ];

    eprintln!("{:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Projection", "stage", "disp", "read", "TOTAL", "CPU", "ANE/CPU");
    eprintln!("{}", "-".repeat(75));

    for (name, ic, oc) in cases {
        let weights: Vec<f32> = (0..oc * ic).map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();
        let x: Vec<f32> = (0..ic).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

        // Compile
        let graph = build_mla_matmul_baked(ic, oc, 1, &weights);
        let exec = match graph.compile(qos) {
            Ok(e) => e,
            Err(e) => { eprintln!("{name}: compile failed: {e}"); continue; }
        };

        let padded_seq = pad16(1);
        let input = TensorData::new(Shape { batch: 1, channels: ic, height: 1, width: padded_seq });
        let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: padded_seq });

        // Warmup
        for _ in 0..10 {
            unsafe { ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq); }
            exec.run_cached_direct(&[&input], &[&output]).unwrap();
        }

        let iters = 50;
        let mut stage_total = 0u64;
        let mut dispatch_total = 0u64;
        let mut read_total = 0u64;

        for _ in 0..iters {
            // Stage
            let t = Instant::now();
            unsafe { ane_bridge::stage_activation_fp16(&input, &x, ic, padded_seq); }
            stage_total += t.elapsed().as_nanos() as u64;

            // Dispatch
            let t = Instant::now();
            exec.run_cached_direct(&[&input], &[&output]).unwrap();
            dispatch_total += t.elapsed().as_nanos() as u64;

            // Read
            let t = Instant::now();
            let mut out = vec![0.0f32; oc];
            unsafe { ane_bridge::read_output_fp16(&output, &mut out, oc, padded_seq); }
            read_total += t.elapsed().as_nanos() as u64;
        }

        let stage_us = stage_total as f64 / iters as f64 / 1000.0;
        let dispatch_us = dispatch_total as f64 / iters as f64 / 1000.0;
        let read_us = read_total as f64 / iters as f64 / 1000.0;
        let total_us = stage_us + dispatch_us + read_us;

        // CPU comparison
        let mut cpu_out = vec![0.0f32; oc];
        for _ in 0..5 { moe_infer::blas::sgemm_custom_1xn(&x, &weights, &mut cpu_out, ic, oc); }
        let t = Instant::now();
        for _ in 0..iters { moe_infer::blas::sgemm_custom_1xn(&x, &weights, &mut cpu_out, ic, oc); }
        let cpu_us = t.elapsed().as_micros() as f64 / iters as f64;

        let ratio = total_us / cpu_us;
        let marker = if ratio < 1.0 { "★" } else { "" };
        eprintln!("{:<22} {:>6.0} {:>6.0} {:>6.0} {:>6.0} {:>6.0}  {:>5.2}x {}",
            name, stage_us, dispatch_us, read_us, total_us, cpu_us, ratio, marker);
    }
}

#[test]
#[ignore = "requires ANE hardware"]
fn ane_baked_batch_dispatch() {
    eprintln!("\n=== ANE Baked: Single vs Batch Token Dispatch ===\n");
    eprintln!("Does ANE improve relative to CPU when processing more tokens?\n");
    let qos = NSQualityOfService::UserInteractive;

    let ic = 7168;
    let oc = 2048; // shared gate (ANE wins at seq=1)

    let weights: Vec<f32> = (0..oc * ic).map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();

    for &seq in &[1, 2, 4, 8, 16, 32, 64] {
        let x: Vec<f32> = (0..ic * seq).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

        // ANE baked — need to rebuild graph for each seq_len
        let graph = build_mla_matmul_baked(ic, oc, seq, &weights);
        let exec = match graph.compile(qos) {
            Ok(e) => e,
            Err(e) => { eprintln!("seq={seq}: compile failed: {e}"); continue; }
        };

        let padded_seq = pad16(seq);
        let input = TensorData::new(Shape { batch: 1, channels: ic, height: 1, width: padded_seq });
        let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: padded_seq });

        // Stage all tokens
        // For baked weights, input is just activations [1, IC, 1, padded_seq]
        // Channel-interleaved: buf[c * padded_seq + t] = activation[t * ic + c]
        {
            let mut buf = input.as_f32_slice_mut();
            for v in buf.iter_mut() { *v = 0.0; }
            for t in 0..seq {
                for c in 0..ic {
                    buf[c * padded_seq + t] = x[t * ic + c];
                }
            }
        }

        // Warmup
        for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

        let iters = 30;
        let t = Instant::now();
        for _ in 0..iters {
            // Restage activations
            {
                let mut buf = input.as_f32_slice_mut();
                for t_idx in 0..seq {
                    for c in 0..ic {
                        buf[c * padded_seq + t_idx] = x[t_idx * ic + c];
                    }
                }
            }
            exec.run_cached_direct(&[&input], &[&output]).unwrap();
        }
        let ane_us = t.elapsed().as_micros() as f64 / iters as f64;
        let ane_per_tok = ane_us / seq as f64;

        // CPU comparison: process seq tokens one at a time
        let mut cpu_out = vec![0.0f32; oc];
        let t = Instant::now();
        for _ in 0..iters {
            for t_idx in 0..seq {
                let xt = &x[t_idx * ic..(t_idx + 1) * ic];
                moe_infer::blas::sgemm_custom_1xn(xt, &weights, &mut cpu_out, ic, oc);
            }
        }
        let cpu_us = t.elapsed().as_micros() as f64 / iters as f64;
        let cpu_per_tok = cpu_us / seq as f64;

        let ratio = ane_per_tok / cpu_per_tok;
        let marker = if ratio < 1.0 { "★ ANE" } else { "" };
        eprintln!("  seq={:<3}  ANE: {:>7.0}µs ({:>5.0}/tok)  CPU: {:>7.0}µs ({:>5.0}/tok)  ratio: {:>.2}x {}",
            seq, ane_us, ane_per_tok, cpu_us, cpu_per_tok, ratio, marker);
    }
}
