//! ANE batch scaling for ALL K2 projections: find the crossover point for each.
//! Shared gate crosses at seq=2. Do the others cross too?

use std::time::Instant;
use moe_kernels::mla::{build_mla_matmul_baked, pad16};
use ane_bridge::ane::{Shape, TensorData};
use objc2_foundation::NSQualityOfService;

#[test]
#[ignore = "requires ANE hardware"]
fn ane_batch_crossover_all() {
    eprintln!("\n=== ANE Batch Crossover — ALL K2 Projections ===\n");
    let qos = NSQualityOfService::UserInteractive;

    let projs = [
        ("Q LoRA compress", 7168, 1536),
        ("Q LoRA expand", 1536, 12288),
        ("KV compress", 7168, 576),
        ("O projection", 8192, 7168),
        ("shared gate", 7168, 2048),
    ];

    for (name, ic, oc) in projs {
        eprintln!("--- {name} [{ic}→{oc}] ---");
        let weights: Vec<f32> = (0..oc * ic).map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();

        let mut crossover = 0;

        for &seq in &[1, 2, 4, 8, 16, 32, 64] {
            let x: Vec<f32> = (0..ic * seq).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

            // ANE baked
            let graph = build_mla_matmul_baked(ic, oc, seq, &weights);
            let exec = match graph.compile(qos) {
                Ok(e) => e,
                Err(e) => { eprintln!("  seq={seq}: COMPILE FAILED: {e}"); continue; }
            };

            let padded_seq = pad16(seq);
            let input = TensorData::new(Shape { batch: 1, channels: ic, height: 1, width: padded_seq });
            let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: padded_seq });

            // Stage
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
            for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

            let iters = 20;
            let t = Instant::now();
            for _ in 0..iters {
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

            // CPU
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
            eprintln!("  seq={:<3} ANE {:>5.0}µs/tok  CPU {:>5.0}µs/tok  ratio {:.2}x {}",
                seq, ane_per_tok, cpu_per_tok, ratio, marker);

            if ratio < 1.0 && crossover == 0 {
                crossover = seq;
            }
        }

        if crossover > 0 {
            eprintln!("  CROSSOVER at seq={crossover} ★★★");
        } else {
            eprintln!("  NO CROSSOVER (CPU always wins at tested seq_lens)");
        }
        eprintln!();
    }
}
