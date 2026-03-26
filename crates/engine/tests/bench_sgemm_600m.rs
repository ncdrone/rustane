//! Sweep 4: GPU backward viability — benchmark CPU sgemm at 600M tensor sizes
//!
//! Measures cblas_sgemm (Accelerate) at the exact matrix sizes used in 579M backward pass.
//! Compare with Metal GPU results (from MLX benchmark) to decide if GPU backward is worth it.
//!
//! Run: cargo test -p engine --test bench_sgemm_600m --release -- --ignored --nocapture

use engine::cpu::vdsp;
use std::time::Instant;

fn bench_sgemm(label: &str, m: usize, n: usize, k: usize, reps: usize) {
    let a = vec![0.01f32; m * k];
    let b = vec![0.01f32; n * k];
    let mut c = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..3 {
        unsafe {
            vdsp::cblas_sgemm(
                101,
                111,
                111, // row-major, no-trans, no-trans
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }

    // Timed
    let t0 = Instant::now();
    for _ in 0..reps {
        unsafe {
            vdsp::cblas_sgemm(
                101,
                111,
                111,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }
    let elapsed = t0.elapsed().as_secs_f64() / reps as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / elapsed / 1e9;
    let ms = elapsed * 1000.0;

    println!("  {label:<45} {ms:>7.3}ms  {gflops:>6.1} GFLOP/s  ({m}×{k})@({k}×{n})→({m}×{n})");
}

#[test]
#[ignore]
fn sweep4_sgemm_600m() {
    println!("\n=== Sweep 4: CPU sgemm at 600M Backward Sizes ===");
    println!("  Apple Accelerate (cblas_sgemm) on M4 Max P-cores\n");

    // 579M config: dim=1536, hidden=4096, heads=12, hd=128, seq=256, nlayers=20
    let dim = 1536;
    let hidden = 4096;
    let seq = 256;
    let hd = 128;
    let heads = 12;

    println!("── FFN backward (per layer, per microbatch) ──");
    // dW1 = x^T @ dh1: [dim, seq] @ [seq, hidden] → [dim, hidden]
    bench_sgemm("dW1: x^T @ dh1", dim, hidden, seq, 20);
    // dW3 = x^T @ dh3: same shape
    bench_sgemm("dW3: x^T @ dh3", dim, hidden, seq, 20);
    // dW2 = gate^T @ dx: [hidden, seq] @ [seq, dim] → [hidden, dim]
    bench_sgemm("dW2: gate^T @ dx", hidden, dim, seq, 20);
    // dx = dh @ W1: [seq, hidden] @ [hidden, dim] → [seq, dim]  (activation grad)
    bench_sgemm("dx_ffn: dh @ W1", seq, dim, hidden, 20);

    println!("\n── Attention backward (per layer, per microbatch) ──");
    // dWq = x^T @ dq: [dim, seq] @ [seq, dim] → [dim, dim]
    bench_sgemm("dWq: x^T @ dq", dim, dim, seq, 20);
    // dWk = x^T @ dk: same
    bench_sgemm("dWk: x^T @ dk", dim, dim, seq, 20);
    // dWv = x^T @ dv: same
    bench_sgemm("dWv: x^T @ dv", dim, dim, seq, 20);
    // dWo = attn_out^T @ dy: [dim, seq] @ [seq, dim] → [dim, dim]
    bench_sgemm("dWo: attn^T @ dy", dim, dim, seq, 20);
    // dx_attn = dy @ Wo: [seq, dim] @ [dim, dim] → [seq, dim]
    bench_sgemm("dx_attn: dy @ Wo", seq, dim, dim, 20);

    println!("\n── Logits (embedding projection) ──");
    // dembed = dl^T @ x_final: [vocab, seq] @ [seq, dim] → [vocab, dim]
    bench_sgemm("dembed: dl^T @ x", 8192, dim, seq, 10);
    // dx_final = embed^T @ dl: [dim, vocab] @ [vocab, seq] → [dim, seq]
    bench_sgemm("dx_final: embed^T @ dl", dim, seq, 8192, 10);

    println!("\n── Per-step totals (20 layers × 1 microbatch) ──");
    // Sum up per-layer costs
    let reps = 10;
    let ops: Vec<(usize, usize, usize, usize)> = vec![
        // FFN: 3 dW + 1 dx per layer
        (dim, hidden, seq, 20), // dW1
        (dim, hidden, seq, 20), // dW3
        (hidden, dim, seq, 20), // dW2
        (seq, dim, hidden, 20), // dx_ffn
        // Attention: 4 dW + 1 dx per layer
        (dim, dim, seq, 20), // dWq
        (dim, dim, seq, 20), // dWk
        (dim, dim, seq, 20), // dWv
        (dim, dim, seq, 20), // dWo
        (seq, dim, dim, 20), // dx_attn
    ];

    let mut total_ms = 0.0;
    let mut total_flops = 0.0;
    for (m, n, k, nlayers) in &ops {
        let a = vec![0.01f32; m * k];
        let b = vec![0.01f32; n * k];
        let mut c_buf = vec![0.0f32; m * n];
        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe {
                vdsp::cblas_sgemm(
                    101,
                    111,
                    111,
                    *m as i32,
                    *n as i32,
                    *k as i32,
                    1.0,
                    a.as_ptr(),
                    *k as i32,
                    b.as_ptr(),
                    *n as i32,
                    0.0,
                    c_buf.as_mut_ptr(),
                    *n as i32,
                );
            }
        }
        let ms_per = t0.elapsed().as_secs_f64() / reps as f64 * 1000.0;
        let flops = 2.0 * *m as f64 * *n as f64 * *k as f64;
        total_ms += ms_per * *nlayers as f64;
        total_flops += flops * *nlayers as f64;
    }

    // Add logits
    let logit_ops: Vec<(usize, usize, usize)> = vec![
        (8192, dim, seq), // dembed
        (dim, seq, 8192), // dx_final
    ];
    for (m, n, k) in &logit_ops {
        let a = vec![0.01f32; m * k];
        let b = vec![0.01f32; n * k];
        let mut c_buf = vec![0.0f32; m * n];
        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe {
                vdsp::cblas_sgemm(
                    101,
                    111,
                    111,
                    *m as i32,
                    *n as i32,
                    *k as i32,
                    1.0,
                    a.as_ptr(),
                    *k as i32,
                    b.as_ptr(),
                    *n as i32,
                    0.0,
                    c_buf.as_mut_ptr(),
                    *n as i32,
                );
            }
        }
        let ms_per = t0.elapsed().as_secs_f64() / reps as f64 * 1000.0;
        let flops = 2.0 * *m as f64 * *n as f64 * *k as f64;
        total_ms += ms_per;
        total_flops += flops;
    }

    let total_gflops = total_flops / (total_ms / 1000.0) / 1e9;
    println!("  Total backward sgemm: {total_ms:.1}ms ({total_gflops:.0} GFLOP/s effective)");
    println!("  Current step time:    710ms (579M, accum=1)");
    println!("  Backward fraction:    {:.0}%", total_ms / 710.0 * 100.0);
    println!();
    println!("  M4 Max GPU fp16:      ~15 TFLOP/s (measured)");
    println!(
        "  If GPU backward:      {:.1}ms (at 15 TFLOP/s fp16)",
        total_flops / 15e12 * 1000.0
    );
    println!(
        "  Speedup potential:    {:.1}x",
        total_ms / (total_flops / 15e12 * 1000.0)
    );
}
