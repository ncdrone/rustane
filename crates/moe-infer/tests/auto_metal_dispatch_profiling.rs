//! Profile Metal dispatch overhead: how much of the 3ms fused kernel is compute vs overhead?
//! And: how does Metal compare to CPU for the shared FFN projections?

use std::time::Instant;

#[test]
#[ignore = "profiling test"]
fn profile_shared_ffn_cpu_paths() {
    eprintln!("\n=== Shared FFN: sgemv vs sgemm vs batched sgemm ===\n");

    let hidden = 7168;
    let inter = 2048;
    let gate_w = vec![0.1f32; inter * hidden];
    let up_w = vec![0.1f32; inter * hidden];
    let down_w = vec![0.1f32; hidden * inter];
    let x = vec![0.1f32; hidden];

    let iters = 50;

    // Current: 3 sequential sgemm_custom_1xn calls
    let mut gate_out = vec![0.0f32; inter];
    let mut up_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; hidden];

    // Warmup
    for _ in 0..3 {
        moe_infer::blas::sgemm_custom_1xn(&x, &gate_w, &mut gate_out, hidden, inter);
        moe_infer::blas::sgemm_custom_1xn(&x, &up_w, &mut up_out, hidden, inter);
        for i in 0..inter {
            let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
            gate_out[i] = silu * up_out[i];
        }
        moe_infer::blas::sgemm_custom_1xn(&gate_out, &down_w, &mut out, inter, hidden);
    }

    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&x, &gate_w, &mut gate_out, hidden, inter);
        moe_infer::blas::sgemm_custom_1xn(&x, &up_w, &mut up_out, hidden, inter);
        for i in 0..inter {
            let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
            gate_out[i] = silu * up_out[i];
        }
        moe_infer::blas::sgemm_custom_1xn(&gate_out, &down_w, &mut out, inter, hidden);
    }
    let total_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("  Shared FFN (3×sgemm + SiLU): {:.0}µs", total_us);

    // Per-component
    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&x, &gate_w, &mut gate_out, hidden, inter);
    }
    let gate_us = t.elapsed().as_micros() as f64 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&x, &up_w, &mut up_out, hidden, inter);
    }
    let up_us = t.elapsed().as_micros() as f64 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        for i in 0..inter {
            let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
            gate_out[i] = silu * up_out[i];
        }
    }
    let silu_us = t.elapsed().as_micros() as f64 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&gate_out, &down_w, &mut out, inter, hidden);
    }
    let down_us = t.elapsed().as_micros() as f64 / iters as f64;

    eprintln!("    gate:  {:.0}µs", gate_us);
    eprintln!("    up:    {:.0}µs", up_us);
    eprintln!("    SiLU:  {:.0}µs", silu_us);
    eprintln!("    down:  {:.0}µs", down_us);
    eprintln!("    sum:   {:.0}µs", gate_us + up_us + silu_us + down_us);

    // Fused gate+up: can we do both in one sgemm?
    // gate_out = x @ gate_w^T, up_out = x @ up_w^T
    // Concatenate gate_w and up_w: [2*inter, hidden]
    let mut fused_w = vec![0.0f32; 2 * inter * hidden];
    fused_w[..inter * hidden].copy_from_slice(&gate_w);
    fused_w[inter * hidden..].copy_from_slice(&up_w);
    let mut fused_out = vec![0.0f32; 2 * inter];

    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&x, &fused_w, &mut fused_out, hidden, 2 * inter);
    }
    let fused_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("\n  Fused gate+up (single sgemm [7168→4096]): {:.0}µs vs separate: {:.0}µs",
        fused_us, gate_us + up_us);
    let gain = ((gate_us + up_us) / fused_us - 1.0) * 100.0;
    if gain > 0.0 {
        eprintln!("  ★ Fused is {:.0}% faster!", gain);
    } else {
        eprintln!("  Separate is {:.0}% faster", -gain);
    }
}

#[test]
#[ignore = "profiling test"]
fn profile_memory_access_patterns() {
    eprintln!("\n=== Memory Access Patterns ===\n");

    // Test: does touching the same memory region twice (gate + up on same x) benefit from cache?
    let hidden = 7168;
    let inter = 2048;
    let w1 = vec![0.1f32; inter * hidden];
    let w2 = vec![0.1f32; inter * hidden];
    let x = vec![0.1f32; hidden];
    let mut y1 = vec![0.0f32; inter];
    let mut y2 = vec![0.0f32; inter];

    let iters = 100;

    // Pattern A: w1 then w2 (different weight matrices, same x)
    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&x, &w1, &mut y1, hidden, inter);
        moe_infer::blas::sgemm_custom_1xn(&x, &w2, &mut y2, hidden, inter);
    }
    let pattern_a_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Pattern B: interleave (w1 chunk, w2 chunk, w1 chunk, w2 chunk)
    // Can't easily do this with BLAS, so just measure the separate total
    let t = Instant::now();
    for _ in 0..iters {
        moe_infer::blas::sgemm_custom_1xn(&x, &w2, &mut y2, hidden, inter);
        moe_infer::blas::sgemm_custom_1xn(&x, &w1, &mut y1, hidden, inter);
    }
    let pattern_b_us = t.elapsed().as_micros() as f64 / iters as f64;

    eprintln!("  w1→w2 order: {:.0}µs", pattern_a_us);
    eprintln!("  w2→w1 order: {:.0}µs", pattern_b_us);
    eprintln!("  Difference: {:.1}%", ((pattern_a_us / pattern_b_us) - 1.0) * 100.0);
    eprintln!("  (Should be ~0% if x stays in L1 cache between calls)");
}
