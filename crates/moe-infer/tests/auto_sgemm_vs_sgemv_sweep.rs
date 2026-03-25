//! Sweep ALL K2 projections: sgemm M=1 vs sgemv.
//! The O projection showed sgemm M=1 is 17% faster.
//! Are there more wins at other dimensions?

use std::time::Instant;

#[test]
#[ignore = "profiling test"]
fn sgemm_vs_sgemv_all_projections() {
    eprintln!("\n=== sgemm M=1 vs sgemv — ALL K2 Projections ===\n");
    eprintln!("{:<30} {:>8} {:>8} {:>8} {:>6}", "Projection", "sgemv", "sgemm", "winner", "gain");
    eprintln!("{}", "-".repeat(65));

    let projs = [
        ("Q LoRA compress", 1536, 7168),     // [q_lora_rank, hidden]
        ("Q LoRA expand", 12288, 1536),       // [q_total, q_lora_rank]
        ("KV compress", 576, 7168),           // [kv_out_dim, hidden]
        ("O projection", 7168, 8192),         // [hidden, v_total]
        ("W_UK single head", 512, 128),       // [kv_rank, nope] via trans
        ("shared gate", 2048, 7168),          // [inter, hidden]
        ("shared up", 2048, 7168),            // [inter, hidden]
        ("shared down", 7168, 2048),          // [hidden, inter]
        ("lm_head (subset)", 4096, 7168),     // truncated for speed
    ];

    let mut total_sgemv = 0.0f64;
    let mut total_sgemm = 0.0f64;

    for (name, rows, cols) in projs {
        let w = vec![0.1f32; rows * cols];
        let x = vec![0.1f32; cols];
        let mut y = vec![0.0f32; rows];

        let iters = if rows * cols > 5_000_000 { 20 } else { 50 };

        // sgemv
        for _ in 0..3 { moe_infer::blas::sgemv_f32(&w, &x, &mut y, rows, cols); }
        let t = Instant::now();
        for _ in 0..iters { moe_infer::blas::sgemv_f32(&w, &x, &mut y, rows, cols); }
        let sgemv_us = t.elapsed().as_micros() as f64 / iters as f64;

        // sgemm M=1
        for _ in 0..3 { moe_infer::blas::sgemm_custom_1xn(&x, &w, &mut y, cols, rows); }
        let t = Instant::now();
        for _ in 0..iters { moe_infer::blas::sgemm_custom_1xn(&x, &w, &mut y, cols, rows); }
        let sgemm_us = t.elapsed().as_micros() as f64 / iters as f64;

        let winner = if sgemm_us < sgemv_us { "sgemm" } else { "sgemv" };
        let gain = ((sgemv_us / sgemm_us) - 1.0) * 100.0;
        let marker = if gain > 5.0 { "★" } else { "" };

        eprintln!("{:<30} {:>7.0}  {:>7.0}  {:>7} {:>5.0}% {}",
            name, sgemv_us, sgemm_us, winner, gain, marker);

        total_sgemv += sgemv_us;
        total_sgemm += sgemm_us;
    }

    eprintln!("{}", "-".repeat(65));
    eprintln!("{:<30} {:>7.0}  {:>7.0}  {:>7} {:>5.0}%",
        "TOTAL", total_sgemv, total_sgemm,
        if total_sgemm < total_sgemv { "sgemm" } else { "sgemv" },
        ((total_sgemv / total_sgemm) - 1.0) * 100.0);

    eprintln!("\n× 61 layers:");
    eprintln!("  sgemv total: {:.1}ms", total_sgemv * 61.0 / 1000.0);
    eprintln!("  sgemm total: {:.1}ms", total_sgemm * 61.0 / 1000.0);
    eprintln!("  Savings: {:.1}ms/token", (total_sgemv - total_sgemm) * 61.0 / 1000.0);
}
