//! Profile W_UK absorption: 64 sequential sgemv_trans vs batched sgemm.
//!
//! Current: 64 × sgemv_f32_trans([128, 512]) = 143µs total.
//! Can we batch all 64 heads into a single sgemm call?

use std::time::Instant;

#[test]
#[ignore = "profiling test"]
fn profile_absorption_alternatives() {
    eprintln!("\n=== W_UK Absorption: Sequential vs Batched ===\n");

    let num_heads = 64;
    let nope = 128;
    let kv_rank = 512;

    // Full W_UK: [num_heads, nope, kv_rank] = [64, 128, 512]
    let w_uk: Vec<f32> = (0..num_heads * nope * kv_rank)
        .map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();

    // q_nope: [num_heads * nope] = [8192]
    let q_nope: Vec<f32> = (0..num_heads * nope)
        .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

    let mut q_absorbed = vec![0.0f32; num_heads * kv_rank]; // [64 * 512] = [32768]

    let iters = 200;

    // Path 1: Current — 64 sequential sgemv_f32_trans
    for _ in 0..5 {
        for head in 0..num_heads {
            let q_head = &q_nope[head * nope..(head + 1) * nope];
            let w_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
            let out = &mut q_absorbed[head * kv_rank..(head + 1) * kv_rank];
            moe_infer::blas::sgemv_f32_trans(w_head, q_head, out, kv_rank, nope);
        }
    }
    let t = Instant::now();
    for _ in 0..iters {
        for head in 0..num_heads {
            let q_head = &q_nope[head * nope..(head + 1) * nope];
            let w_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
            let out = &mut q_absorbed[head * kv_rank..(head + 1) * kv_rank];
            moe_infer::blas::sgemv_f32_trans(w_head, q_head, out, kv_rank, nope);
        }
    }
    let seq_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("  Sequential 64×sgemv_f32_trans: {:.0}µs", seq_us);

    // Path 2: Sequential sgemm M=1 (different AMX path)
    let t = Instant::now();
    for _ in 0..iters {
        for head in 0..num_heads {
            let q_head = &q_nope[head * nope..(head + 1) * nope];
            let w_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
            let out = &mut q_absorbed[head * kv_rank..(head + 1) * kv_rank];
            // sgemm: [1, nope] × [nope, kv_rank]^T → [1, kv_rank]
            // W_UK stored [nope, kv_rank], need trans
            moe_infer::blas::sgemm_custom_1xn_transb(q_head, w_head, out, nope, kv_rank);
        }
    }
    let sgemm_seq_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("  Sequential 64×sgemm_1xn_transb: {:.0}µs", sgemm_seq_us);

    // Path 3: Batched sgemm — reshape all heads into one big matmul
    // q_nope: [64, 128] → [64, 128] (already contiguous per head)
    // W_UK: [64, 128, 512] → need to reshape for batched mm
    // If we treat q as [64, 128] and W as [64*128, 512] (stacked):
    //   q_absorbed[h] = q_nope[h] @ W_UK[h]^T
    // This is a batch of 64 independent [1, 128] × [128, 512]^T = [1, 512]
    //
    // Can express as sgemm: [64, 128] × [stacked 128, 512]^T → [64, 512]
    // But only if W_UK is block-diagonal, which it is (each head independent).
    // Standard sgemm treats it as dense, so this won't work directly.
    //
    // Alternative: single sgemm_nt [64, 128] × [64*512, 128]? No, dimensions don't match.
    //
    // The real batched approach: q_absorbed = q_nope × W_UK_reshaped
    // where W_UK_reshaped[8192, 512] = block_diag(W_UK[0], ..., W_UK[63])
    // But block_diag is sparse, not dense — sgemm would waste 63/64 of compute.
    //
    // Conclusion: batched sgemm not viable for block-diagonal structure.
    // Keep sequential — the overhead is in BLAS function-call, not compute.
    eprintln!("  Batched sgemm: NOT VIABLE (block-diagonal W_UK)");

    eprintln!("\n  Winner: {}", if sgemm_seq_us < seq_us { "sgemm_1xn_transb" } else { "sgemv_trans" });
    eprintln!("  Ratio: {:.2}x", seq_us / sgemm_seq_us);
}
