//! Correctness + micro-benchmark: manual vectorized loops vs per-head BLAS
//! for MLA Q absorption (W_UK) and V computation (W_UV).
//!
//! What was optimized: per-head sgemv_f32_trans/sgemv_f32 calls replaced with
//! manual loops that auto-vectorize to NEON fmla. Avoids ~5μs BLAS dispatch
//! overhead per call × 128 heads × 2 ops = ~1.3ms/layer savings.

/// Verify manual Q absorption matches BLAS for V3 dimensions.
#[test]
fn q_absorption_manual_matches_blas() {
    let h = 128;      // V3 heads
    let nope = 128;    // qk_nope_head_dim
    let kv_rank = 512; // kv_lora_rank

    // Deterministic test data
    let q_nope: Vec<f32> = (0..h * nope).map(|i| (i as f32 * 0.001).sin()).collect();
    let w_uk: Vec<f32> = (0..h * nope * kv_rank).map(|i| (i as f32 * 0.0001).cos()).collect();

    // BLAS path
    let mut blas_result = vec![0.0f32; h * kv_rank];
    for head in 0..h {
        let q_head = &q_nope[head * nope..(head + 1) * nope];
        let w_uk_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
        let out = &mut blas_result[head * kv_rank..(head + 1) * kv_rank];
        moe_infer::blas::sgemv_f32_trans(w_uk_head, q_head, out, kv_rank, nope);
    }

    // Manual path
    let mut manual_result = vec![0.0f32; h * kv_rank];
    for head in 0..h {
        let q_head = &q_nope[head * nope..(head + 1) * nope];
        let w_uk_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
        let out = &mut manual_result[head * kv_rank..(head + 1) * kv_rank];
        for n in 0..nope {
            let q_n = q_head[n];
            let w_row = &w_uk_head[n * kv_rank..(n + 1) * kv_rank];
            for k in 0..kv_rank {
                out[k] += q_n * w_row[k];
            }
        }
    }

    // Allow small precision differences (f32 accumulation order may differ)
    let max_diff = blas_result.iter().zip(manual_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "Q absorption max_diff={max_diff} (must be <1e-3)");
}

/// Verify manual W_UV computation matches BLAS for V3 dimensions.
#[test]
fn w_uv_manual_matches_blas() {
    let h = 128;
    let v_dim = 128;
    let kv_rank = 512;

    let w_uv: Vec<f32> = (0..h * v_dim * kv_rank).map(|i| (i as f32 * 0.0001).sin()).collect();
    let v_latents: Vec<f32> = (0..h * kv_rank).map(|i| (i as f32 * 0.001).cos()).collect();

    // BLAS path
    let mut blas_result = vec![0.0f32; h * v_dim];
    for head in 0..h {
        let w_uv_head = &w_uv[head * v_dim * kv_rank..(head + 1) * v_dim * kv_rank];
        let v_lat = &v_latents[head * kv_rank..(head + 1) * kv_rank];
        let v_out = &mut blas_result[head * v_dim..(head + 1) * v_dim];
        moe_infer::blas::sgemv_f32(w_uv_head, v_lat, v_out, v_dim, kv_rank);
    }

    // Manual path
    let mut manual_result = vec![0.0f32; h * v_dim];
    for head in 0..h {
        let w_uv_head = &w_uv[head * v_dim * kv_rank..(head + 1) * v_dim * kv_rank];
        let v_lat = &v_latents[head * kv_rank..(head + 1) * kv_rank];
        let v_out = &mut manual_result[head * v_dim..(head + 1) * v_dim];
        for m in 0..v_dim {
            let mut sum = 0.0f32;
            let w_row = &w_uv_head[m * kv_rank..(m + 1) * kv_rank];
            for k in 0..kv_rank {
                sum += w_row[k] * v_lat[k];
            }
            v_out[m] = sum;
        }
    }

    let max_diff = blas_result.iter().zip(manual_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "W_UV max_diff={max_diff} (must be <1e-3)");
}

/// Benchmark: 128-head BLAS vs manual for Q absorption (V3 dimensions).
#[test]
fn bench_q_absorption_blas_vs_manual() {
    let h = 128;
    let nope = 128;
    let kv_rank = 512;
    let iters = 100;

    let q_nope: Vec<f32> = (0..h * nope).map(|i| (i as f32 * 0.001).sin()).collect();
    let w_uk: Vec<f32> = (0..h * nope * kv_rank).map(|i| (i as f32 * 0.0001).cos()).collect();

    // BLAS
    let mut result = vec![0.0f32; h * kv_rank];
    let t = std::time::Instant::now();
    for _ in 0..iters {
        result.fill(0.0);
        for head in 0..h {
            let q_head = &q_nope[head * nope..(head + 1) * nope];
            let w_uk_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
            let out = &mut result[head * kv_rank..(head + 1) * kv_rank];
            moe_infer::blas::sgemv_f32_trans(w_uk_head, q_head, out, kv_rank, nope);
        }
        std::hint::black_box(&result);
    }
    let blas_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Manual
    let t = std::time::Instant::now();
    for _ in 0..iters {
        result.fill(0.0);
        for head in 0..h {
            let q_head = &q_nope[head * nope..(head + 1) * nope];
            let w_uk_head = &w_uk[head * nope * kv_rank..(head + 1) * nope * kv_rank];
            let out = &mut result[head * kv_rank..(head + 1) * kv_rank];
            for n in 0..nope {
                let q_n = q_head[n];
                let w_row = &w_uk_head[n * kv_rank..(n + 1) * kv_rank];
                for k in 0..kv_rank {
                    out[k] += q_n * w_row[k];
                }
            }
        }
        std::hint::black_box(&result);
    }
    let manual_us = t.elapsed().as_micros() as f64 / iters as f64;

    let speedup = blas_us / manual_us;
    eprintln!("\n=== Q absorption (128 heads, [128,512]) ===");
    eprintln!("  BLAS:   {blas_us:.0}μs");
    eprintln!("  Manual: {manual_us:.0}μs");
    eprintln!("  Speedup: {speedup:.2}x");
}
