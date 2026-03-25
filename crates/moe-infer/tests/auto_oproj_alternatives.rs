//! Profile O projection alternatives: find the fastest path for [8192→7168].
//!
//! O projection is 60% of MLA time (1665µs at 141 GB/s).
//! Other sgemv ops hit 200-361 GB/s. Why is O proj slower?
//! Test every available path to find the fastest.

use std::time::Instant;

fn measure<F: FnMut()>(name: &str, mut f: F, iters: usize) -> f64 {
    // Warmup
    for _ in 0..3 { f(); }
    let t = Instant::now();
    for _ in 0..iters { f(); }
    let us = t.elapsed().as_micros() as f64 / iters as f64;
    let mb = (8192 * 7168 * 4) as f64 / 1e6;
    let gb_s = mb / (us / 1e6) / 1e9;
    eprintln!("  {:<40} {:>7.0}µs  {:>6.1} GB/s", name, us, gb_s);
    us
}

#[test]
#[ignore = "profiling test"]
fn profile_oproj_all_paths() {
    eprintln!("\n=== O Projection [8192→7168] — All Paths ===\n");

    let rows = 7168;  // hidden (output dim)
    let cols = 8192;  // v_total (input dim)

    // Weight matrix [rows, cols] = [7168, 8192] row-major
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 13 + 7) % 200) as f32 / 1000.0 - 0.1).collect();
    let x: Vec<f32> = (0..cols).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
    let mut y = vec![0.0f32; rows];

    let iters = 20;

    // Path 1: sgemv_f32 (current path)
    measure("sgemv_f32 [7168,8192] NoTrans", || {
        moe_infer::blas::sgemv_f32(&w, &x, &mut y, rows, cols);
    }, iters);

    // Path 2: sgemv_f32_trans — transpose the matrix
    let mut wt = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            wt[j * rows + i] = w[i * cols + j];
        }
    }
    measure("sgemv_f32_trans [8192,7168] Trans", || {
        moe_infer::blas::sgemv_f32_trans(&wt, &x, &mut y, rows, cols);
    }, iters);

    // Path 3: sgemm with M=1 (might use different AMX path)
    let mut y_sgemm = vec![0.0f32; rows];
    measure("sgemm M=1 K=8192 N=7168 (NoTrans)", || {
        // Treat as: [1, 8192] × [8192, 7168] → [1, 7168]
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 7168, 8192, ...)
        moe_infer::blas::sgemm_custom_1xn(&x, &w, &mut y_sgemm, cols, rows);
    }, iters);

    // Path 4: sgemm with transB
    measure("sgemm M=1 K=8192 N=7168 (TransB)", || {
        moe_infer::blas::sgemm_custom_1xn_transb(&x, &wt, &mut y_sgemm, cols, rows);
    }, iters);
}
