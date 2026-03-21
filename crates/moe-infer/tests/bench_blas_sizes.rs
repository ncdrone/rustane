//! BLAS performance at inference-relevant dimensions.

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

#[test]
fn bench_blas_sizes() {
    use moe_infer::blas::{sgemv_f32, sgemm};

    let iters = 200;

    // sgemv at Qwen3-MoE-30B inference dims
    let cases_v: Vec<(&str, usize, usize)> = vec![
        ("Q proj [4096,2048]", 4096, 2048),
        ("K proj [512,2048]", 512, 2048),
        ("V proj [512,2048]", 512, 2048),
        ("O proj [2048,4096]", 2048, 4096),
        ("router [128,2048]", 128, 2048),
    ];

    eprintln!("\n=== SGEMV (matrix-vector) ===");
    eprintln!("{:<25} {:>8} {:>8}", "operation", "µs", "GB/s");
    for (name, rows, cols) in &cases_v {
        let w: Vec<f32> = vec![0.01f32; rows * cols];
        let x: Vec<f32> = vec![0.01f32; *cols];
        let mut y = vec![0.0f32; *rows];

        // Warmup
        for _ in 0..10 { sgemv_f32(&w, &x, &mut y, *rows, *cols); }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            sgemv_f32(&w, &x, &mut y, *rows, *cols);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        let bytes = (rows * cols * 4 + cols * 4 + rows * 4) as f64;
        let gbps = bytes / (med * 1e-6) / 1e9;
        eprintln!("{:<25} {:>8.0} {:>8.1}", name, med, gbps);
    }

    // sgemm for batched O_proj (prefill)
    let cases_m: Vec<(&str, usize, usize, usize)> = vec![
        ("O_proj batch [2048,4096]×[4096,13]", 2048, 13, 4096),
        ("O_proj batch [2048,4096]×[4096,32]", 2048, 32, 4096),
        ("O_proj batch [2048,4096]×[4096,64]", 2048, 64, 4096),
    ];

    eprintln!("\n=== SGEMM (matrix-matrix) ===");
    eprintln!("{:<40} {:>8} {:>8}", "operation", "µs", "GFLOPS");
    for (name, m, n, k) in &cases_m {
        let a: Vec<f32> = vec![0.01f32; m * k];
        let b: Vec<f32> = vec![0.01f32; k * n];
        let mut c = vec![0.0f32; m * n];

        // Warmup
        for _ in 0..10 { sgemm(&a, &b, &mut c, *m, *n, *k); }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            sgemm(&a, &b, &mut c, *m, *n, *k);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        let flops = 2.0 * (*m as f64) * (*n as f64) * (*k as f64);
        let gflops = flops / (med * 1e-6) / 1e9;
        eprintln!("{:<40} {:>8.0} {:>8.1}", name, med, gflops);
    }

    // Compare: 13 × sgemv vs 1 × sgemm for O_proj
    {
        let rows = 2048;
        let cols = 4096;
        let batch = 13;
        let w: Vec<f32> = vec![0.01f32; rows * cols];

        // 13 × sgemv
        let x_vecs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.01f32; cols]).collect();
        let mut y_vecs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; rows]).collect();

        // Warmup
        for _ in 0..5 {
            for i in 0..batch {
                sgemv_f32(&w, &x_vecs[i], &mut y_vecs[i], rows, cols);
            }
        }

        let mut times_v = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            for i in 0..batch {
                sgemv_f32(&w, &x_vecs[i], &mut y_vecs[i], rows, cols);
            }
            times_v.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med_v = median(&mut times_v);

        // 1 × sgemm (need to pack x_vecs as [cols, batch] column-major = [batch, cols] transposed)
        // Actually sgemm C = W @ X where W=[rows,cols], X=[cols,batch] → C=[rows,batch]
        // But our sgemm is C = A @ B, A=[m,k], B=[k,n], C=[m,n]
        // So A=W [rows,cols], B=X_packed [cols,batch], C=[rows,batch]
        let mut x_packed = vec![0.0f32; cols * batch]; // [cols, batch] row-major
        for i in 0..batch {
            for j in 0..cols {
                x_packed[j * batch + i] = x_vecs[i][j];
            }
        }
        let mut c = vec![0.0f32; rows * batch];

        // Warmup
        for _ in 0..5 { sgemm(&w, &x_packed, &mut c, rows, batch, cols); }

        let mut times_m = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            sgemm(&w, &x_packed, &mut c, rows, batch, cols);
            times_m.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med_m = median(&mut times_m);

        eprintln!("\n=== O_PROJ COMPARISON (batch={}) ===", batch);
        eprintln!("  {}× sgemv: {:.0} µs", batch, med_v);
        eprintln!("  1× sgemm:  {:.0} µs", med_m);
        eprintln!("  speedup:   {:.1}x", med_v / med_m);
    }
}
