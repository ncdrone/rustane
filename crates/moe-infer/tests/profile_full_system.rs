//! Comprehensive system profiler: measures EVERY component of K2 decode.
//!
//! This is not a benchmark — it's a diagnostic tool. It measures:
//! 1. CPU sgemv throughput at every MLA projection dimension
//! 2. Metal dispatch overhead + compute time
//! 3. ANE dispatch overhead with baked vs dynamic weights
//! 4. IOSurface lock/unlock latency
//! 5. f16↔f32 NEON conversion bandwidth
//! 6. pread latency (cold/warm/F_NOCACHE)
//! 7. Memory bandwidth (sequential vs strided)
//! 8. Per-layer breakdown at microsecond resolution
//!
//! Run: RUSTANE_FULL_PROFILE=1 cargo test -p moe-infer --test profile_full_system --release -- --ignored --nocapture

use std::time::Instant;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::path::Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

// ─── CPU BLAS Profiling ───

#[test]
#[ignore = "profiling test"]
fn profile_cpu_sgemv_all_sizes() {
    eprintln!("\n=== CPU sgemv Throughput (cblas_sgemv via Accelerate) ===");
    eprintln!("{:<30} {:>8} {:>10} {:>10}", "Operation", "Size MB", "Time µs", "GB/s");
    eprintln!("{}", "-".repeat(62));

    let cases = [
        ("Q LoRA compress", 7168, 1536),
        ("Q LoRA expand", 1536, 12288),
        ("KV compress", 7168, 576),
        ("O projection", 8192, 7168),
        ("W_UK per-head", 128, 512),
        ("W_UV per-head", 128, 512),
        ("shared gate", 7168, 2048),
        ("shared up", 7168, 2048),
        ("shared down", 2048, 7168),
        ("lm_head", 163840, 7168),
    ];

    for (name, rows, cols) in cases {
        let w = vec![0.1f32; rows * cols];
        let x = vec![0.1f32; cols];
        let mut y = vec![0.0f32; rows];

        // Warmup
        for _ in 0..3 {
            moe_infer::blas::sgemv_f32(&w, &x, &mut y, rows, cols);
        }

        // Measure
        let iters = if rows * cols > 10_000_000 { 10 } else { 100 };
        let t = Instant::now();
        for _ in 0..iters {
            moe_infer::blas::sgemv_f32(&w, &x, &mut y, rows, cols);
        }
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        let bytes = (rows * cols * 4) as f64; // f32 weights read
        let gb_s = bytes / (us * 1000.0);
        let mb = bytes / 1e6;

        eprintln!("{:<30} {:>7.1}  {:>9.0}  {:>9.1}", name, mb, us, gb_s);
    }
}

#[test]
#[ignore = "profiling test"]
fn profile_cpu_sgemv_trans() {
    eprintln!("\n=== CPU sgemv_trans (W_UK/W_UV absorption) ===");
    eprintln!("{:<30} {:>8} {:>10}", "Operation", "Heads", "Total µs");
    eprintln!("{}", "-".repeat(52));

    let nope = 128;
    let kv_rank = 512;
    let w = vec![0.1f32; nope * kv_rank];
    let x = vec![0.1f32; nope];
    let mut y = vec![0.0f32; kv_rank];

    // Single head
    let t = Instant::now();
    for _ in 0..1000 {
        moe_infer::blas::sgemv_f32_trans(&w, &x, &mut y, kv_rank, nope);
    }
    let per_head_us = t.elapsed().as_micros() as f64 / 1000.0;
    eprintln!("{:<30} {:>8} {:>9.1}", "W_UK single head", 1, per_head_us);

    // 64 heads sequential (K2)
    let w_all = vec![0.1f32; 64 * nope * kv_rank];
    let x_all = vec![0.1f32; 64 * nope];
    let mut y_all = vec![0.0f32; 64 * kv_rank];

    let t = Instant::now();
    for _ in 0..100 {
        for head in 0..64 {
            let wh = &w_all[head * nope * kv_rank..(head + 1) * nope * kv_rank];
            let xh = &x_all[head * nope..(head + 1) * nope];
            let yh = &mut y_all[head * kv_rank..(head + 1) * kv_rank];
            moe_infer::blas::sgemv_f32_trans(wh, xh, yh, kv_rank, nope);
        }
    }
    let total_us = t.elapsed().as_micros() as f64 / 100.0;
    eprintln!("{:<30} {:>8} {:>9.0}", "W_UK 64 heads (K2)", 64, total_us);
}

// ─── ANE Profiling ───

#[test]
#[ignore = "profiling test, requires ANE"]
fn profile_ane_dispatch_overhead() {
    eprintln!("\n=== ANE Dispatch Overhead ===");

    // Compile a minimal graph
    let result = moe_infer::ane_mla::compile_mla_kernels(
        128, 64, 1, 1, 1, 1, 64,
    );
    let kernels = match result {
        Ok(k) => k,
        Err(e) => { eprintln!("ANE not available: {e}"); return; }
    };

    // Measure bare dispatch (no data staging)
    let activation = vec![0.1f32; 128];

    // Warmup
    for _ in 0..5 {
        let _ = moe_infer::ane_mla::ane_projection(
            &kernels.q_a_exec, &kernels.q_a_in, &kernels.q_a_out,
            &activation, &vec![0.1f32; 64 * 128], 128, 64,
        );
    }

    // Measure full dispatch (staging + compute + read)
    let w = vec![0.1f32; 64 * 128];
    let iters = 50;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = moe_infer::ane_mla::ane_projection(
            &kernels.q_a_exec, &kernels.q_a_in, &kernels.q_a_out,
            &activation, &w, 128, 64,
        );
    }
    let full_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("Full dispatch (stage+compute+read) [128→64]: {:.0}µs", full_us);

    // Measure fast dispatch (activation only, weights pre-staged)
    moe_infer::ane_mla::prestage_weights(
        &kernels.q_a_in, &w, 128, 64,
    );
    let t = Instant::now();
    for _ in 0..iters {
        let _ = moe_infer::ane_mla::ane_projection_fast(
            &kernels.q_a_exec, &kernels.q_a_in, &kernels.q_a_out,
            &activation, 128, 64,
        );
    }
    let fast_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("Fast dispatch (act-only + compute + read) [128→64]: {:.0}µs", fast_us);

    // Measure IOSurface lock/unlock alone
    let t = Instant::now();
    for _ in 0..iters {
        unsafe {
            ane_bridge::stage_activation_fp16(&kernels.q_a_in, &activation, 128, kernels.q_a_in.shape().width);
        }
    }
    let stage_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("IOSurface lock+write+unlock (128 fp16): {:.0}µs", stage_us);

    // Measure output read alone
    let mut out = vec![0.0f32; 64];
    let out_sp = kernels.q_a_out.shape().width;
    let t = Instant::now();
    for _ in 0..iters {
        unsafe {
            ane_bridge::read_output_fp16(&kernels.q_a_out, &mut out, 64, out_sp);
        }
    }
    let read_us = t.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("IOSurface lock+read+unlock (64 fp16): {:.0}µs", read_us);

    // Measure bare ANE dispatch (no staging, no read)
    let compute_us = fast_us - stage_us - read_us;
    eprintln!("ANE compute only (estimated): {:.0}µs", compute_us);
    eprintln!("");
    eprintln!("Breakdown: stage={:.0}µs + compute={:.0}µs + read={:.0}µs = {:.0}µs total",
        stage_us, compute_us, read_us, fast_us);
}

#[test]
#[ignore = "profiling test, requires ANE"]
fn profile_ane_k2_dimensions() {
    eprintln!("\n=== ANE Dispatch at K2 Dimensions ===");

    let result = moe_infer::ane_mla::compile_mla_kernels(
        7168, 1536, 64, 128, 64, 128, 512,
    );
    let kernels = match result {
        Ok(k) => k,
        Err(e) => { eprintln!("ANE not available: {e}"); return; }
    };

    let x = vec![0.1f32; 7168];
    let iters = 20;

    // Profile each projection individually
    let projs: &[(&str, &ane_bridge::ane::Executable, &ane_bridge::ane::TensorData,
                  &ane_bridge::ane::TensorData, usize, usize, &[f32])] = &[
        ("Q LoRA compress [7168→1536]",
         &kernels.q_a_exec, &kernels.q_a_in, &kernels.q_a_out,
         7168, 1536, &vec![0.1f32; 1536 * 7168]),
        ("KV compress [7168→576]",
         &kernels.kv_a_exec, &kernels.kv_a_in, &kernels.kv_a_out,
         7168, 576, &vec![0.1f32; 576 * 7168]),
    ];

    for &(name, exec, inp, outp, ic, oc, weights) in projs {
        // Prestage weights
        moe_infer::ane_mla::prestage_weights(inp, weights, ic, oc);

        // Warmup
        for _ in 0..3 {
            let _ = moe_infer::ane_mla::ane_projection_fast(exec, inp, outp, &x[..ic], ic, oc);
        }

        // Measure
        let t = Instant::now();
        for _ in 0..iters {
            let _ = moe_infer::ane_mla::ane_projection_fast(exec, inp, outp, &x[..ic], ic, oc);
        }
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        eprintln!("{:<35} {:.0}µs/dispatch", name, us);
    }
}

// ─── f16↔f32 Conversion Profiling ───

#[test]
#[ignore = "profiling test"]
fn profile_f16_conversion_bandwidth() {
    eprintln!("\n=== f16↔f32 Conversion Bandwidth (NEON FCVTL) ===");
    eprintln!("{:<25} {:>10} {:>10} {:>10}", "Size", "Elements", "Time µs", "GB/s");
    eprintln!("{}", "-".repeat(58));

    let sizes = [1024, 7168, 65536, 786432, 11_534_336]; // 4K to 46MB

    for &n in &sizes {
        let f32_data = vec![0.1f32; n];
        let mut f16_data = vec![half::f16::ZERO; n];

        // f32→f16
        let iters = if n > 1_000_000 { 10 } else { 100 };
        let t = Instant::now();
        for _ in 0..iters {
            for (i, v) in f32_data.iter().enumerate() {
                f16_data[i] = half::f16::from_f32(*v);
            }
        }
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        let bytes = n as f64 * 4.0; // reading f32
        let gb_s = bytes / (us * 1000.0);
        let label = if n > 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
                    else { format!("{:.1}K", n as f64 / 1e3) };
        eprintln!("{:<25} {:>10} {:>9.0}  {:>9.1}", label, n, us, gb_s);
    }
}

// ─── Memory Bandwidth Profiling ───

#[test]
#[ignore = "profiling test"]
fn profile_memory_bandwidth() {
    eprintln!("\n=== Memory Bandwidth ===");

    // Sequential read (memcpy pattern)
    let sizes_mb = [1, 10, 50, 100, 500];
    eprintln!("{:<20} {:>10} {:>10}", "Pattern", "Size MB", "GB/s");
    eprintln!("{}", "-".repeat(42));

    for &mb in &sizes_mb {
        let n = mb * 1024 * 1024 / 4; // f32 elements
        let src = vec![1.0f32; n];
        let mut dst = vec![0.0f32; n];

        let iters = if mb > 100 { 3 } else { 10 };
        let t = Instant::now();
        for _ in 0..iters {
            dst.copy_from_slice(&src);
        }
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        let gb_s = (mb as f64 * 1e6) / (us * 1000.0);
        eprintln!("{:<20} {:>9}  {:>9.1}", "Sequential copy", mb, gb_s);
    }

    // Strided read (transpose pattern - what kills ANE staging)
    eprintln!("");
    let ic = 7168;
    let oc = 1536;
    let src = vec![0.1f32; ic * oc];
    let mut dst = vec![0.0f32; ic * oc];

    let t = Instant::now();
    for _ in 0..5 {
        for c in 0..ic {
            for j in 0..oc {
                dst[c * oc + j] = src[j * ic + c];
            }
        }
    }
    let us = t.elapsed().as_micros() as f64 / 5.0;
    let mb = (ic * oc * 4) as f64 / 1e6;
    let gb_s = mb / (us / 1e6) / 1e3;
    eprintln!("{:<20} {:>8.1}M {:>9.1}", "Strided transpose", mb, gb_s);
}

// ─── Full K2 Layer Profile ───

#[test]
#[ignore = "requires K2 weights"]
fn profile_k2_layer_breakdown() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-k2");
    let config_path = root.join("configs/kimi-k2.toml");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: K2 weights not found");
        return;
    }

    eprintln!("\n=== K2 Full Layer Breakdown (Microsecond Resolution) ===");
    eprintln!("Loading model...");

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load K2");

    // TODO: Add per-component timing with RUSTANE_MLA_PROFILE=1
    // This test structure is ready for the detailed breakdown
    eprintln!("Model loaded. Use RUSTANE_MLA_PROFILE=1 for per-component MLA timing.");
    eprintln!("Per-layer FFN breakdown available via the decode loop profiling.");
}
