//! Correctness test: o_proj on Metal GPU f32 GEMV vs CPU cblas_sgemv.
//!
//! What was optimized: o_proj [7168, 8192] sgemv moved from CPU (cblas_sgemv, ~273 GB/s)
//! to Metal GPU (f32_gemv shader, ~400 GB/s). Saves ~0.5ms/layer × 61 = ~30ms/token.
//!
//! Invariant: GPU f32 GEMV must match CPU cblas_sgemv within tolerance.
//! Both use f32 arithmetic but different reduction orders (SIMD sum vs AMX).
//!
//! Failure means: f32_gemv shader produces incorrect results for o_proj dimensions.

use moe_kernels::MetalDequantGemv;

/// K2 o_proj dimensions: [hidden_size=7168, v_total=8192]
const OUT_FEATURES: usize = 7168;
const IN_FEATURES: usize = 8192;

#[test]
fn gpu_oproj_matches_cpu() {
    let metal = MetalDequantGemv::new();
    let mut metal = match metal {
        Some(m) => m,
        None => { eprintln!("SKIP: no Metal GPU"); return; }
    };
    metal.init_scratch(OUT_FEATURES, 2048, 8, 128);
    metal.ensure_oproj_scratch(OUT_FEATURES, IN_FEATURES);

    // Deterministic weights and input
    let w: Vec<f32> = (0..OUT_FEATURES * IN_FEATURES)
        .map(|i| ((i as f32 * 0.00001).sin() * 0.01))
        .collect();
    let x: Vec<f32> = (0..IN_FEATURES)
        .map(|i| ((i as f32 * 0.01).cos() * 0.1))
        .collect();

    // CPU reference: cblas_sgemv
    let mut cpu_out = vec![0.0f32; OUT_FEATURES];
    moe_infer::blas::sgemv_f32(&w, &x, &mut cpu_out, OUT_FEATURES, IN_FEATURES);

    // GPU path: wrap weights as Metal buffer + dispatch
    let w_bytes = unsafe {
        std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
    };
    let w_buf = metal.wrap_mmap(w_bytes);
    let gpu_out = metal.gemv_f32_gpu(&w_buf, &x, OUT_FEATURES, IN_FEATURES);

    // Compare: same f32 precision but different reduction order → tolerance 1e-3
    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-3,
        "GPU vs CPU o_proj max_diff={max_diff} (expected <1e-3)");
    eprintln!("GPU vs CPU o_proj max_diff={max_diff:.6} — PASS");
}

#[test]
fn gpu_oproj_zero_input() {
    let metal = MetalDequantGemv::new();
    let mut metal = match metal {
        Some(m) => m,
        None => { eprintln!("SKIP: no Metal GPU"); return; }
    };
    metal.init_scratch(OUT_FEATURES, 2048, 8, 128);
    metal.ensure_oproj_scratch(OUT_FEATURES, IN_FEATURES);

    let w: Vec<f32> = (0..OUT_FEATURES * IN_FEATURES)
        .map(|i| ((i as f32 * 0.00001).sin() * 0.01))
        .collect();
    let x = vec![0.0f32; IN_FEATURES];

    let mut cpu_out = vec![0.0f32; OUT_FEATURES];
    moe_infer::blas::sgemv_f32(&w, &x, &mut cpu_out, OUT_FEATURES, IN_FEATURES);

    let w_bytes = unsafe {
        std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
    };
    let w_buf = metal.wrap_mmap(w_bytes);
    let gpu_out = metal.gemv_f32_gpu(&w_buf, &x, OUT_FEATURES, IN_FEATURES);

    // Zero input → zero output
    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-6,
        "GPU vs CPU zero-input max_diff={max_diff} (expected <1e-6)");
    eprintln!("Zero input: max_diff={max_diff:.8} — PASS");
}

#[test]
fn gpu_oproj_uniform_input() {
    let metal = MetalDequantGemv::new();
    let mut metal = match metal {
        Some(m) => m,
        None => { eprintln!("SKIP: no Metal GPU"); return; }
    };
    metal.init_scratch(OUT_FEATURES, 2048, 8, 128);
    metal.ensure_oproj_scratch(OUT_FEATURES, IN_FEATURES);

    let w: Vec<f32> = (0..OUT_FEATURES * IN_FEATURES)
        .map(|i| ((i as f32 * 0.00001).sin() * 0.01))
        .collect();
    let x = vec![1.0f32; IN_FEATURES];

    let mut cpu_out = vec![0.0f32; OUT_FEATURES];
    moe_infer::blas::sgemv_f32(&w, &x, &mut cpu_out, OUT_FEATURES, IN_FEATURES);

    let w_bytes = unsafe {
        std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
    };
    let w_buf = metal.wrap_mmap(w_bytes);
    let gpu_out = metal.gemv_f32_gpu(&w_buf, &x, OUT_FEATURES, IN_FEATURES);

    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-3,
        "GPU vs CPU uniform-input max_diff={max_diff} (expected <1e-3)");
    eprintln!("Uniform input: max_diff={max_diff:.6} — PASS");
}

#[test]
fn oproj_gpu_persistent_buffer_matches_cpu() {
    let metal = MetalDequantGemv::new();
    let mut metal = match metal {
        Some(m) => m,
        None => { eprintln!("SKIP: no Metal GPU"); return; }
    };
    metal.init_scratch(OUT_FEATURES, 2048, 8, 128);
    metal.ensure_oproj_scratch(OUT_FEATURES, IN_FEATURES);

    let w: Vec<f32> = (0..OUT_FEATURES * IN_FEATURES)
        .map(|i| ((i as f32 * 0.00001).sin() * 0.01))
        .collect();
    let x: Vec<f32> = (0..IN_FEATURES)
        .map(|i| ((i as f32 * 0.01).cos() * 0.1))
        .collect();

    // CPU reference
    let mut cpu_out = vec![0.0f32; OUT_FEATURES];
    moe_infer::blas::sgemv_f32(&w, &x, &mut cpu_out, OUT_FEATURES, IN_FEATURES);

    // GPU via oproj_gpu (persistent buffer path — memcpy + dispatch)
    let gpu_out = metal.oproj_gpu(&w, &x, OUT_FEATURES, IN_FEATURES);

    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-3,
        "oproj_gpu vs CPU max_diff={max_diff} (expected <1e-3)");
    eprintln!("oproj_gpu persistent buffer: max_diff={max_diff:.6} — PASS");

    // Call twice to verify buffer reuse works (different weights)
    let w2: Vec<f32> = (0..OUT_FEATURES * IN_FEATURES)
        .map(|i| ((i as f32 * 0.00003).cos() * 0.02))
        .collect();
    let mut cpu_out2 = vec![0.0f32; OUT_FEATURES];
    moe_infer::blas::sgemv_f32(&w2, &x, &mut cpu_out2, OUT_FEATURES, IN_FEATURES);
    let gpu_out2 = metal.oproj_gpu(&w2, &x, OUT_FEATURES, IN_FEATURES);

    let max_diff2 = cpu_out2.iter().zip(gpu_out2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff2 < 1e-3,
        "oproj_gpu reuse vs CPU max_diff={max_diff2} (expected <1e-3)");
    eprintln!("oproj_gpu buffer reuse: max_diff={max_diff2:.6} — PASS");
}
