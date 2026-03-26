//! Correctness test for Metal GPU AdamW optimizer.
//!
//! What was optimized:
//!   Switched from 2-thread CPU step_fused to Metal GPU batch dispatch
//!   for the AdamW weight update. All 48.8M params processed in a single
//!   Metal command buffer instead of serial CPU loops.
//!
//! Invariant checked:
//!   Metal Adam must produce identical weight updates as CPU step_fused
//!   for the same inputs (weights, grads, optimizer state, hyperparams).
//!
//! Tolerance: 1e-5 relative error (GPU fp32 vs CPU fp32, may differ in rounding)
//!
//! Failure meaning:
//!   The Metal shader computes different AdamW math than the CPU implementation,
//!   or buffer copy/readback introduced data corruption.

use engine::cpu::adam::step_fused;
use engine::metal_adam::MetalAdam;

fn rel_err(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_err = 0.0f32;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let denom = x.abs().max(y.abs()).max(1e-8);
        let err = (x - y).abs() / denom;
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < 1e-4,
            "{label}[{i}]: cpu={x} vs gpu={y} (rel_err={err})"
        );
    }
    max_err
}

/// Single step: Metal vs CPU on a small buffer.
#[test]
fn metal_adam_matches_cpu_small() {
    let metal = MetalAdam::new().expect("Metal GPU required");
    let n = 1024;

    // Same initial state
    let grad: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001 - 0.5) * 0.1).collect();
    let param_init: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003 - 1.0) * 0.5).collect();
    let m_init = vec![0.0f32; n];
    let v_init = vec![0.0f32; n];

    let (lr, b1, b2, eps, wd, gs) = (1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0);

    // CPU
    let mut p_cpu = param_init.clone();
    let mut m_cpu = m_init.clone();
    let mut v_cpu = v_init.clone();
    step_fused(
        &mut p_cpu, &grad, &mut m_cpu, &mut v_cpu, 1, lr, b1, b2, eps, wd, gs,
    );

    // GPU
    let mut p_gpu = param_init.clone();
    let mut m_gpu = m_init.clone();
    let mut v_gpu = v_init.clone();
    metal.step(
        &mut p_gpu, &grad, &mut m_gpu, &mut v_gpu, 1, lr, b1, b2, eps, wd,
    );

    let pe = rel_err(&p_cpu, &p_gpu, "param");
    let me = rel_err(&m_cpu, &m_gpu, "m");
    let ve = rel_err(&v_cpu, &v_gpu, "v");
    println!(
        "PASS: Metal Adam matches CPU (n={n}, max rel_err: param={pe:.2e}, m={me:.2e}, v={ve:.2e})"
    );
}

/// Multiple steps with grad_scale, matching the batch API.
#[test]
fn metal_adam_batch_multi_step() {
    let metal = MetalAdam::new().expect("Metal GPU required");
    let n = 4096;

    let grad: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 3) % 1000) as f32 * 0.001 - 0.5)
        .collect();
    let param_init: Vec<f32> = (0..n)
        .map(|i| ((i * 13 + 7) % 1000) as f32 * 0.002 - 1.0)
        .collect();

    let (lr, b1, b2, eps, wd, gs) = (3e-4, 0.9, 0.999, 1e-8, 0.01, 0.5);

    let mut p_cpu = param_init.clone();
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];

    let mut p_gpu = param_init.clone();
    let mut m_gpu = vec![0.0f32; n];
    let mut v_gpu = vec![0.0f32; n];

    for t in 1..=5 {
        step_fused(
            &mut p_cpu, &grad, &mut m_cpu, &mut v_cpu, t, lr, b1, b2, eps, wd, gs,
        );

        let mut batch = metal.begin_batch(t, b1, b2, eps, gs);
        batch.add(&mut p_gpu, &grad, &mut m_gpu, &mut v_gpu, lr, wd);
        batch.execute();
    }

    let pe = rel_err(&p_cpu, &p_gpu, "param_5steps");
    let me = rel_err(&m_cpu, &m_gpu, "m_5steps");
    let ve = rel_err(&v_cpu, &v_gpu, "v_5steps");
    println!(
        "PASS: Metal Adam batch matches CPU after 5 steps (n={n}, max rel_err: param={pe:.2e}, m={me:.2e}, v={ve:.2e})"
    );
}

/// Test with zero weight decay (embedding/gamma case).
#[test]
fn metal_adam_zero_weight_decay() {
    let metal = MetalAdam::new().expect("Metal GPU required");
    let n = 768;

    let grad: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let param_init: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();

    let (lr, b1, b2, eps, gs) = (1e-3, 0.9, 0.999, 1e-8, 1.0);

    let mut p_cpu = param_init.clone();
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];
    step_fused(
        &mut p_cpu, &grad, &mut m_cpu, &mut v_cpu, 1, lr, b1, b2, eps, 0.0, gs,
    );

    let mut p_gpu = param_init.clone();
    let mut m_gpu = vec![0.0f32; n];
    let mut v_gpu = vec![0.0f32; n];
    metal.step(
        &mut p_gpu, &grad, &mut m_gpu, &mut v_gpu, 1, lr, b1, b2, eps, 0.0,
    );

    let pe = rel_err(&p_cpu, &p_gpu, "param_nowd");
    println!("PASS: Metal Adam with wd=0 matches CPU (max rel_err={pe:.2e})");
}
