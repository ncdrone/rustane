//! Correctness test for CPU fused Adam replacing Metal GPU Adam.
//!
//! What was optimized:
//!   Replaced 56 separate Metal GPU dispatches with CPU fused single-pass Adam.
//!   Metal Adam spent ~16ms on GPU execution for 48.8M params across 56 dispatches
//!   (GPU compute was 0.03ms, rest was driver/scheduling overhead).
//!   CPU fused processes each element in a single pass (28 bytes/elem vs 56 dispatches).
//!
//! Invariant checked:
//!   CPU step_fused must produce results matching Metal adam_step shader within tolerance.
//!   They implement the same AdamW formula with grad_scale, but FMA differences exist
//!   (CPU uses x87/NEON FMA, GPU uses Metal FMA).
//!
//! Tolerance: 1e-5 (same as metal_adam_test.rs — GPU FMA vs CPU precision)
//!
//! Failure meaning:
//!   The CPU fused loop computes different values than the Metal shader.
//!   This would cause training divergence between CPU and GPU Adam paths.

use engine::cpu::adam;
use engine::metal_adam::MetalAdam;

const TOL: f32 = 1e-5;

fn make_grad(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Test 1: CPU fused matches Metal GPU for a single large tensor.
#[test]
fn cpu_fused_vs_metal_single_tensor() {
    let metal = MetalAdam::new().expect("Metal GPU required");
    let n = 768 * 768; // wq size
    let (beta1, beta2, eps, lr, wd, gs) = (0.9f32, 0.95, 1e-8, 3e-4 * 0.05, 0.1, 1.0 / 256.0);
    let grad = make_grad(n, 42);
    let t = 1u32;

    // Metal GPU
    let mut param_gpu = vec![0.5f32; n];
    let mut m_gpu = vec![0.0f32; n];
    let mut v_gpu = vec![0.0f32; n];
    let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
    batch.add(&mut param_gpu, &grad, &mut m_gpu, &mut v_gpu, lr, wd);
    batch.execute();

    // CPU fused
    let mut param_cpu = vec![0.5f32; n];
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];
    adam::step_fused(
        &mut param_cpu,
        &grad,
        &mut m_cpu,
        &mut v_cpu,
        t,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        gs,
    );

    let max_diff_param = param_gpu
        .iter()
        .zip(param_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_diff_m = m_gpu
        .iter()
        .zip(m_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_diff_v = v_gpu
        .iter()
        .zip(v_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff_param < TOL, "param max_diff={max_diff_param}");
    assert!(max_diff_m < TOL, "m max_diff={max_diff_m}");
    assert!(max_diff_v < TOL, "v max_diff={max_diff_v}");
}

/// Test 2: CPU fused matches Metal GPU over 10 steps (accumulated precision drift).
#[test]
fn cpu_fused_vs_metal_10_steps() {
    let metal = MetalAdam::new().expect("Metal GPU required");
    let n = 589_824; // wq
    let (beta1, beta2, eps, lr, wd, gs) = (0.9f32, 0.95, 1e-8, 3e-4 * 0.05, 0.1, 1.0 / 256.0);
    let grad = make_grad(n, 42);

    let mut param_gpu = vec![0.5f32; n];
    let mut m_gpu = vec![0.0f32; n];
    let mut v_gpu = vec![0.0f32; n];
    let mut param_cpu = vec![0.5f32; n];
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];

    for t in 1u32..=10 {
        let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
        batch.add(&mut param_gpu, &grad, &mut m_gpu, &mut v_gpu, lr, wd);
        batch.execute();

        adam::step_fused(
            &mut param_cpu,
            &grad,
            &mut m_cpu,
            &mut v_cpu,
            t,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            gs,
        );

        let max_diff = param_gpu
            .iter()
            .zip(param_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < TOL, "step {t}: param max_diff={max_diff}");
    }
}

/// Test 3: Edge case — zero weight decay (embedding/norm tensors).
#[test]
fn cpu_fused_vs_metal_no_weight_decay() {
    let metal = MetalAdam::new().expect("Metal GPU required");
    let n = 768 * 8192; // embed size
    let (beta1, beta2, eps, lr, gs) = (0.9f32, 0.95, 1e-8, 3e-4, 1.0 / 256.0);
    let grad = make_grad(n, 99);
    let t = 5u32;

    let mut param_gpu = vec![0.01f32; n];
    let mut m_gpu = vec![0.0f32; n];
    let mut v_gpu = vec![0.0f32; n];
    let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
    batch.add(&mut param_gpu, &grad, &mut m_gpu, &mut v_gpu, lr, 0.0);
    batch.execute();

    let mut param_cpu = vec![0.01f32; n];
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];
    adam::step_fused(
        &mut param_cpu,
        &grad,
        &mut m_cpu,
        &mut v_cpu,
        t,
        lr,
        beta1,
        beta2,
        eps,
        0.0,
        gs,
    );

    let max_diff = param_gpu
        .iter()
        .zip(param_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < TOL, "embed param max_diff={max_diff}");
}
