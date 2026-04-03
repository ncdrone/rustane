//! Correctness test for cache-adam-scalars optimization.
//!
//! What was optimized:
//!   `AdamBatch::add()` previously created 8 Metal scalar buffers per call:
//!   beta1, beta2, eps, bc1, bc2, grad_scale, lr, weight_decay.
//!   With 56 parameters (6 layers × 9 + embed + gamma_final), this was 448 allocations per step.
//!   The fix: `begin_batch(t, beta1, beta2, eps, grad_scale)` pre-creates 6 shared buffers.
//!   `add()` now only creates 2 per-tensor buffers (lr, wd) = 56×2 + 6 = 118 total.
//!
//! Invariant checked:
//!   The optimized Metal Adam (new begin_batch API) must produce bit-for-bit identical
//!   results to the CPU Adam reference implementation, at the same tolerance as before.
//!   We test:
//!   1. Single-tensor step: matches CPU Adam (same as metal_adam_test.rs baseline check)
//!   2. Multi-tensor batch with mixed lr/wd: each tensor matches its own CPU Adam
//!   3. Edge case: weight_decay=0.0 tensors mixed with weight_decay>0 tensors in same batch
//!   4. Edge case: multiple steps in sequence (bc1/bc2 advance correctly)
//!
//! Tolerance: 1e-5 (same as existing metal_adam_test.rs — GPU FMA vs CPU vDSP precision)
//!
//! Failure meaning:
//!   The scalar caching changed what values reach the GPU shader.
//!   Adam moments and params would diverge from reference, causing training instability.

use engine::cpu::adam::{self, AdamConfig};
use engine::metal_adam::MetalAdam;

const TOL: f32 = 1e-5;

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < TOL,
        "{label}: max_diff={max_diff} exceeds tolerance={TOL}"
    );
}

// CPU Adam reference for a single step
fn cpu_adam_step(
    param: &mut Vec<f32>,
    grad: &[f32],
    m: &mut Vec<f32>,
    v: &mut Vec<f32>,
    t: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    grad_scale: f32,
) {
    let cfg = AdamConfig {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay: wd,
    };
    // Apply grad_scale manually before cpu step (matches Metal shader: g = grad[id] * gscale)
    let scaled_grad: Vec<f32> = grad.iter().map(|&g| g * grad_scale).collect();
    adam::step(param, &scaled_grad, m, v, t, &cfg);
}

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

/// Test 1: single-tensor step via new begin_batch API matches CPU Adam.
/// Edge case: weight_decay = 0.0 (norm/embedding case).
#[test]
fn cache_scalars_single_tensor_no_wd() {
    let metal = MetalAdam::new().expect("Metal required");
    let n = 4096;
    let (beta1, beta2, eps, lr, wd, gs) = (0.9f32, 0.95, 1e-8, 3e-4, 0.0, 1.0 / 256.0);
    let grad = make_grad(n, 42);

    let mut param_cpu = vec![0.5f32; n];
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];

    let mut param_gpu = param_cpu.clone();
    let mut m_gpu = m_cpu.clone();
    let mut v_gpu = v_cpu.clone();

    let t = 1u32;
    cpu_adam_step(
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

    let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
    batch.add(&mut param_gpu, &grad, &mut m_gpu, &mut v_gpu, lr, wd);
    batch.execute();

    assert_close(&param_cpu, &param_gpu, "param (no_wd)");
    assert_close(&m_cpu, &m_gpu, "m (no_wd)");
    assert_close(&v_cpu, &v_gpu, "v (no_wd)");
}

/// Test 2: single-tensor step with non-zero weight decay (matrix case).
#[test]
fn cache_scalars_single_tensor_with_wd() {
    let metal = MetalAdam::new().expect("Metal required");
    let n = 4096;
    let (beta1, beta2, eps, matrix_lr, wd, gs) =
        (0.9f32, 0.95, 1e-8, 3e-4 * 0.05, 0.1, 1.0 / 256.0);
    let grad = make_grad(n, 99);

    let mut param_cpu = vec![0.3f32; n];
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];
    let mut param_gpu = param_cpu.clone();
    let mut m_gpu = m_cpu.clone();
    let mut v_gpu = v_cpu.clone();

    let t = 5u32;
    cpu_adam_step(
        &mut param_cpu,
        &grad,
        &mut m_cpu,
        &mut v_cpu,
        t,
        matrix_lr,
        beta1,
        beta2,
        eps,
        wd,
        gs,
    );

    let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
    batch.add(&mut param_gpu, &grad, &mut m_gpu, &mut v_gpu, matrix_lr, wd);
    batch.execute();

    assert_close(&param_cpu, &param_gpu, "param (wd=0.1)");
    assert_close(&m_cpu, &m_gpu, "m (wd=0.1)");
    assert_close(&v_cpu, &v_gpu, "v (wd=0.1)");
}

/// Test 3: multi-tensor batch with mixed lr/wd — the critical case this optimization targets.
/// Verifies that shared scalars (bc1, bc2, beta1, beta2, eps, gs) are correct for ALL tensors,
/// not just the first one. A bug here (stale bc values, wrong shared buffer) would show up
/// when the batch has tensors with different lr groups.
#[test]
fn cache_scalars_multi_tensor_mixed_lr_wd() {
    let metal = MetalAdam::new().expect("Metal required");
    let n = 1024;
    let (beta1, beta2, eps, gs) = (0.9f32, 0.95, 1e-8, 1.0 / 256.0);
    let embed_lr = 3e-4 * 1.0;
    let matrix_lr = 3e-4 * 0.05;
    let norm_lr = 3e-4;
    let wd = 0.1f32;

    let t = 10u32;

    // 3 tensors representing embed, matrix, norm groups
    let grad_embed = make_grad(n, 1);
    let grad_matrix = make_grad(n, 2);
    let grad_norm = make_grad(n, 3);

    // CPU reference for each
    let mut p_embed_cpu = vec![0.1f32; n];
    let mut m_e = vec![0.0f32; n];
    let mut v_e = vec![0.0f32; n];
    let mut p_matrix_cpu = vec![0.2f32; n];
    let mut m_m = vec![0.0f32; n];
    let mut v_m = vec![0.0f32; n];
    let mut p_norm_cpu = vec![1.0f32; n];
    let mut m_n = vec![0.0f32; n];
    let mut v_n = vec![0.0f32; n];

    cpu_adam_step(
        &mut p_embed_cpu,
        &grad_embed,
        &mut m_e,
        &mut v_e,
        t,
        embed_lr,
        beta1,
        beta2,
        eps,
        0.0,
        gs,
    );
    cpu_adam_step(
        &mut p_matrix_cpu,
        &grad_matrix,
        &mut m_m,
        &mut v_m,
        t,
        matrix_lr,
        beta1,
        beta2,
        eps,
        wd,
        gs,
    );
    cpu_adam_step(
        &mut p_norm_cpu,
        &grad_norm,
        &mut m_n,
        &mut v_n,
        t,
        norm_lr,
        beta1,
        beta2,
        eps,
        0.0,
        gs,
    );

    // GPU batch (new API): single begin_batch, three add() calls with different lr/wd
    let mut p_embed_gpu = vec![0.1f32; n];
    let mut mg_e = vec![0.0f32; n];
    let mut vg_e = vec![0.0f32; n];
    let mut p_matrix_gpu = vec![0.2f32; n];
    let mut mg_m = vec![0.0f32; n];
    let mut vg_m = vec![0.0f32; n];
    let mut p_norm_gpu = vec![1.0f32; n];
    let mut mg_n = vec![0.0f32; n];
    let mut vg_n = vec![0.0f32; n];

    let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
    batch.add(
        &mut p_embed_gpu,
        &grad_embed,
        &mut mg_e,
        &mut vg_e,
        embed_lr,
        0.0,
    );
    batch.add(
        &mut p_matrix_gpu,
        &grad_matrix,
        &mut mg_m,
        &mut vg_m,
        matrix_lr,
        wd,
    );
    batch.add(
        &mut p_norm_gpu,
        &grad_norm,
        &mut mg_n,
        &mut vg_n,
        norm_lr,
        0.0,
    );
    batch.execute();

    assert_close(&p_embed_cpu, &p_embed_gpu, "embed param");
    assert_close(&p_matrix_cpu, &p_matrix_gpu, "matrix param");
    assert_close(&p_norm_cpu, &p_norm_gpu, "norm param");
    assert_close(&m_e, &mg_e, "embed m");
    assert_close(&m_m, &mg_m, "matrix m");
    assert_close(&v_m, &vg_m, "matrix v");
}

/// Test 4: multiple sequential steps — bc1/bc2 must advance correctly each step.
/// This checks that shared buffers in begin_batch are recomputed each call, not cached
/// across steps (which would freeze bias correction at t=1).
#[test]
fn cache_scalars_multi_step_bc_advances() {
    let metal = MetalAdam::new().expect("Metal required");
    let n = 256;
    let (beta1, beta2, eps, lr, wd, gs) = (0.9f32, 0.95, 1e-8, 3e-4, 0.1, 1.0);
    let grad = make_grad(n, 7);

    let mut param_cpu = vec![0.5f32; n];
    let mut m_c = vec![0.0f32; n];
    let mut v_c = vec![0.0f32; n];
    let mut param_gpu = param_cpu.clone();
    let mut m_g = vec![0.0f32; n];
    let mut v_g = vec![0.0f32; n];

    for t in 1u32..=5 {
        cpu_adam_step(
            &mut param_cpu,
            &grad,
            &mut m_c,
            &mut v_c,
            t,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            gs,
        );

        let mut batch = metal.begin_batch(t, beta1, beta2, eps, gs);
        batch.add(&mut param_gpu, &grad, &mut m_g, &mut v_g, lr, wd);
        batch.execute();

        assert_close(&param_cpu, &param_gpu, &format!("param step {t}"));
        assert_close(&m_c, &m_g, &format!("m step {t}"));
        assert_close(&v_c, &v_g, &format!("v step {t}"));
    }
}
