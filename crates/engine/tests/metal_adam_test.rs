//! Test: Metal Adam produces the same results as CPU Adam.

use engine::cpu::adam::{self, AdamConfig};
use engine::metal_adam::MetalAdam;

fn run_both(n: usize, steps: u32) {
    let cfg = AdamConfig::default();
    let metal = MetalAdam::new().expect("Metal device available");

    // Identical starting state
    let mut param_cpu = vec![1.0f32; n];
    let mut param_gpu = param_cpu.clone();
    let mut m_cpu = vec![0.0f32; n];
    let mut m_gpu = m_cpu.clone();
    let mut v_cpu = vec![0.0f32; n];
    let mut v_gpu = v_cpu.clone();

    // Pseudo-random gradients
    let mut grad = vec![0.0f32; n];
    let mut seed: u64 = 42;
    for g in grad.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *g = ((seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }

    for t in 1..=steps {
        adam::step(&mut param_cpu, &grad, &mut m_cpu, &mut v_cpu, t, &cfg);
        metal.step(
            &mut param_gpu,
            &grad,
            &mut m_gpu,
            &mut v_gpu,
            t,
            cfg.lr,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
        );

        // Check agreement after each step
        let max_param_diff = param_cpu
            .iter()
            .zip(&param_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_m_diff = m_cpu
            .iter()
            .zip(&m_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_v_diff = v_cpu
            .iter()
            .zip(&v_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Allow small float precision differences (GPU uses fma, CPU uses vDSP)
        let tol = 1e-5;
        assert!(
            max_param_diff < tol,
            "step {t}: param diff {max_param_diff}"
        );
        assert!(max_m_diff < tol, "step {t}: m diff {max_m_diff}");
        assert!(max_v_diff < tol, "step {t}: v diff {max_v_diff}");
    }
}

#[test]
fn metal_matches_cpu_small() {
    run_both(256, 10);
}

#[test]
fn metal_matches_cpu_medium() {
    run_both(4096, 5);
}

#[test]
fn metal_matches_cpu_large() {
    run_both(262_144, 3); // 1MB — representative of weight matrices
}
