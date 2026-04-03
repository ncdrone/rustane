//! Correctness test for 2-thread parallel Adam update.
//!
//! What was optimized:
//!   Split 56 step_fused calls across 2 threads using std::thread::scope.
//!   step_fused is compute-bound (sqrt+div per element): single-thread achieves
//!   ~80 GB/s of M4 Max's 400 GB/s bandwidth. 2 threads should roughly double throughput.
//!
//! Invariant checked:
//!   Parallel update_weights must produce bit-exact results vs sequential step_fused.
//!   Both call the same step_fused function on disjoint memory; only execution order
//!   and thread assignment differ. Since each tensor is processed independently,
//!   results must be identical regardless of which thread handles them.
//!
//! Tolerance: 0.0 (exact match — no math change, only parallelism)
//!
//! Failure meaning:
//!   A data race or incorrect memory split exists in the parallel implementation.
//!   This would cause non-deterministic weight updates and training divergence.

use engine::cpu::adam;
use engine::full_model::{ModelGrads, ModelOptState, ModelWeights, TrainConfig, update_weights};
use engine::layer::{LayerGrads, LayerWeights};
use engine::metal_adam::MetalAdam;
use engine::model::{FfnActivation, ModelConfig};
use engine::training::LayerOptState;

fn make_data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .map(|x| x * scale)
        .collect()
}

fn gpt_config() -> ModelConfig {
    ModelConfig {
        dim: 768,
        seq: 512,
        vocab: 8192,
        nlayers: 6,
        heads: 12,
        kv_heads: 4,
        hd: 64,
        q_dim: 768,
        kv_dim: 256,
        gqa_ratio: 3,
        hidden: 2048,
        ffn_activation: FfnActivation::SwiGlu,
    }
}

fn random_model_weights(cfg: &ModelConfig, seed: u64) -> ModelWeights {
    let mut s = seed;
    let next_seed = |s: &mut u64| {
        *s += 1;
        *s
    };
    ModelWeights {
        embed: make_data(cfg.vocab * cfg.dim, next_seed(&mut s), 0.005),
        layers: (0..cfg.nlayers)
            .map(|_| {
                let mut layer = LayerWeights {
                    wq: make_data(cfg.dim * cfg.q_dim, next_seed(&mut s), 0.01),
                    wk: make_data(cfg.dim * cfg.kv_dim, next_seed(&mut s), 0.01),
                    wv: make_data(cfg.dim * cfg.kv_dim, next_seed(&mut s), 0.01),
                    wo: make_data(cfg.dim * cfg.q_dim, next_seed(&mut s), 0.01),
                    w1: make_data(cfg.dim * cfg.hidden, next_seed(&mut s), 0.01),
                    w3: make_data(cfg.dim * cfg.hidden, next_seed(&mut s), 0.01),
                    w2: make_data(cfg.hidden * cfg.dim, next_seed(&mut s), 0.01),
                    wqt: vec![0.0; cfg.q_dim * cfg.dim],
                    wkt: vec![0.0; cfg.kv_dim * cfg.dim],
                    wvt: vec![0.0; cfg.kv_dim * cfg.dim],
                    wot: vec![0.0; cfg.dim * cfg.q_dim],
                    w1t: vec![0.0; cfg.hidden * cfg.dim],
                    w3t: vec![0.0; cfg.hidden * cfg.dim],
                    gamma1: vec![1.0; cfg.dim],
                    gamma2: vec![1.0; cfg.dim],
                    generation: 0,
                };
                layer.refresh_transposes(cfg);
                layer
            })
            .collect(),
        gamma_final: vec![1.0; cfg.dim],
    }
}

fn random_model_grads(cfg: &ModelConfig, seed: u64) -> ModelGrads {
    let mut s = seed;
    let next_seed = |s: &mut u64| {
        *s += 1;
        *s
    };
    ModelGrads {
        dembed: make_data(cfg.vocab * cfg.dim, next_seed(&mut s), 0.1),
        layers: (0..cfg.nlayers)
            .map(|_| LayerGrads {
                dwq: make_data(cfg.dim * cfg.q_dim, next_seed(&mut s), 0.1),
                dwk: make_data(cfg.dim * cfg.kv_dim, next_seed(&mut s), 0.1),
                dwv: make_data(cfg.dim * cfg.kv_dim, next_seed(&mut s), 0.1),
                dwo: make_data(cfg.dim * cfg.q_dim, next_seed(&mut s), 0.1),
                dw1: make_data(cfg.dim * cfg.hidden, next_seed(&mut s), 0.1),
                dw3: make_data(cfg.dim * cfg.hidden, next_seed(&mut s), 0.1),
                dw2: make_data(cfg.hidden * cfg.dim, next_seed(&mut s), 0.1),
                dgamma1: make_data(cfg.dim, next_seed(&mut s), 0.1),
                dgamma2: make_data(cfg.dim, next_seed(&mut s), 0.1),
            })
            .collect(),
        dgamma_final: make_data(cfg.dim, next_seed(&mut s), 0.1),
    }
}

fn clone_weights(w: &ModelWeights) -> ModelWeights {
    ModelWeights {
        embed: w.embed.clone(),
        layers: w
            .layers
            .iter()
            .map(|l| LayerWeights {
                wq: l.wq.clone(),
                wk: l.wk.clone(),
                wv: l.wv.clone(),
                wo: l.wo.clone(),
                w1: l.w1.clone(),
                w3: l.w3.clone(),
                w2: l.w2.clone(),
                wqt: l.wqt.clone(),
                wkt: l.wkt.clone(),
                wvt: l.wvt.clone(),
                wot: l.wot.clone(),
                w1t: l.w1t.clone(),
                w3t: l.w3t.clone(),
                gamma1: l.gamma1.clone(),
                gamma2: l.gamma2.clone(),
                generation: l.generation,
            })
            .collect(),
        gamma_final: w.gamma_final.clone(),
    }
}

fn clone_opt(o: &ModelOptState) -> ModelOptState {
    ModelOptState {
        embed_m: o.embed_m.clone(),
        embed_v: o.embed_v.clone(),
        layers: o
            .layers
            .iter()
            .map(|l| LayerOptState {
                m_wq: l.m_wq.clone(),
                v_wq: l.v_wq.clone(),
                m_wk: l.m_wk.clone(),
                v_wk: l.v_wk.clone(),
                m_wv: l.m_wv.clone(),
                v_wv: l.v_wv.clone(),
                m_wo: l.m_wo.clone(),
                v_wo: l.v_wo.clone(),
                m_w1: l.m_w1.clone(),
                v_w1: l.v_w1.clone(),
                m_w3: l.m_w3.clone(),
                v_w3: l.v_w3.clone(),
                m_w2: l.m_w2.clone(),
                v_w2: l.v_w2.clone(),
                m_gamma1: l.m_gamma1.clone(),
                v_gamma1: l.v_gamma1.clone(),
                m_gamma2: l.m_gamma2.clone(),
                v_gamma2: l.v_gamma2.clone(),
            })
            .collect(),
        gamma_final_m: o.gamma_final_m.clone(),
        gamma_final_v: o.gamma_final_v.clone(),
    }
}

/// Reference sequential Adam: calls step_fused in the original order on a single thread.
fn sequential_adam(
    cfg: &ModelConfig,
    weights: &mut ModelWeights,
    grads: &ModelGrads,
    opt: &mut ModelOptState,
    t: u32,
    lr: f32,
    tc: &TrainConfig,
    grad_scale: f32,
) {
    let wd = tc.weight_decay;
    let matrix_lr = lr * tc.matrix_lr_scale;
    let embed_lr = lr * tc.embed_lr_scale;
    let (b1, b2, eps) = (tc.beta1, tc.beta2, tc.eps);

    adam::step_fused(
        &mut weights.embed,
        &grads.dembed,
        &mut opt.embed_m,
        &mut opt.embed_v,
        t,
        embed_lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    adam::step_fused(
        &mut weights.gamma_final,
        &grads.dgamma_final,
        &mut opt.gamma_final_m,
        &mut opt.gamma_final_v,
        t,
        lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    for l in 0..cfg.nlayers {
        let w = &mut weights.layers[l];
        let g = &grads.layers[l];
        let o = &mut opt.layers[l];
        adam::step_fused(
            &mut w.wq,
            &g.dwq,
            &mut o.m_wq,
            &mut o.v_wq,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.wk,
            &g.dwk,
            &mut o.m_wk,
            &mut o.v_wk,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.wv,
            &g.dwv,
            &mut o.m_wv,
            &mut o.v_wv,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.wo,
            &g.dwo,
            &mut o.m_wo,
            &mut o.v_wo,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.w1,
            &g.dw1,
            &mut o.m_w1,
            &mut o.v_w1,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.w3,
            &g.dw3,
            &mut o.m_w3,
            &mut o.v_w3,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.w2,
            &g.dw2,
            &mut o.m_w2,
            &mut o.v_w2,
            t,
            matrix_lr,
            b1,
            b2,
            eps,
            wd,
            grad_scale,
        );
        adam::step_fused(
            &mut w.gamma1,
            &g.dgamma1,
            &mut o.m_gamma1,
            &mut o.v_gamma1,
            t,
            lr,
            b1,
            b2,
            eps,
            0.0,
            grad_scale,
        );
        adam::step_fused(
            &mut w.gamma2,
            &g.dgamma2,
            &mut o.m_gamma2,
            &mut o.v_gamma2,
            t,
            lr,
            b1,
            b2,
            eps,
            0.0,
            grad_scale,
        );
        w.generation += 1;
    }
}

fn assert_vecs_exact(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff == 0.0,
        "{name}: max_diff={max_diff} (expected exact match)"
    );
}

fn compare_weights(a: &ModelWeights, b: &ModelWeights, step_label: &str) {
    assert_vecs_exact(&a.embed, &b.embed, &format!("{step_label} embed"));
    assert_vecs_exact(
        &a.gamma_final,
        &b.gamma_final,
        &format!("{step_label} gamma_final"),
    );
    for l in 0..a.layers.len() {
        let al = &a.layers[l];
        let bl = &b.layers[l];
        assert_vecs_exact(&al.wq, &bl.wq, &format!("{step_label} L{l}.wq"));
        assert_vecs_exact(&al.wk, &bl.wk, &format!("{step_label} L{l}.wk"));
        assert_vecs_exact(&al.wv, &bl.wv, &format!("{step_label} L{l}.wv"));
        assert_vecs_exact(&al.wo, &bl.wo, &format!("{step_label} L{l}.wo"));
        assert_vecs_exact(&al.w1, &bl.w1, &format!("{step_label} L{l}.w1"));
        assert_vecs_exact(&al.w3, &bl.w3, &format!("{step_label} L{l}.w3"));
        assert_vecs_exact(&al.w2, &bl.w2, &format!("{step_label} L{l}.w2"));
        assert_vecs_exact(&al.gamma1, &bl.gamma1, &format!("{step_label} L{l}.gamma1"));
        assert_vecs_exact(&al.gamma2, &bl.gamma2, &format!("{step_label} L{l}.gamma2"));
    }
}

/// Test 1: Parallel Adam produces bit-exact results vs sequential for all 56 tensors.
#[test]
fn parallel_adam_matches_sequential() {
    let cfg = gpt_config();
    let tc = TrainConfig::default();
    let metal = MetalAdam::new().expect("Metal required");
    let grads = random_model_grads(&cfg, 1000);

    let mut w_seq = random_model_weights(&cfg, 42);
    let mut o_seq = ModelOptState::zeros(&cfg);
    let mut w_par = clone_weights(&w_seq);
    let mut o_par = clone_opt(&o_seq);

    let t = 1u32;
    let lr = 3e-4;
    let grad_scale = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);

    sequential_adam(&cfg, &mut w_seq, &grads, &mut o_seq, t, lr, &tc, grad_scale);
    update_weights(
        &cfg, &mut w_par, &grads, &mut o_par, t, lr, &tc, &metal, grad_scale,
    );

    compare_weights(&w_seq, &w_par, "step1");
}

/// Test 2: 5 steps with changing gradients — catch accumulation drift over multiple steps.
#[test]
fn parallel_adam_matches_5_steps() {
    let cfg = gpt_config();
    let tc = TrainConfig::default();
    let metal = MetalAdam::new().expect("Metal required");

    let mut w_seq = random_model_weights(&cfg, 42);
    let mut o_seq = ModelOptState::zeros(&cfg);
    let mut w_par = clone_weights(&w_seq);
    let mut o_par = clone_opt(&o_seq);

    for t in 1u32..=5 {
        let grads = random_model_grads(&cfg, 1000 + t as u64);
        let lr = engine::full_model::learning_rate(t - 1, &tc);
        let grad_scale = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);

        sequential_adam(&cfg, &mut w_seq, &grads, &mut o_seq, t, lr, &tc, grad_scale);
        update_weights(
            &cfg, &mut w_par, &grads, &mut o_par, t, lr, &tc, &metal, grad_scale,
        );

        compare_weights(&w_seq, &w_par, &format!("step{t}"));
    }
}

/// Test 3: Edge case — back-to-back calls (stale data check).
/// Verifies no state leaks between consecutive update_weights invocations.
#[test]
fn parallel_adam_no_stale_state() {
    let cfg = gpt_config();
    let tc = TrainConfig::default();
    let metal = MetalAdam::new().expect("Metal required");

    let grads_a = random_model_grads(&cfg, 2000);
    let grads_b = random_model_grads(&cfg, 3000);

    let mut w_seq = random_model_weights(&cfg, 42);
    let mut o_seq = ModelOptState::zeros(&cfg);
    let mut w_par = clone_weights(&w_seq);
    let mut o_par = clone_opt(&o_seq);

    let gs = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);

    // Step 1 with grads_a
    sequential_adam(&cfg, &mut w_seq, &grads_a, &mut o_seq, 1, 3e-4, &tc, gs);
    update_weights(
        &cfg, &mut w_par, &grads_a, &mut o_par, 1, 3e-4, &tc, &metal, gs,
    );

    // Step 2 with different grads_b
    sequential_adam(&cfg, &mut w_seq, &grads_b, &mut o_seq, 2, 3e-4, &tc, gs);
    update_weights(
        &cfg, &mut w_par, &grads_b, &mut o_par, 2, 3e-4, &tc, &metal, gs,
    );

    compare_weights(&w_seq, &w_par, "stale_check");
}
