//! Diagnostic v2: match exact bench_scale_correctness config.
//! embed_lr_scale=1.0 (same as actual test), 10 steps.
//!
//! Run: cargo test -p engine --test diag_10b_grads -- --ignored --nocapture

use engine::full_model::{self, ModelWeights, ModelGrads, ModelOptState, ModelForwardWorkspace, ModelBackwardWorkspace, TrainConfig};
use engine::layer::{CompiledKernels, LayerGrads};
use engine::model::ModelConfig;
use engine::metal_adam::MetalAdam;
use engine::cpu::vdsp;

fn layer_grad_norms(grads: &LayerGrads) -> (f32, f32, f32, f32, f32) {
    let dw1_norm = vdsp::svesq(&grads.dw1).sqrt();
    let dw2_norm = vdsp::svesq(&grads.dw2).sqrt();
    let dw3_norm = vdsp::svesq(&grads.dw3).sqrt();
    let dwq_norm = vdsp::svesq(&grads.dwq).sqrt();
    let dwo_norm = vdsp::svesq(&grads.dwo).sqrt();
    (dw1_norm, dw2_norm, dw3_norm, dwq_norm, dwo_norm)
}

fn run_diag(cfg: &ModelConfig, name: &str) {
    println!();
    println!("============================================================");
    println!("  DIAG: {} -- {}d/{}h/{}L/seq{} -- ~{:.0}M params",
             name, cfg.dim, cfg.hidden, cfg.nlayers, cfg.seq, cfg.param_count() as f64 / 1e6);
    println!("============================================================");

    let kernels = CompiledKernels::compile(cfg);
    let weights = ModelWeights::random(cfg);
    let mut tc = TrainConfig::default();
    tc.embed_lr_scale = 1.0;  // Match bench_scale_correctness
    let tokens: Vec<u32> = (0..cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (1..=cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let mut grads = ModelGrads::zeros(cfg);
    let mut opt = ModelOptState::zeros(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut weights = weights;

    println!();
    println!("  TrainConfig: max_lr={}, embed_lr_scale={}, matrix_lr_scale={}, loss_scale={}, grad_clip={}",
             tc.max_lr, tc.embed_lr_scale, tc.matrix_lr_scale, tc.loss_scale, tc.grad_clip);
    println!();
    println!("  step | loss      | raw_gnorm   | clip_ratio | eff_scale    | lr           | embed_lr     | matrix_lr");
    println!("  -----|-----------|-------------|------------|--------------|--------------|--------------|----------");

    let mut losses = Vec::new();

    for step in 0..10u32 {
        grads.zero_out();
        let loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
        losses.push(loss);
        full_model::backward_ws(cfg, &kernels, &weights, &fwd_ws, &tokens, tc.softcap, tc.loss_scale, &mut grads, &mut bwd_ws);

        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip { tc.grad_clip / raw_norm } else { gsc };
        let clip_ratio = if raw_norm * gsc > tc.grad_clip { tc.grad_clip / (raw_norm * gsc) } else { 1.0 };
        let lr = full_model::learning_rate(step, &tc);
        let embed_lr = lr * tc.embed_lr_scale;
        let matrix_lr = lr * tc.matrix_lr_scale;

        println!("  {:>4} | {:.4}  | {:.5e} | {:.4}     | {:.5e}  | {:.5e}  | {:.5e}  | {:.5e}",
                 step, loss, raw_norm, clip_ratio, combined_scale, lr, embed_lr, matrix_lr);

        // Per-layer breakdown at key steps
        if step == 0 || step == 4 || step == 9 {
            println!();
            println!("  Layer grad norms (step {}):", step);
            println!("  layer |    dW1       |    dW2       |    dW3       |    dWq       |    dWo");
            println!("  ------|-------------|-------------|-------------|-------------|------------");
            let nl = cfg.nlayers;
            let mut show: Vec<usize> = vec![0, 1];
            if nl > 6 { show.push(nl / 2); }
            if nl > 3 {
                show.push(nl - 2);
                show.push(nl - 1);
            }
            for l in &show {
                let (dw1, dw2, dw3, dwq, dwo) = layer_grad_norms(&grads.layers[*l]);
                println!("  {:>5} | {:.5e} | {:.5e} | {:.5e} | {:.5e} | {:.5e}",
                         l, dw1, dw2, dw3, dwq, dwo);
            }
            // Also show embed grad norm
            let dembed_norm = vdsp::svesq(&grads.dembed).sqrt();
            println!("  embed | {:.5e}", dembed_norm);
            println!();
        }

        full_model::update_weights(cfg, &mut weights, &grads, &mut opt, step + 1, lr, &tc, &metal_adam, combined_scale);
    }

    // Final forward
    let final_loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
    losses.push(final_loss);
    let delta = final_loss - losses[0];
    println!("  Loss trajectory: {:.4} -> {:.4} (delta={:+.4})", losses[0], final_loss, delta);
    println!("  All: {:?}", losses.iter().map(|l| format!("{:.4}", l)).collect::<Vec<_>>());
    if delta < -0.01 {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL");
    }
    println!();
}

#[test]
#[ignore]
fn diag_5b_vs_10b() {
    // 5B
    run_diag(&ModelConfig {
        dim: 3072, hidden: 8192, heads: 24,
        kv_heads: 24, hd: 128, seq: 512, nlayers: 44, vocab: 8192,
        q_dim: 24 * 128, kv_dim: 24 * 128, gqa_ratio: 1,
    }, "5B");

    // 10B
    run_diag(&ModelConfig {
        dim: 4096, hidden: 11008, heads: 32,
        kv_heads: 32, hd: 128, seq: 512, nlayers: 48, vocab: 8192,
        q_dim: 32 * 128, kv_dim: 32 * 128, gqa_ratio: 1,
    }, "10B");
}
