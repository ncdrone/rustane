//! Profile: per-operation breakdown at 600M+ scale.
//!
//! Run:
//!   cargo test -p engine --test profile_600m --release -- --ignored --nocapture

use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig,
};
use engine::layer::{self, CompiledKernels};
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

fn profile_config(cfg: &ModelConfig, name: &str) {
    println!("\n{}", "=".repeat(60));
    println!("=== {} Profile ===", name);
    println!(
        "Model: {}L {}D {}H seq={} vocab={} — ~{:.0}M params\n",
        cfg.nlayers,
        cfg.dim,
        cfg.hidden,
        cfg.seq,
        cfg.vocab,
        cfg.param_count() as f64 / 1e6
    );

    println!("Compiling 10 ANE kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(cfg);
    println!("Compiled in {:.1}s\n", t0.elapsed().as_secs_f32());

    let mut weights = ModelWeights::random(cfg);
    let mut grads = ModelGrads::zeros(cfg);
    let mut opt = ModelOptState::zeros(cfg);
    let tc = TrainConfig::default();

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);

    // Warmup
    println!("Warmup...");
    {
        grads.zero_out();
        let _loss = full_model::forward_ws(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        full_model::backward_ws(
            cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(0, &tc);
        full_model::update_weights(
            cfg,
            &mut weights,
            &grads,
            &mut opt,
            1,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
    }

    // 3 timed steps
    println!(
        "\n{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}   {}",
        "step", "total", "fwd", "bwd", "norm", "upd", "loss"
    );
    println!("{}", "-".repeat(80));

    for step in 0..3u32 {
        grads.zero_out();
        let t0 = Instant::now();

        let t_fwd = Instant::now();
        let loss = full_model::forward_ws(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        let fwd_ms = t_fwd.elapsed().as_secs_f32() * 1000.0;

        let t_bwd = Instant::now();
        full_model::backward_ws(
            cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
        let bwd_ms = t_bwd.elapsed().as_secs_f32() * 1000.0;

        let t_norm = Instant::now();
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let norm_ms = t_norm.elapsed().as_secs_f32() * 1000.0;

        let t_upd = Instant::now();
        let lr = full_model::learning_rate(step + 2, &tc);
        full_model::update_weights(
            cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 2,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
        let upd_ms = t_upd.elapsed().as_secs_f32() * 1000.0;

        let total = t0.elapsed().as_secs_f32() * 1000.0;

        println!(
            "{:<6} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms   {:.4}",
            step, total, fwd_ms, bwd_ms, norm_ms, upd_ms, loss
        );
    }

    // Per-layer forward timing (single layer, run 2 is reported)
    println!("\n--- Per-Layer Forward Breakdown (layer 0, run 2 of 3) ---");
    let x_dummy = vec![0.1f32; cfg.dim * cfg.seq];
    for run in 0..3 {
        let (_, _, timings) = layer::forward_timed(cfg, &kernels, &weights.layers[0], &x_dummy);
        if run == 1 {
            timings.print();

            // Categorize
            let ane_ms = timings.ane_sdpa_ms + timings.ane_wo_ms + timings.ane_ffn_ms;
            let stage_ms = timings.stage_sdpa_ms + timings.stage_wo_ms + timings.stage_ffn_ms;
            let read_ms = timings.read_sdpa_ms + timings.read_wo_ms + timings.read_ffn_ms;
            let cpu_ms = timings.rmsnorm1_ms + timings.residual_rmsnorm2_ms;
            let total = timings.total_ms;
            println!("\n  Category breakdown:");
            println!(
                "    ANE compute:    {:>6.2}ms ({:.0}%)",
                ane_ms,
                ane_ms / total * 100.0
            );
            println!(
                "    IOSurf staging: {:>6.2}ms ({:.0}%)",
                stage_ms,
                stage_ms / total * 100.0
            );
            println!(
                "    IOSurf read:    {:>6.2}ms ({:.0}%)",
                read_ms,
                read_ms / total * 100.0
            );
            println!(
                "    CPU (rmsnorm):  {:>6.2}ms ({:.0}%)",
                cpu_ms,
                cpu_ms / total * 100.0
            );
        }
    }

    println!("\n--- Per-Layer Backward Breakdown (layer 0, run 2 of 3) ---");
    let dy_dummy = vec![0.1f32; cfg.dim * cfg.seq];
    let (_, cache_dummy) = layer::forward(cfg, &kernels, &weights.layers[0], &x_dummy);
    let mut layer_grads = engine::layer::LayerGrads::zeros(cfg);
    let mut layer_ws = engine::layer::BackwardWorkspace::new(cfg);
    for run in 0..3 {
        layer_grads.zero_out();
        let (_, timings) = layer::backward_timed(
            cfg,
            &kernels,
            &weights.layers[0],
            &cache_dummy,
            &dy_dummy,
            &mut layer_grads,
            &mut layer_ws,
        );
        if run == 1 {
            timings.print();

            let stage_ms = timings.stage_run_ffn_bwd_w2t_ms
                + timings.stage_ffn_bwd_w13t_ms
                + timings.stage_run_wot_bwd_ms
                + timings.stage_sdpa_bwd1_ms
                + timings.read_sdpa_bwd1_ms
                + timings.stage_run_sdpa_bwd2_ms
                + timings.stage_q_bwd_ms
                + timings.stage_run_kv_bwd_ms;
            let async_ms = timings.async_ffn_bwd_w13t_plus_dw_ms
                + timings.async_sdpa_bwd1_plus_dwo_ms
                + timings.async_q_bwd_plus_dw_ms;
            let cpu_ms = timings.scale_dy_ms
                + timings.silu_deriv_ms
                + timings.rmsnorm2_bwd_ms
                + timings.rope_bwd_ms
                + timings.rmsnorm1_bwd_ms
                + timings.merge_dx_ms;
            let total = timings.total_ms;
            println!("\n  Category breakdown:");
            println!(
                "    Stage+read+sync: {:>6.2}ms ({:.0}%)",
                stage_ms,
                stage_ms / total * 100.0
            );
            println!(
                "    Async ANE+CPU:   {:>6.2}ms ({:.0}%)",
                async_ms,
                async_ms / total * 100.0
            );
            println!(
                "    Pure CPU:        {:>6.2}ms ({:.0}%)",
                cpu_ms,
                cpu_ms / total * 100.0
            );
        }
    }

    println!("\n=== {} Done ===\n", name);
}

#[test]
#[ignore]
fn profile_600m() {
    profile_config(&ModelConfig::target_600m(), "600M (1536d/20L)");
}

#[test]
#[ignore]
fn profile_800m() {
    profile_config(&ModelConfig::target_800m(), "800M (1792d/24L)");
}

#[test]
#[ignore]
fn profile_1b() {
    profile_config(&ModelConfig::target_1b(), "1B (2048d/28L)");
}

#[test]
#[ignore]
fn profile_all_scales() {
    profile_config(&ModelConfig::gpt_karpathy(), "48.8M (768d/6L)");
    profile_config(&ModelConfig::target_600m(), "600M (1536d/20L)");
    profile_config(&ModelConfig::target_800m(), "800M (1792d/24L)");
    // 1B may take a long time — uncomment if patient
    // profile_config(&ModelConfig::target_1b(), "1B (2048d/28L)");
}
