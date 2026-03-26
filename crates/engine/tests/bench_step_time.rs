//! Benchmark: full training step timing breakdown.
//!
//! Run manually:
//!   cargo test -p engine --test bench_step_time --release -- --ignored --nocapture

use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

#[test]
#[ignore] // Run manually: cargo test -p engine --test bench_step_time --release -- --ignored --nocapture
fn bench_768() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("\n=== Rustane Training Step Benchmark ===");
    println!(
        "Model: gpt_karpathy (NL={}, DIM={}, SEQ={}, VOCAB={})",
        cfg.nlayers, cfg.dim, cfg.seq, cfg.vocab
    );

    // Compile all 10 ANE kernels
    println!("\nCompiling 10 ANE kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    println!("Compiled in {:.2}s", t0.elapsed().as_secs_f32());

    // Create random model state
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let tc = TrainConfig::default();

    // Fixed tokens (deterministic, reproducible)
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // Init Metal Adam optimizer
    let metal_adam = MetalAdam::new().expect("Metal GPU required");

    // Pre-allocated workspaces (zero allocations in steady state)
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    // Warmup (1 step, not timed)
    println!("\nWarmup step...");
    {
        grads.zero_out();
        let _loss = full_model::forward_ws(
            &cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        full_model::backward_ws(
            &cfg,
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
        let scaled_norm = raw_norm * gsc;
        let combined_scale = if scaled_norm > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(0, &tc);
        full_model::update_weights(
            &cfg,
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

    // Benchmark (5 steps)
    println!(
        "\n{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}   {}",
        "step", "total", "fwd", "bwd", "norm", "upd", "loss"
    );
    println!("{}", "-".repeat(80));

    for step in 0..5u32 {
        grads.zero_out();

        let t0 = Instant::now();

        // Forward
        let t_fwd_start = Instant::now();
        let loss = full_model::forward_ws(
            &cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        let t_fwd = t_fwd_start.elapsed();

        // Backward
        let t_bwd_start = Instant::now();
        full_model::backward_ws(
            &cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
        let t_bwd = t_bwd_start.elapsed();

        // Grad norm only (scale fused into Adam GPU kernel)
        let t_norm_start = Instant::now();
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let scaled_norm = raw_norm * gsc;
        let combined_scale = if scaled_norm > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let t_norm = t_norm_start.elapsed();

        // Update weights (Adam with fused grad scaling)
        let t_upd_start = Instant::now();
        let lr = full_model::learning_rate(step + 2, &tc); // +2 because warmup was step 1
        full_model::update_weights(
            &cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 2,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
        let t_upd = t_upd_start.elapsed();

        let total = t0.elapsed();

        println!(
            "{:<6} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms   {:.4}",
            step,
            total.as_secs_f32() * 1000.0,
            t_fwd.as_secs_f32() * 1000.0,
            t_bwd.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_upd.as_secs_f32() * 1000.0,
            loss
        );
    }

    println!("\n=== Benchmark complete ===\n");
}

#[test]
#[ignore]
fn bench_training_step_1024() {
    bench_config(ModelConfig::gpt_1024(), "gpt_1024 (8L, 1024dim)");
}

#[test]
#[ignore]
fn bench_600m() {
    let cfg = ModelConfig::target_600m();
    println!("~{:.0}M params", cfg.param_count() as f64 / 1e6);
    bench_config(cfg, "target_600m (20L, 1536dim)");
}

#[test]
#[ignore]
fn bench_800m() {
    let cfg = ModelConfig::target_800m();
    println!("~{:.0}M params", cfg.param_count() as f64 / 1e6);
    bench_config(cfg, "target_800m (24L, 1792dim)");
}

#[test]
#[ignore]
fn bench_1b() {
    let cfg = ModelConfig::target_1b();
    println!("~{:.0}M params", cfg.param_count() as f64 / 1e6);
    bench_config(cfg, "target_1b (28L, 2048dim)");
}

fn bench_config(cfg: ModelConfig, name: &str) {
    println!("\n=== Rustane Training Step Benchmark ===");
    println!(
        "Model: {} (NL={}, DIM={}, HIDDEN={}, SEQ={}, VOCAB={})",
        name, cfg.nlayers, cfg.dim, cfg.hidden, cfg.seq, cfg.vocab
    );

    println!("\nCompiling 10 ANE kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    println!("Compiled in {:.2}s", t0.elapsed().as_secs_f32());

    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let tc = TrainConfig::default();

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    // Warmup
    println!("\nWarmup step...");
    {
        grads.zero_out();
        let _loss = full_model::forward_ws(
            &cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        full_model::backward_ws(
            &cfg,
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
            &cfg,
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

    println!(
        "\n{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}   {}",
        "step", "total", "fwd", "bwd", "norm", "upd", "loss"
    );
    println!("{}", "-".repeat(80));

    for step in 0..5u32 {
        grads.zero_out();
        let t0 = Instant::now();
        let t_fwd_start = Instant::now();
        let loss = full_model::forward_ws(
            &cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        let t_fwd = t_fwd_start.elapsed();
        let t_bwd_start = Instant::now();
        full_model::backward_ws(
            &cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
        let t_bwd = t_bwd_start.elapsed();
        let t_norm_start = Instant::now();
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let t_norm = t_norm_start.elapsed();
        let t_upd_start = Instant::now();
        let lr = full_model::learning_rate(step + 2, &tc);
        full_model::update_weights(
            &cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 2,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
        let t_upd = t_upd_start.elapsed();
        let total = t0.elapsed();
        println!(
            "{:<6} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms   {:.4}",
            step,
            total.as_secs_f32() * 1000.0,
            t_fwd.as_secs_f32() * 1000.0,
            t_bwd.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_upd.as_secs_f32() * 1000.0,
            loss
        );
    }
    println!("\n=== Benchmark complete ===\n");
}
