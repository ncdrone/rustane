//! Memory leak detection: run 50 training steps and track RSS growth.
//! Run: cargo test -p engine --test memory_leak_test --release -- --ignored --nocapture

use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

fn get_rss_mb() -> f64 {
    use std::process::Command;
    let pid = std::process::id();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps command failed");
    let rss_kb: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0);
    rss_kb / 1024.0
}

#[test]
#[ignore]
fn detect_memory_leak() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("\n=== Memory Leak Detection (workspace path) ===");
    println!("Model: 6L 768D 512S 8192V\n");

    let rss_before_kernels = get_rss_mb();
    println!("RSS before kernel compile: {:.1} MB", rss_before_kernels);

    let kernels = CompiledKernels::compile(&cfg);
    let rss_after_kernels = get_rss_mb();
    println!(
        "RSS after kernel compile:  {:.1} MB (+{:.1} MB)",
        rss_after_kernels,
        rss_after_kernels - rss_before_kernels
    );

    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let tc = TrainConfig::default();
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let rss_after_init = get_rss_mb();
    println!(
        "RSS after model init:      {:.1} MB (+{:.1} MB for weights+grads+opt+ws)",
        rss_after_init,
        rss_after_init - rss_after_kernels
    );

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // Warmup
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
            1.0,
        );
    }

    let rss_after_warmup = get_rss_mb();
    println!(
        "RSS after warmup step:     {:.1} MB (+{:.1} MB)",
        rss_after_warmup,
        rss_after_warmup - rss_after_init
    );

    // Run 50 steps, log RSS every 10
    println!(
        "\n{:>5}  {:>8}  {:>8}  {:>8}  {}",
        "step", "RSS(MB)", "delta", "loss", "time"
    );
    println!("{}", "-".repeat(55));

    let rss_baseline = get_rss_mb();
    let mut rss_samples: Vec<f64> = Vec::new();

    for step in 0..50u32 {
        grads.zero_out();
        let t = Instant::now();
        let loss = full_model::forward_ws(
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
        let elapsed = t.elapsed().as_secs_f32() * 1000.0;

        let rss = get_rss_mb();
        rss_samples.push(rss);

        if step % 10 == 0 || step == 49 {
            println!(
                "{:>5}  {:>7.1}  {:>+7.1}  {:>8.4}  {:.0}ms",
                step,
                rss,
                rss - rss_baseline,
                loss,
                elapsed
            );
        }
    }

    let rss_final = get_rss_mb();
    println!("\n=== LEAK ANALYSIS ===");
    println!("RSS baseline (after warmup): {:.1} MB", rss_baseline);
    println!("RSS final (after step 49):   {:.1} MB", rss_final);
    println!(
        "Growth over 50 steps:        {:.1} MB",
        rss_final - rss_baseline
    );

    // Check growth between step 10 and step 49 (ignore initial stabilization)
    let rss_10 = rss_samples[10];
    let rss_49 = rss_samples[49];
    let growth_per_step = (rss_49 - rss_10) / 39.0;
    println!(
        "Growth step 10→49:           {:.2} MB ({:.2} MB/step)",
        rss_49 - rss_10,
        growth_per_step
    );

    if growth_per_step > 0.5 {
        println!(
            "WARNING: Possible memory leak ({:.2} MB/step)",
            growth_per_step
        );
    } else {
        println!("OK: No significant leak detected");
    }

    // Projected overnight (72000 steps)
    println!(
        "\nProjected RSS at 72000 steps: {:.0} MB ({:.1} GB)",
        rss_baseline + growth_per_step * 72000.0,
        (rss_baseline + growth_per_step * 72000.0) / 1024.0
    );
}

/// Isolate leak source: run 50 steps with fwd+bwd only (NO Adam/Metal).
#[test]
#[ignore]
fn isolate_leak_no_adam() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("\n=== Memory Leak Isolation: No Adam/Metal ===");
    println!("Model: 6L 768D 512S 8192V\n");

    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let tc = TrainConfig::default();
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // Warmup
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
    }

    let rss_baseline = get_rss_mb();
    println!("RSS baseline: {:.1} MB\n", rss_baseline);
    println!(
        "{:>5}  {:>8}  {:>8}  {}",
        "step", "RSS(MB)", "delta", "time"
    );
    println!("{}", "-".repeat(40));

    let mut rss_samples: Vec<f64> = Vec::new();
    for step in 0..50u32 {
        grads.zero_out();
        let t = Instant::now();
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
        let elapsed = t.elapsed().as_secs_f32() * 1000.0;
        let rss = get_rss_mb();
        rss_samples.push(rss);
        if step % 10 == 0 || step == 49 {
            println!(
                "{:>5}  {:>7.1}  {:>+7.1}  {:.0}ms",
                step,
                rss,
                rss - rss_baseline,
                elapsed
            );
        }
    }
    let growth = (rss_samples[49] - rss_samples[10]) / 39.0;
    println!("\nGrowth step 10→49: {:.2} MB/step", growth);
    println!("→ Source: ANE (no Metal Adam in this test)");
}

/// Extended stability test: 200 steps to confirm RSS plateau.
#[test]
#[ignore]
fn extended_stability_200() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("\n=== Extended Stability Test (200 steps) ===");
    println!("Model: 6L 768D 512S 8192V\n");

    let kernels = CompiledKernels::compile(&cfg);
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let tc = TrainConfig::default();
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // Warmup
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
            1.0,
        );
    }

    let rss_baseline = get_rss_mb();
    println!("RSS baseline: {:.1} MB\n", rss_baseline);
    println!(
        "{:>5}  {:>8}  {:>8}  {:>8}  {}",
        "step", "RSS(MB)", "delta", "loss", "time"
    );
    println!("{}", "-".repeat(55));

    let mut rss_samples: Vec<f64> = Vec::new();
    for step in 0..200u32 {
        grads.zero_out();
        let t = Instant::now();
        let loss = full_model::forward_ws(
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
        let elapsed = t.elapsed().as_secs_f32() * 1000.0;

        let rss = get_rss_mb();
        rss_samples.push(rss);

        if step % 25 == 0 || step == 199 {
            println!(
                "{:>5}  {:>7.1}  {:>+7.1}  {:>8.4}  {:.0}ms",
                step,
                rss,
                rss - rss_baseline,
                loss,
                elapsed
            );
        }
    }

    // Check last 100 steps for sustained growth
    let rss_100 = rss_samples[100];
    let rss_199 = rss_samples[199];
    let late_growth = (rss_199 - rss_100) / 99.0;
    println!("\n=== STABILITY ANALYSIS ===");
    println!("RSS at step 100:  {:.1} MB", rss_100);
    println!("RSS at step 199:  {:.1} MB", rss_199);
    println!(
        "Growth step 100→199: {:.2} MB ({:.3} MB/step)",
        rss_199 - rss_100,
        late_growth
    );
    if late_growth.abs() < 0.1 {
        println!("STABLE: No sustained leak (< 0.1 MB/step in late phase)");
    } else {
        println!(
            "WARNING: Sustained growth of {:.3} MB/step in late phase",
            late_growth
        );
        println!(
            "Projected RSS at 72000 steps: {:.0} MB ({:.1} GB)",
            rss_199 + late_growth * 71800.0,
            (rss_199 + late_growth * 71800.0) / 1024.0
        );
    }
}
