//! Stress test: find the maximum model scale that compiles + runs on ANE.
//!
//! Run: cargo test -p engine --test stress_max_scale --release -- --ignored --nocapture

use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

fn make_config(dim: usize, hidden: usize, heads: usize, nlayers: usize) -> ModelConfig {
    ModelConfig {
        dim,
        hidden,
        heads,
        kv_heads: heads,
        hd: 128,
        seq: 512,
        nlayers,
        vocab: 8192,
        q_dim: dim,
        kv_dim: dim,
        gqa_ratio: 1,
    }
}

fn try_scale(name: &str, cfg: &ModelConfig) -> bool {
    let params = cfg.param_count() as f64 / 1e9;
    let ffn_sp = 2 * cfg.seq + 3 * cfg.hidden;
    let ffn_in_mb = (cfg.dim * ffn_sp * 4) as f64 / 1e6;
    let weight_gb = cfg.param_count() as f64 * 4.0 / 1e9;
    let total_gb = cfg.param_count() as f64 * 20.0 / 1e9; // w+g+m+v+ws

    println!(
        "\n--- {name}: {:.2}B params, {}d/{}h/{}L ---",
        params, cfg.dim, cfg.hidden, cfg.nlayers
    );
    println!(
        "    ffnFused IOSurf: {:.0} MB, weight mem: {:.1} GB, total est: {:.1} GB",
        ffn_in_mb, weight_gb, total_gb
    );

    // 1. Compile kernels
    print!("    [1] Compile... ");
    let t = Instant::now();
    let kernels = match std::panic::catch_unwind(|| CompiledKernels::compile(cfg)) {
        Ok(k) => k,
        Err(e) => {
            println!(
                "PANIC: {:?}",
                e.downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown")
            );
            return false;
        }
    };
    println!("OK ({:.1}s)", t.elapsed().as_secs_f32());

    // 2. Allocate weights
    print!("    [2] Allocate... ");
    let t = Instant::now();
    let mut weights = ModelWeights::random(cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    println!("OK ({:.1}s)", t.elapsed().as_secs_f32());

    // 3. Forward
    print!("    [3] Forward... ");
    let tc = TrainConfig::default();
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let t = Instant::now();
    let loss = full_model::forward_ws(
        cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
    );
    let fwd_ms = t.elapsed().as_secs_f32() * 1000.0;
    if !loss.is_finite() {
        println!("FAIL: loss={loss}");
        return false;
    }
    println!("OK (loss={loss:.4}, {fwd_ms:.0}ms)");

    // 4. Backward
    print!("    [4] Backward... ");
    let mut grads = ModelGrads::zeros(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);
    let t = Instant::now();
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
    let bwd_ms = t.elapsed().as_secs_f32() * 1000.0;
    println!("OK ({bwd_ms:.0}ms)");

    // 5. One full step
    print!("    [5] Full step... ");
    let mut opt = ModelOptState::zeros(cfg);
    let metal_adam = MetalAdam::new().expect("Metal");
    let gsc = 1.0 / tc.loss_scale;
    let raw_norm = full_model::grad_norm(&grads);
    let combined_scale = if raw_norm * gsc > tc.grad_clip {
        tc.grad_clip / raw_norm
    } else {
        gsc
    };
    let lr = full_model::learning_rate(0, &tc);
    let t = Instant::now();
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
    let upd_ms = t.elapsed().as_secs_f32() * 1000.0;
    let total = fwd_ms + bwd_ms + upd_ms;
    println!("OK ({total:.0}ms/step = {fwd_ms:.0} fwd + {bwd_ms:.0} bwd + {upd_ms:.0} upd)");

    println!("    PASS: {name} at {:.2}B — {total:.0}ms/step", params);
    true
}

#[test]
#[ignore]
fn find_max_scale() {
    println!("\n=== Finding Maximum ANE Training Scale ===\n");

    let configs = vec![
        ("2B", make_config(2560, 6912, 20, 32)), // ~2.6B
        ("3B", make_config(2816, 7680, 22, 36)), // ~3.5B
        ("4B", make_config(3072, 8192, 24, 36)), // ~4.2B
        ("5B", make_config(3328, 8960, 26, 40)), // ~5.5B
    ];

    let mut last_pass = "1.5B".to_string();
    for (name, cfg) in &configs {
        if !try_scale(name, cfg) {
            println!("\n=== BROKE AT {name} ===");
            println!("Last passing: {last_pass}");
            println!(
                "Breaking config: {}d/{}h/{}L",
                cfg.dim, cfg.hidden, cfg.nlayers
            );
            return;
        }
        last_pass = name.to_string();
    }

    println!("\n=== ALL PASSED through 5B! ===");
}
