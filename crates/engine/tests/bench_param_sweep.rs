//! Parameter sweep: test different dim/hidden/depth/seq combos at 600M/1B/1.5B scale.
//!
//! Each config: compile → 10 training steps (validate loss decreases) → 5 timed steps (timing).
//! Matches bench_scale_correctness approach which is proven stable across all scales.
//!
//! Run all:    cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture
//! Run one:    cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_600m_a
//! By scale:   cargo test -p engine --test bench_param_sweep --release -- --ignored --nocapture sweep_all_600m

use engine::bench_result;
use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::metal_ffn::MetalFFN;
use engine::model::{FfnActivation, ModelConfig};
use std::time::Instant;

/// Build a custom ModelConfig (all MHA, hd=128, vocab=8192).
fn custom_config(
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
) -> ModelConfig {
    ModelConfig {
        dim,
        hidden,
        heads,
        kv_heads: heads,
        hd: 128,
        seq,
        nlayers,
        vocab: 8192,
        q_dim: heads * 128,
        kv_dim: heads * 128,
        gqa_ratio: 1,
        ffn_activation: FfnActivation::SwiGlu,
    }
}

/// Result from a single sweep run.
struct SweepResult {
    name: String,
    params_m: f64,
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
    hd_ratio: f64,
    compile_s: f32,
    ms_per_step: f32,
    ms_fwd: f32,
    ms_bwd: f32,
    ms_upd: f32,
    tok_per_s: f32,
    loss_start: f32,
    loss_end: f32,
    loss_delta: f32,
    all_finite: bool,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

/// Run sweep on a config: 10 training steps (validate) + 5 timed steps (benchmark).
/// Mirrors bench_scale_correctness which is proven stable at all scales with TrainConfig::default().
fn run_sweep(cfg: &ModelConfig, name: &str) -> SweepResult {
    let params_m = cfg.param_count() as f64 / 1e6;
    let hd_ratio = cfg.hidden as f64 / cfg.dim as f64;
    let profile_ffn_cache = env_flag("RUSTANE_PROFILE_FFN_CACHE");
    let requested_gpu_ffn = env_flag("RUSTANE_USE_GPU_FFN");
    let requested_gpu_ffn_bwd_dx = env_flag("RUSTANE_USE_GPU_FFN_BWD_DX");
    let metal_ffn = if requested_gpu_ffn {
        MetalFFN::new(cfg)
    } else {
        None
    };
    let fwd_opts = if let Some(ref metal_ffn) = metal_ffn {
        full_model::ForwardOptions::gpu_ffn(metal_ffn)
    } else {
        full_model::ForwardOptions::default()
    };
    let mut bwd_opts = if requested_gpu_ffn_bwd_dx {
        metal_ffn
            .as_ref()
            .map(full_model::BackwardOptions::gpu_ffn_dx)
            .unwrap_or_default()
    } else {
        full_model::BackwardOptions::default()
    };
    if profile_ffn_cache {
        bwd_opts = bwd_opts.with_ffn_weight_cache();
    }

    println!("\n{}", "=".repeat(60));
    println!(
        "  {name} — {d}d/{h}h/{nl}L/seq{s} — ~{p:.0}M params — h/d={r:.2}x",
        d = cfg.dim,
        h = cfg.hidden,
        nl = cfg.nlayers,
        s = cfg.seq,
        p = params_m,
        r = hd_ratio
    );
    println!("{}", "=".repeat(60));
    println!(
        "  forward path: {}",
        if fwd_opts.use_gpu_ffn {
            "GPU FFN"
        } else if requested_gpu_ffn {
            "GPU FFN requested but unavailable"
        } else {
            "ANE FFN"
        }
    );
    println!(
        "  backward ffn dx: {}",
        if bwd_opts.use_gpu_ffn_dx {
            "GPU"
        } else if requested_gpu_ffn_bwd_dx {
            "requested but unavailable"
        } else {
            "ANE"
        }
    );

    // 1. Compile
    print!("  [1/4] Compiling ANE kernels... ");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(cfg);
    let compile_s = t0.elapsed().as_secs_f32();
    println!("{compile_s:.1}s");

    // 2. Forward validation
    print!("  [2/4] Forward pass... ");
    let mut weights = ModelWeights::random(cfg);
    let tc = TrainConfig::default();
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let loss0 = full_model::forward_ws_with_options(
        cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        fwd_opts,
    );
    assert!(loss0.is_finite(), "initial loss is not finite: {loss0}");
    println!("loss={loss0:.4}");

    // 3. Train 10 steps — loss must decrease
    print!("  [3/4] Training 10 steps... ");
    let mut grads = ModelGrads::zeros(cfg);
    let mut opt = ModelOptState::zeros(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);
    let metal_adam = MetalAdam::new().expect("Metal GPU required");

    let mut losses = vec![loss0];
    for step in 0..10u32 {
        grads.zero_out();
        let loss = full_model::forward_ws_with_options(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
            fwd_opts,
        );
        losses.push(loss);
        full_model::backward_ws_with_options(
            cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
            bwd_opts,
        );
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights(
            cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 1,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
    }
    let final_loss = full_model::forward_ws_with_options(
        cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        fwd_opts,
    );
    losses.push(final_loss);

    let loss_delta = final_loss - loss0;
    let all_finite = losses.iter().all(|l| l.is_finite());
    println!("{loss0:.4} → {final_loss:.4} (delta={loss_delta:+.4})");

    // 4. Timing benchmark (5 steps, report median)
    print!("  [4/4] Timing 5 steps... ");
    let mut step_times = Vec::with_capacity(5);
    let mut fwd_times = Vec::with_capacity(5);
    let mut bwd_times = Vec::with_capacity(5);
    let mut upd_times = Vec::with_capacity(5);
    if profile_ffn_cache {
        bwd_ws.reset_ffn_bwd_w13t_stats();
    }

    for step in 10..15u32 {
        grads.zero_out();
        let t_total = Instant::now();

        let t_fwd = Instant::now();
        let _loss = full_model::forward_ws_with_options(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
            fwd_opts,
        );
        let fwd_ms = t_fwd.elapsed().as_secs_f32() * 1000.0;

        let t_bwd = Instant::now();
        full_model::backward_ws_with_options(
            cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
            bwd_opts,
        );
        let bwd_ms = t_bwd.elapsed().as_secs_f32() * 1000.0;

        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };

        let t_upd = Instant::now();
        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights(
            cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 1,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
        let upd_ms = t_upd.elapsed().as_secs_f32() * 1000.0;

        let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;
        step_times.push(total_ms);
        fwd_times.push(fwd_ms);
        bwd_times.push(bwd_ms);
        upd_times.push(upd_ms);
    }

    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fwd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bwd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    upd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ms_per_step = step_times[2]; // median of 5
    let ms_fwd = fwd_times[2];
    let ms_bwd = bwd_times[2];
    let ms_upd = upd_times[2];
    let tok_per_s = cfg.seq as f32 * 1000.0 / ms_per_step;

    println!(
        "{ms_per_step:.0}ms/step (fwd={ms_fwd:.0} bwd={ms_bwd:.0} upd={ms_upd:.0}) = {tok_per_s:.0} tok/s"
    );
    if profile_ffn_cache {
        let stats = bwd_ws.ffn_bwd_w13t_stats();
        println!(
            "  ffnBwdW13t cache: refreshes={} reuse_hits={} weight_stage={:.1}ms activation_stage={:.1}ms",
            stats.weight_refreshes,
            stats.weight_reuse_hits,
            stats.stage_weights_ms,
            stats.stage_activations_ms,
        );
    }

    if !all_finite {
        println!("  WARNING: NaN/Inf detected!");
    }

    // Write leaderboard-ready JSON
    let mut bench = bench_result::BenchResult {
        schema_version: 1,
        rustane_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: bench_result::git_sha(),
        benchmark: if fwd_opts.use_gpu_ffn {
            format!("sweep_{}_gpu_ffn", name.to_lowercase().replace("-", "_"))
        } else {
            format!("sweep_{}", name.to_lowercase().replace("-", "_"))
        },
        config: bench_result::ModelInfo {
            name: name.to_string(),
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            nlayers: cfg.nlayers,
            seq: cfg.seq,
            params_m,
        },
        results: bench_result::TimingResults {
            ms_per_step,
            ms_fwd,
            ms_bwd,
            ms_upd,
            tok_per_s,
            loss_start: loss0,
            loss_end: final_loss,
            loss_delta,
        },
        loss_trace: losses.clone(),
        hardware: bench_result::collect_hardware_info(),
        submitter: bench_result::Submitter::default(),
        timestamp_utc: bench_result::utc_timestamp(),
        fingerprint: String::new(),
    };
    bench.fingerprint = bench_result::compute_fingerprint(&bench);
    bench_result::write_result(&bench);

    SweepResult {
        name: name.to_string(),
        params_m,
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
        hd_ratio,
        compile_s,
        ms_per_step,
        ms_fwd,
        ms_bwd,
        ms_upd,
        tok_per_s,
        loss_start: loss0,
        loss_end: final_loss,
        loss_delta,
        all_finite,
    }
}

/// Print results table for a set of sweep results.
fn print_results_table(results: &[SweepResult]) {
    println!("\n{}", "=".repeat(130));
    println!("  SWEEP RESULTS SUMMARY");
    println!("{}", "=".repeat(130));
    println!(
        "  {:<10} {:>8} {:>6} {:>6} {:>5} {:>4} {:>4} {:>6} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "name",
        "params",
        "dim",
        "hid",
        "hds",
        "nl",
        "seq",
        "h/d",
        "ms/step",
        "fwd",
        "bwd",
        "upd",
        "tok/s",
        "loss0",
        "lossN",
        "delta"
    );
    println!("  {}", "-".repeat(125));
    for r in results {
        let flag = if !r.all_finite {
            " NaN!"
        } else if r.loss_delta >= 0.0 {
            " BAD!"
        } else {
            ""
        };
        println!(
            "  {:<10} {:>7.0}M {:>6} {:>6} {:>5} {:>4} {:>4} {:>5.2}x {:>8.1}ms {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>7.0} {:>8.4} {:>8.4} {:>+7.4}{}",
            r.name,
            r.params_m,
            r.dim,
            r.hidden,
            r.heads,
            r.nlayers,
            r.seq,
            r.hd_ratio,
            r.ms_per_step,
            r.ms_fwd,
            r.ms_bwd,
            r.ms_upd,
            r.tok_per_s,
            r.loss_start,
            r.loss_end,
            r.loss_delta,
            flag
        );
    }
    println!();

    // Find best tok/s per scale
    let scales = ["600m", "1b-", "1.5b"];
    for scale in &scales {
        let group: Vec<&SweepResult> = results
            .iter()
            .filter(|r| r.name.starts_with(scale) && r.all_finite && r.loss_delta < 0.0)
            .collect();
        if let Some(best) = group
            .iter()
            .max_by(|a, b| a.tok_per_s.partial_cmp(&b.tok_per_s).unwrap())
        {
            let baseline = group.iter().find(|r| r.name.ends_with("-A"));
            if let Some(base) = baseline {
                let pct = (best.tok_per_s - base.tok_per_s) / base.tok_per_s * 100.0;
                println!(
                    "  Best {scale}: {} ({:.0} tok/s, {pct:+.1}% vs baseline)",
                    best.name, best.tok_per_s
                );
            } else {
                println!(
                    "  Best {scale}: {} ({:.0} tok/s)",
                    best.name, best.tok_per_s
                );
            }
        }
    }
}

// ── 600M variants ──────────────────────────────────────────────────────

#[test]
#[ignore]
fn sweep_600m_a() {
    let r = run_sweep(&custom_config(1536, 4096, 12, 20, 512), "600m-A");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_600m_b() {
    let r = run_sweep(&custom_config(1280, 3456, 10, 28, 512), "600m-B");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_600m_c() {
    let r = run_sweep(&custom_config(1792, 4864, 14, 14, 512), "600m-C");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_600m_d() {
    let r = run_sweep(&custom_config(1536, 6144, 12, 16, 512), "600m-D");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_600m_e() {
    let r = run_sweep(&custom_config(1536, 4096, 12, 20, 256), "600m-E");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

// ── 1B variants ────────────────────────────────────────────────────────

#[test]
#[ignore]
fn sweep_1b_a() {
    let r = run_sweep(&custom_config(2048, 5632, 16, 28, 512), "1b-A");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1b_b() {
    let r = run_sweep(&custom_config(1792, 4864, 14, 36, 512), "1b-B");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1b_c() {
    let r = run_sweep(&custom_config(2304, 6144, 18, 20, 512), "1b-C");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1b_d() {
    let r = run_sweep(&custom_config(2048, 8192, 16, 20, 512), "1b-D");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1b_e() {
    let r = run_sweep(&custom_config(2048, 5632, 16, 28, 256), "1b-E");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

// ── 1.5B variants ──────────────────────────────────────────────────────

#[test]
#[ignore]
fn sweep_1_5b_a() {
    let r = run_sweep(&custom_config(2304, 6144, 18, 32, 512), "1.5b-A");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1_5b_b() {
    let r = run_sweep(&custom_config(2048, 5632, 16, 40, 512), "1.5b-B");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1_5b_c() {
    let r = run_sweep(&custom_config(2560, 6912, 20, 24, 512), "1.5b-C");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1_5b_d() {
    let r = run_sweep(&custom_config(2304, 9216, 18, 22, 512), "1.5b-D");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_1_5b_e() {
    let r = run_sweep(&custom_config(2304, 6144, 18, 32, 256), "1.5b-E");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

// ── 3B variants (~55GB RAM, fits 128GB) ────────────────────────────────

#[test]
#[ignore]
fn sweep_3b_a() {
    let r = run_sweep(&custom_config(2560, 6912, 20, 40, 512), "3b-A");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_3b_b() {
    let r = run_sweep(&custom_config(2304, 6144, 18, 48, 512), "3b-B");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_3b_c() {
    let r = run_sweep(&custom_config(3072, 8192, 24, 28, 512), "3b-C");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_3b_d() {
    let r = run_sweep(&custom_config(2560, 10240, 20, 28, 512), "3b-D");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_3b_e() {
    let r = run_sweep(&custom_config(2560, 6912, 20, 40, 256), "3b-E");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

// ── 5B variants (~85GB RAM, tight on 128GB) ────────────────────────────

#[test]
#[ignore]
fn sweep_5b_a() {
    let r = run_sweep(&custom_config(3072, 8192, 24, 44, 512), "5b-A");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_5b_b() {
    let r = run_sweep(&custom_config(2560, 6912, 20, 60, 512), "5b-B");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_5b_c() {
    let r = run_sweep(&custom_config(3584, 9600, 28, 32, 512), "5b-C");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_5b_d() {
    let r = run_sweep(&custom_config(3072, 12288, 24, 32, 512), "5b-D");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_5b_e() {
    let r = run_sweep(&custom_config(3072, 8192, 24, 44, 256), "5b-E");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

// ── 7B+ full training probes (~112GB RAM for 7B) ───────────────────────

#[test]
#[ignore]
fn sweep_7b() {
    // Llama-2-7B shape: dim=4096, hidden=11008, 32 heads, 32 layers
    // ~6.5B params, ~112GB training RAM — tight on 128GB
    let r = run_sweep(&custom_config(4096, 11008, 32, 32, 512), "7b");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

#[test]
#[ignore]
fn sweep_10b() {
    // 10B: dim=4096, hidden=11008, 48 layers
    // ~9.8B params, ~168GB training RAM — will use swap on 128GB
    let r = run_sweep(&custom_config(4096, 11008, 32, 48, 512), "10b");
    assert!(r.all_finite, "NaN/Inf detected");
    assert!(
        r.loss_delta < 0.0,
        "loss did not decrease: delta={}",
        r.loss_delta
    );
}

// ── Grouped sweeps ─────────────────────────────────────────────────────

#[test]
#[ignore]
fn sweep_all_600m() {
    let configs = vec![
        (custom_config(1536, 4096, 12, 20, 512), "600m-A"),
        (custom_config(1280, 3456, 10, 28, 512), "600m-B"),
        (custom_config(1792, 4864, 14, 14, 512), "600m-C"),
        (custom_config(1536, 6144, 12, 16, 512), "600m-D"),
        (custom_config(1536, 4096, 12, 20, 256), "600m-E"),
    ];
    let results: Vec<SweepResult> = configs
        .iter()
        .map(|(cfg, name)| run_sweep(cfg, name))
        .collect();
    print_results_table(&results);
    for r in &results {
        assert!(r.all_finite, "{}: NaN/Inf detected", r.name);
        assert!(
            r.loss_delta < 0.0,
            "{}: loss did not decrease: delta={}",
            r.name,
            r.loss_delta
        );
    }
}

#[test]
#[ignore]
fn sweep_all_1b() {
    let configs = vec![
        (custom_config(2048, 5632, 16, 28, 512), "1b-A"),
        (custom_config(1792, 4864, 14, 36, 512), "1b-B"),
        (custom_config(2304, 6144, 18, 20, 512), "1b-C"),
        (custom_config(2048, 8192, 16, 20, 512), "1b-D"),
        (custom_config(2048, 5632, 16, 28, 256), "1b-E"),
    ];
    let results: Vec<SweepResult> = configs
        .iter()
        .map(|(cfg, name)| run_sweep(cfg, name))
        .collect();
    print_results_table(&results);
    for r in &results {
        assert!(r.all_finite, "{}: NaN/Inf detected", r.name);
        assert!(
            r.loss_delta < 0.0,
            "{}: loss did not decrease: delta={}",
            r.name,
            r.loss_delta
        );
    }
}

#[test]
#[ignore]
fn sweep_all_1_5b() {
    let configs = vec![
        (custom_config(2304, 6144, 18, 32, 512), "1.5b-A"),
        (custom_config(2048, 5632, 16, 40, 512), "1.5b-B"),
        (custom_config(2560, 6912, 20, 24, 512), "1.5b-C"),
        (custom_config(2304, 9216, 18, 22, 512), "1.5b-D"),
        (custom_config(2304, 6144, 18, 32, 256), "1.5b-E"),
    ];
    let results: Vec<SweepResult> = configs
        .iter()
        .map(|(cfg, name)| run_sweep(cfg, name))
        .collect();
    print_results_table(&results);
    for r in &results {
        assert!(r.all_finite, "{}: NaN/Inf detected", r.name);
        assert!(
            r.loss_delta < 0.0,
            "{}: loss did not decrease: delta={}",
            r.name,
            r.loss_delta
        );
    }
}

#[test]
#[ignore]
fn sweep_all_3b() {
    let configs = vec![
        (custom_config(2560, 6912, 20, 40, 512), "3b-A"),
        (custom_config(2304, 6144, 18, 48, 512), "3b-B"),
        (custom_config(3072, 8192, 24, 28, 512), "3b-C"),
        (custom_config(2560, 10240, 20, 28, 512), "3b-D"),
        (custom_config(2560, 6912, 20, 40, 256), "3b-E"),
    ];
    let results: Vec<SweepResult> = configs
        .iter()
        .map(|(cfg, name)| run_sweep(cfg, name))
        .collect();
    print_results_table(&results);
    for r in &results {
        assert!(r.all_finite, "{}: NaN/Inf detected", r.name);
        assert!(
            r.loss_delta < 0.0,
            "{}: loss did not decrease: delta={}",
            r.name,
            r.loss_delta
        );
    }
}

#[test]
#[ignore]
fn sweep_all_5b() {
    let configs = vec![
        (custom_config(3072, 8192, 24, 44, 512), "5b-A"),
        (custom_config(2560, 6912, 20, 60, 512), "5b-B"),
        (custom_config(3584, 9600, 28, 32, 512), "5b-C"),
        (custom_config(3072, 12288, 24, 32, 512), "5b-D"),
        (custom_config(3072, 8192, 24, 44, 256), "5b-E"),
    ];
    let results: Vec<SweepResult> = configs
        .iter()
        .map(|(cfg, name)| run_sweep(cfg, name))
        .collect();
    print_results_table(&results);
    for r in &results {
        assert!(r.all_finite, "{}: NaN/Inf detected", r.name);
        assert!(
            r.loss_delta < 0.0,
            "{}: loss did not decrease: delta={}",
            r.name,
            r.loss_delta
        );
    }
}

#[test]
#[ignore]
fn sweep_full() {
    let configs = vec![
        // 600M
        (custom_config(1536, 4096, 12, 20, 512), "600m-A"),
        (custom_config(1280, 3456, 10, 28, 512), "600m-B"),
        (custom_config(1792, 4864, 14, 14, 512), "600m-C"),
        (custom_config(1536, 6144, 12, 16, 512), "600m-D"),
        (custom_config(1536, 4096, 12, 20, 256), "600m-E"),
        // 1B
        (custom_config(2048, 5632, 16, 28, 512), "1b-A"),
        (custom_config(1792, 4864, 14, 36, 512), "1b-B"),
        (custom_config(2304, 6144, 18, 20, 512), "1b-C"),
        (custom_config(2048, 8192, 16, 20, 512), "1b-D"),
        (custom_config(2048, 5632, 16, 28, 256), "1b-E"),
        // 1.5B
        (custom_config(2304, 6144, 18, 32, 512), "1.5b-A"),
        (custom_config(2048, 5632, 16, 40, 512), "1.5b-B"),
        (custom_config(2560, 6912, 20, 24, 512), "1.5b-C"),
        (custom_config(2304, 9216, 18, 22, 512), "1.5b-D"),
        (custom_config(2304, 6144, 18, 32, 256), "1.5b-E"),
    ];
    let results: Vec<SweepResult> = configs
        .iter()
        .map(|(cfg, name)| run_sweep(cfg, name))
        .collect();
    print_results_table(&results);
    for r in &results {
        assert!(r.all_finite, "{}: NaN/Inf detected", r.name);
        assert!(
            r.loss_delta < 0.0,
            "{}: loss did not decrease: delta={}",
            r.name,
            r.loss_delta
        );
    }
}
