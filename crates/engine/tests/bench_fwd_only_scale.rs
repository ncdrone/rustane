//! Forward-only scale probes: test ANE kernel compilation + forward pass at 5B–20B+.
//!
//! No backward pass or optimizer — only needs weights + forward workspace.
//! Memory: ~5 bytes/param (weights=4 + workspace overhead).
//!
//! Run all:   cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture
//! Run one:   cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_10b

use engine::bench_result;
use engine::full_model::{self, ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::layer::CompiledKernels;
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

struct FwdResult {
    name: String,
    params_b: f64,
    dim: usize,
    hidden: usize,
    nlayers: usize,
    compile_s: f32,
    alloc_s: f32,
    fwd_ms: f32,
    loss: f32,
    mem_est_gb: f64,
}

/// Compile kernels, allocate weights, run 3 forward passes, report median timing.
fn run_fwd_probe(cfg: &ModelConfig, name: &str) -> FwdResult {
    let params_b = cfg.param_count() as f64 / 1e9;
    // Estimate: weights (4B/param) + fwd workspace (caches + buffers)
    let per_layer_cache_mb = (cfg.dim * cfg.seq * 4 * 5
        + cfg.hidden * cfg.seq * 4 * 3
        + cfg.heads * cfg.seq * cfg.seq * 4) as f64
        / 1e6;
    let mem_est_gb = params_b * 4.0 + cfg.nlayers as f64 * per_layer_cache_mb / 1000.0 + 0.5;

    println!("\n{}", "=".repeat(70));
    println!(
        "  {name} — {d}d/{h}h/{nl}L/seq{s} — {p:.2}B params — est. {m:.1}GB",
        d = cfg.dim,
        h = cfg.hidden,
        nl = cfg.nlayers,
        s = cfg.seq,
        p = params_b,
        m = mem_est_gb
    );
    println!("{}", "=".repeat(70));

    // 1. Compile
    print!("  [1/3] Compiling ANE kernels... ");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(cfg);
    let compile_s = t0.elapsed().as_secs_f32();
    println!("{compile_s:.1}s");

    // 2. Allocate weights + workspace
    print!("  [2/3] Allocating {:.1}GB... ", mem_est_gb);
    let t0 = Instant::now();
    let weights = ModelWeights::random(cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let alloc_s = t0.elapsed().as_secs_f32();
    println!("{alloc_s:.1}s");

    let tc = TrainConfig::default();
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // 3. Forward passes (1 warmup + 3 timed)
    print!("  [3/3] Forward pass (1 warmup + 3 timed)... ");
    let _warmup_loss = full_model::forward_ws(
        cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
    );

    let mut times = Vec::with_capacity(3);
    let mut loss = 0.0f32;
    for _ in 0..3 {
        let t0 = Instant::now();
        loss = full_model::forward_ws(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        times.push(t0.elapsed().as_secs_f32() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fwd_ms = times[1]; // median

    println!("{fwd_ms:.0}ms (loss={loss:.4})");
    assert!(loss.is_finite(), "forward pass produced NaN/Inf at {name}");

    // Write leaderboard-ready JSON
    let mut bench = bench_result::BenchResult {
        schema_version: 1,
        rustane_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: bench_result::git_sha(),
        benchmark: format!("fwd_{}", name.to_lowercase()),
        config: bench_result::ModelInfo {
            name: name.to_string(),
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            nlayers: cfg.nlayers,
            seq: cfg.seq,
            params_m: params_b * 1000.0,
        },
        results: bench_result::TimingResults {
            ms_per_step: fwd_ms,
            ms_fwd: fwd_ms,
            ms_bwd: 0.0,
            ms_upd: 0.0,
            tok_per_s: cfg.seq as f32 * 1000.0 / fwd_ms,
            loss_start: loss,
            loss_end: loss,
            loss_delta: 0.0,
        },
        loss_trace: vec![loss],
        hardware: bench_result::collect_hardware_info(),
        submitter: bench_result::Submitter::default(),
        timestamp_utc: bench_result::utc_timestamp(),
        fingerprint: String::new(),
    };
    bench.fingerprint = bench_result::compute_fingerprint(&bench);
    bench_result::write_result(&bench);

    FwdResult {
        name: name.to_string(),
        params_b,
        dim: cfg.dim,
        hidden: cfg.hidden,
        nlayers: cfg.nlayers,
        compile_s,
        alloc_s,
        fwd_ms,
        loss,
        mem_est_gb,
    }
}

fn print_fwd_table(results: &[FwdResult]) {
    println!("\n{}", "=".repeat(100));
    println!("  FORWARD-ONLY SCALE RESULTS");
    println!("{}", "=".repeat(100));
    println!(
        "  {:<12} {:>8} {:>6} {:>6} {:>4} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "name", "params", "dim", "hid", "nl", "compile", "alloc", "fwd", "loss", "est.RAM"
    );
    println!("  {}", "-".repeat(95));
    for r in results {
        println!(
            "  {:<12} {:>7.2}B {:>6} {:>6} {:>4} {:>7.1}s {:>7.1}s {:>9.0}ms {:>8.4} {:>7.1}GB",
            r.name,
            r.params_b,
            r.dim,
            r.hidden,
            r.nlayers,
            r.compile_s,
            r.alloc_s,
            r.fwd_ms,
            r.loss,
            r.mem_est_gb
        );
    }
    println!();
}

// ── Individual probes ──────────────────────────────────────────────────

#[test]
#[ignore]
fn fwd_5b() {
    let r = run_fwd_probe(&custom_config(3072, 8192, 24, 44, 512), "5B");
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_7b() {
    // ~Llama-2-7B shape
    let r = run_fwd_probe(&custom_config(4096, 11008, 32, 32, 512), "7B");
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_10b() {
    let r = run_fwd_probe(&custom_config(4096, 11008, 32, 48, 512), "10B");
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_13b() {
    // ~Llama-2-13B shape
    let r = run_fwd_probe(&custom_config(5120, 13824, 40, 40, 512), "13B");
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_15b() {
    let r = run_fwd_probe(&custom_config(5120, 13824, 40, 48, 512), "15B");
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_20b() {
    let r = run_fwd_probe(&custom_config(5120, 13824, 40, 64, 512), "20B");
    assert!(r.loss.is_finite());
}

// ── Grouped probes ─────────────────────────────────────────────────────

/// Progressive scale test: 5B → 7B → 10B → 13B → 15B → 20B.
#[test]
#[ignore]
fn fwd_scale_ladder() {
    let configs: Vec<(ModelConfig, &str)> = vec![
        (custom_config(3072, 8192, 24, 44, 512), "5B"),
        (custom_config(4096, 11008, 32, 32, 512), "7B"),
        (custom_config(4096, 11008, 32, 48, 512), "10B"),
        (custom_config(5120, 13824, 40, 40, 512), "13B"),
        (custom_config(5120, 13824, 40, 48, 512), "15B"),
        (custom_config(5120, 13824, 40, 64, 512), "20B"),
    ];

    let mut results = Vec::new();
    for (cfg, name) in &configs {
        let r = run_fwd_probe(cfg, name);
        let ok = r.loss.is_finite();
        results.push(r);
        if !ok {
            println!("\n  STOPPED: {name} produced NaN/Inf — likely OOM or ANE limit");
            break;
        }
    }

    print_fwd_table(&results);
}

/// Push to the absolute limit: 25B and 30B.
/// These will eat nearly all 128GB RAM. Close all other apps first.
#[test]
#[ignore]
fn fwd_find_ceiling() {
    let configs: Vec<(ModelConfig, &str)> = vec![
        (custom_config(5120, 13824, 40, 80, 512), "25B"),
        (custom_config(6144, 16384, 48, 64, 512), "30B"),
    ];

    let mut results = Vec::new();
    for (cfg, name) in &configs {
        let r = run_fwd_probe(cfg, name);
        let ok = r.loss.is_finite();
        results.push(r);
        if !ok {
            println!("\n  STOPPED: {name} — likely OOM");
            break;
        }
    }

    print_fwd_table(&results);
}
