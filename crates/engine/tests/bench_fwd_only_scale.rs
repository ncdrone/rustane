//! Forward-only scale probes: test ANE kernel compilation + forward pass at 5B–20B+.
//!
//! No backward pass or optimizer — only needs weights + forward workspace.
//! Memory: ~5 bytes/param (weights=4 + workspace overhead).
//!
//! Run all:   cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture
//! Run one:   cargo test -p engine --test bench_fwd_only_scale --release -- --ignored --nocapture fwd_10b

use engine::bench_result;
use engine::full_model::{ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::model::ModelConfig;
use engine::parallel_bench::{ParallelBenchRequest, ParallelBenchRunner, ShardPolicy};
use std::time::Instant;

fn use_lean_workspace() -> bool {
    std::env::var("USE_LEAN_WORKSPACE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v != 0)
        .unwrap_or(true)
}

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
fn run_fwd_probe(cfg: &ModelConfig, name: &str, policy: ShardPolicy) -> FwdResult {
    let params_b = cfg.param_count() as f64 / 1e9;
    let use_lean = use_lean_workspace();
    // Estimate: weights (4B/param) + workspace.
    // Lean mode: 2 cache slots instead of nlayers (saves ~462MB/layer at dim=5120).
    let per_layer_cache_mb = (cfg.dim * cfg.seq * 4 * 5
        + cfg.hidden * cfg.seq * 4 * 3
        + cfg.heads * cfg.seq * cfg.seq * 4) as f64
        / 1e6;
    let cache_layers = if use_lean { 2 } else { cfg.nlayers };
    let mem_est_gb = params_b * 4.0 + cache_layers as f64 * per_layer_cache_mb / 1000.0 + 0.5;

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
    let request = ParallelBenchRequest::from_env(policy).expect("parse shard requests from env");
    let (mut runner, compile_s) = ParallelBenchRunner::compile(cfg, request, use_lean)
        .expect("compile parallel benchmark runner");
    println!("{compile_s:.1}s");
    if runner.resolved().mode_label() != "baseline" {
        println!("      mode={}", runner.resolved().mode_label());
    }
    for note in &runner.resolved().notes {
        println!("      note: {note}");
    }

    // 2. Allocate weights + workspace
    print!("  [2/3] Allocating {:.1}GB... ", mem_est_gb);
    let t0 = Instant::now();
    let weights = ModelWeights::random(cfg);
    let mut fwd_ws = if use_lean {
        ModelForwardWorkspace::new_lean(cfg)
    } else {
        ModelForwardWorkspace::new(cfg)
    };
    let alloc_s = t0.elapsed().as_secs_f32();
    println!("{alloc_s:.1}s (lean={use_lean})");

    let tc = TrainConfig::default();
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    // 3. Forward passes (1 warmup + 3 timed)
    print!("  [3/3] Forward pass (1 warmup + 3 timed)... ");
    let _warmup_loss = runner.forward_loss(&weights, &tokens, &targets, tc.softcap, &mut fwd_ws);

    let mut times = Vec::with_capacity(3);
    let mut loss = 0.0f32;
    for _ in 0..3 {
        let t0 = Instant::now();
        loss = runner.forward_loss(&weights, &tokens, &targets, tc.softcap, &mut fwd_ws);
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
        benchmark: format!(
            "fwd_{}{}",
            name.to_lowercase(),
            runner.resolved().benchmark_suffix()
        ),
        config: bench_result::ModelInfo {
            name: format!("{}{}", name, runner.resolved().benchmark_suffix()),
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
    let r = run_fwd_probe(
        &custom_config(3072, 8192, 24, 44, 512),
        "5B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_7b() {
    // ~Llama-2-7B shape
    let r = run_fwd_probe(
        &custom_config(4096, 11008, 32, 32, 512),
        "7B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_10b() {
    let r = run_fwd_probe(
        &custom_config(4096, 11008, 32, 48, 512),
        "10B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_13b() {
    // ~Llama-2-13B shape
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 40, 512),
        "13B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_15b() {
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 48, 512),
        "15B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_20b() {
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 64, 512),
        "20B",
        ShardPolicy::FailFast,
    );
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
        let r = run_fwd_probe(cfg, name, ShardPolicy::AutoAdjustNearest);
        let ok = r.loss.is_finite();
        results.push(r);
        if !ok {
            println!("\n  STOPPED: {name} produced NaN/Inf — likely OOM or ANE limit");
            break;
        }
    }

    print_fwd_table(&results);
}

// ── 25B+ individual probes ────────────────────────────────────────────

#[test]
#[ignore]
fn fwd_25b() {
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 80, 512),
        "25B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_30b() {
    // dim=5120, hidden=13824, 40 heads, 96 layers → ~30.5B
    // Keeps SDPA spatial width = 15872 < 16384 (ANE limit)
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 96, 512),
        "30B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_40b() {
    // dim=5120, hidden=13824, 40 heads, 128 layers → ~40.6B
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 128, 512),
        "40B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_50b() {
    // dim=5120, hidden=13824, 40 heads, 160 layers → ~50.8B
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 160, 512),
        "50B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_60b() {
    // dim=5120, hidden=13824, 40 heads, 192 layers → ~60.9B
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 192, 512),
        "60B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_70b() {
    // dim=5120, hidden=13824, 40 heads, 224 layers → ~71.1B
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 224, 512),
        "70B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_80b() {
    // dim=5120, hidden=13824, 40 heads, 256 layers → ~81.2B
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 256, 512),
        "80B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_100b() {
    // dim=5120, hidden=13824, 40 heads, 320 layers → ~101.5B
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 320, 512),
        "100B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_110b() {
    // dim=5120, hidden=13824, 40 heads, 352 layers → ~111.7B (~447GB weights)
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 352, 512),
        "110B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_115b() {
    // dim=5120, hidden=13824, 40 heads, 368 layers → ~116.7B (~467GB weights)
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 368, 512),
        "115B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_120b() {
    // dim=5120, hidden=13824, 40 heads, 384 layers → ~121.8B (~487GB weights)
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 384, 512),
        "120B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

#[test]
#[ignore]
fn fwd_122b() {
    // dim=5120, hidden=13824, 40 heads, 390 layers → ~123.7B (~496GB)
    // CEILING on 512GB M3 Ultra — 124B (503GB) OOM kills
    let r = run_fwd_probe(
        &custom_config(5120, 13824, 40, 390, 512),
        "122B",
        ShardPolicy::FailFast,
    );
    assert!(r.loss.is_finite());
}

/// Push from 25B to 125B. Find the ceiling on 512GB M3 Ultra.
/// All configs use dim=5120 to stay within ANE spatial width limit (16384).
#[test]
#[ignore]
fn fwd_find_ceiling() {
    let configs: Vec<(ModelConfig, &str)> = vec![
        (custom_config(5120, 13824, 40, 80, 512), "25B"),
        (custom_config(5120, 13824, 40, 96, 512), "30B"),
        (custom_config(5120, 13824, 40, 128, 512), "40B"),
        (custom_config(5120, 13824, 40, 160, 512), "50B"),
        (custom_config(5120, 13824, 40, 192, 512), "60B"),
        (custom_config(5120, 13824, 40, 224, 512), "70B"),
        (custom_config(5120, 13824, 40, 256, 512), "80B"),
        (custom_config(5120, 13824, 40, 320, 512), "100B"),
        (custom_config(5120, 13824, 40, 352, 512), "110B"),
        (custom_config(5120, 13824, 40, 368, 512), "115B"),
        (custom_config(5120, 13824, 40, 384, 512), "120B"),
        (custom_config(5120, 13824, 40, 390, 512), "122B"),
    ];

    let mut results = Vec::new();
    for (cfg, name) in &configs {
        let r = run_fwd_probe(cfg, name, ShardPolicy::AutoAdjustNearest);
        let ok = r.loss.is_finite();
        results.push(r);
        if !ok {
            println!("\n  STOPPED: {name} — likely OOM");
            break;
        }
    }

    print_fwd_table(&results);
}
