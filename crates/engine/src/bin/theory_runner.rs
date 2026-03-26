use std::thread;
use std::time::{Duration, Instant};

use engine::bench_result;
use engine::full_model::{ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::model::ModelConfig;
use engine::parallel_bench::{ParallelBenchRequest, ParallelBenchRunner, ShardPolicy};

fn install_abort_on_panic() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        std::process::abort();
    }));
}

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

fn use_lean_workspace() -> bool {
    std::env::var("USE_LEAN_WORKSPACE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v != 0)
        .unwrap_or(true)
}

fn run_forward_probe(cfg: &ModelConfig, name: &str) {
    let params_b = cfg.param_count() as f64 / 1e9;
    let use_lean = use_lean_workspace();
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

    print!("  [1/3] Compiling ANE kernels... ");
    let request = ParallelBenchRequest::from_env(ShardPolicy::FailFast)
        .expect("parse shard requests from env");
    let (mut runner, compile_s) = ParallelBenchRunner::compile(cfg, request, use_lean)
        .expect("compile parallel benchmark runner");
    println!("{compile_s:.1}s");
    if runner.resolved().mode_label() != "baseline" {
        println!("      mode={}", runner.resolved().mode_label());
    }
    for note in &runner.resolved().notes {
        println!("      note: {note}");
    }

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
    let fwd_ms = times[1];
    println!("{fwd_ms:.0}ms (loss={loss:.4})");
    assert!(loss.is_finite(), "forward pass produced NaN/Inf at {name}");

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
}

fn main() {
    install_abort_on_panic();

    let mode = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: theory_runner <noop|panic-main|panic-thread|sleep-ms>");
        std::process::exit(2);
    });

    match mode.as_str() {
        "noop" => {}
        "panic-main" => panic!("intentional main-thread panic"),
        "panic-thread" => {
            let handle = thread::spawn(|| {
                panic!("intentional worker-thread panic");
            });
            let _ = handle.join();
        }
        "forward-7b" => run_forward_probe(&custom_config(4096, 11008, 32, 32, 512), "7B"),
        "forward-10b" => run_forward_probe(&custom_config(4096, 11008, 32, 48, 512), "10B"),
        "sleep-ms" => {
            let ms = std::env::args()
                .nth(2)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1000);
            thread::sleep(Duration::from_millis(ms));
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }
}
