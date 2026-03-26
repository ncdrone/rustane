//! Multi-stream forward throughput probe for m3-ultra.
//!
//! Run:
//!   cargo test -p engine --test bench_forward_multistream --release -- --ignored --nocapture

use engine::bench_result;
use engine::full_model::{self, ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};

fn run_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
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

fn scenarios() -> Vec<(ModelConfig, &'static str, Vec<usize>)> {
    vec![
        (custom_config(3072, 8192, 24, 44, 512), "5B", vec![1, 2, 4]),
        (
            custom_config(5120, 13824, 40, 40, 512),
            "13B",
            vec![1, 2, 4],
        ),
        (custom_config(5120, 13824, 40, 64, 512), "20B", vec![1, 2]),
    ]
}

fn stream_count_from_env() -> usize {
    std::env::var("STREAMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamScenarioResult {
    name: String,
    streams: usize,
    iter_count: usize,
    total_wall_ms: f32,
    aggregate_tok_per_s: f32,
    mean_stream_tok_per_s: f32,
    min_stream_tok_per_s: f32,
    max_stream_tok_per_s: f32,
    peak_rss_mb: f32,
    est_ram_gb: f64,
    loss: Option<f32>,
    success: bool,
    error: Option<String>,
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/pr22_multistream")
}

fn json_path() -> PathBuf {
    results_dir().join("summary.json")
}

fn summary_path() -> PathBuf {
    results_dir().join("summary.md")
}

fn ensure_results_dir() {
    fs::create_dir_all(results_dir()).expect("create results dir");
}

fn rss_mb() -> Option<f32> {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let kb: f32 = String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<f32>()
        .ok()?;
    Some(kb / 1024.0)
}

fn physical_mem_gb() -> Option<f64> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let bytes: f64 = text.trim().parse::<u64>().ok()? as f64;
    Some(bytes / (1024.0 * 1024.0 * 1024.0))
}

fn deterministic_tokens(cfg: &ModelConfig) -> (Vec<u32>, Vec<u32>) {
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    (tokens, targets)
}

fn current_rss_mb() -> Option<f32> {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let kb: f32 = String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<f32>()
        .ok()?;
    Some(kb / 1024.0)
}

fn shared_weight_mem_gb(cfg: &ModelConfig) -> f64 {
    cfg.param_count() as f64 * 4.0 / 1e9
}

fn per_stream_lean_overhead_gb(cfg: &ModelConfig) -> f64 {
    let per_layer_cache_mb = (cfg.dim * cfg.seq * 4 * 5
        + cfg.hidden * cfg.seq * 4 * 3
        + cfg.heads * cfg.seq * cfg.seq * 4) as f64
        / 1e6;
    2.0 * per_layer_cache_mb / 1000.0 + 0.5
}

fn lean_mem_estimate_gb(cfg: &ModelConfig, streams: usize) -> f64 {
    // All streams share one weight set via Arc<ModelWeights>; only workspace-like
    // overhead scales with stream count.
    shared_weight_mem_gb(cfg) + per_stream_lean_overhead_gb(cfg) * streams as f64
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with non-string payload".to_string()
    }
}

fn write_submit_artifact(
    cfg: &ModelConfig,
    name: &str,
    streams: usize,
    result: &StreamScenarioResult,
) {
    let loss = result.loss.unwrap_or(0.0);
    let ms_per_step = result.total_wall_ms / (result.iter_count as f32 * streams as f32);
    let mut bench = bench_result::BenchResult {
        schema_version: 1,
        rustane_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: bench_result::git_sha(),
        benchmark: format!("fwd_{}_multistream_{}x", name.to_lowercase(), streams),
        config: bench_result::ModelInfo {
            name: name.to_string(),
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            nlayers: cfg.nlayers,
            seq: cfg.seq,
            params_m: cfg.param_count() as f64 / 1e6,
        },
        results: bench_result::TimingResults {
            ms_per_step,
            ms_fwd: ms_per_step,
            ms_bwd: 0.0,
            ms_upd: 0.0,
            tok_per_s: result.aggregate_tok_per_s,
            loss_start: loss,
            loss_end: loss,
            loss_delta: 0.0,
        },
        loss_trace: if result.loss.is_some() {
            vec![loss]
        } else {
            Vec::new()
        },
        hardware: bench_result::collect_hardware_info(),
        submitter: bench_result::Submitter::default(),
        timestamp_utc: bench_result::utc_timestamp(),
        fingerprint: String::new(),
    };
    bench.fingerprint = bench_result::compute_fingerprint(&bench);
    bench_result::write_result(&bench);
}

fn run_scenario(
    cfg: &ModelConfig,
    name: &str,
    streams: usize,
    physical_gb: Option<f64>,
) -> StreamScenarioResult {
    const TIMED_ITERS: usize = 3;
    let est_ram_gb = lean_mem_estimate_gb(cfg, streams);
    if let Some(total_gb) = physical_gb {
        if est_ram_gb > total_gb * 0.85 {
            return StreamScenarioResult {
                name: name.to_string(),
                streams,
                iter_count: TIMED_ITERS,
                total_wall_ms: 0.0,
                aggregate_tok_per_s: 0.0,
                mean_stream_tok_per_s: 0.0,
                min_stream_tok_per_s: 0.0,
                max_stream_tok_per_s: 0.0,
                peak_rss_mb: rss_mb().unwrap_or(0.0),
                est_ram_gb,
                loss: None,
                success: false,
                error: Some(format!(
                    "skipped for safety: estimated {:.1}GB exceeds 85% of physical {:.1}GB",
                    est_ram_gb, total_gb
                )),
            };
        }
    }

    let tc = TrainConfig::default();
    let (tokens, targets) = deterministic_tokens(cfg);
    let tokens = Arc::new(tokens);
    let targets = Arc::new(targets);
    let weights = Arc::new(ModelWeights::random(cfg));
    let barrier = Arc::new(Barrier::new(streams + 1));
    let (tx, rx) = mpsc::channel();
    let (loss_tx, loss_rx) = mpsc::channel();

    let run = std::panic::catch_unwind(|| {
        let mut handles = Vec::with_capacity(streams);
        for worker_idx in 0..streams {
            let barrier = barrier.clone();
            let tx = tx.clone();
            let loss_tx = loss_tx.clone();
            let weights = Arc::clone(&weights);
            let tokens = Arc::clone(&tokens);
            let targets = Arc::clone(&targets);
            let cfg = cfg.clone();
            handles.push(thread::spawn(move || {
                let kernels = CompiledKernels::compile_forward_only(&cfg);
                let mut ws = ModelForwardWorkspace::new_lean(&cfg);
                let warmup_loss = full_model::forward_only_ws(
                    &cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut ws,
                );
                if worker_idx == 0 {
                    let _ = loss_tx.send(warmup_loss);
                }

                barrier.wait();
                let t0 = Instant::now();
                for _ in 0..TIMED_ITERS {
                    let _ = full_model::forward_only_ws(
                        &cfg, &kernels, &weights, &tokens, &targets, tc.softcap, &mut ws,
                    );
                }
                let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
                tx.send(elapsed_ms).expect("send stream timing");
            }));
        }
        drop(tx);
        drop(loss_tx);

        barrier.wait();
        let t0 = Instant::now();
        let mut stream_ms = Vec::with_capacity(streams);
        let mut peak_rss = rss_mb().unwrap_or(0.0);
        while stream_ms.len() < streams {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(ms) => stream_ms.push(ms),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            peak_rss = peak_rss.max(rss_mb().unwrap_or(peak_rss));
        }
        let total_wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let loss = loss_rx.recv().ok();

        for handle in handles {
            handle.join().expect("multistream thread panicked");
        }

        (stream_ms, total_wall_ms, peak_rss, loss)
    });

    match run {
        Ok((stream_ms, total_wall_ms, peak_rss, loss)) => {
            let mut stream_tok = Vec::new();
            for ms in &stream_ms {
                let tok = cfg.seq as f32 * TIMED_ITERS as f32 * 1000.0 / *ms;
                stream_tok.push(tok);
            }
            let total_tokens = cfg.seq as f32 * TIMED_ITERS as f32 * streams as f32;
            let aggregate_tok_per_s = total_tokens * 1000.0 / total_wall_ms;
            let mean_stream_tok_per_s = stream_tok.iter().sum::<f32>() / stream_tok.len() as f32;
            let min_stream_tok_per_s = stream_tok.iter().copied().fold(f32::INFINITY, f32::min);
            let max_stream_tok_per_s = stream_tok.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            StreamScenarioResult {
                name: name.to_string(),
                streams,
                iter_count: TIMED_ITERS,
                total_wall_ms,
                aggregate_tok_per_s,
                mean_stream_tok_per_s,
                min_stream_tok_per_s,
                max_stream_tok_per_s,
                peak_rss_mb: peak_rss,
                est_ram_gb,
                loss,
                success: true,
                error: None,
            }
        }
        Err(payload) => StreamScenarioResult {
            name: name.to_string(),
            streams,
            iter_count: TIMED_ITERS,
            total_wall_ms: 0.0,
            aggregate_tok_per_s: 0.0,
            mean_stream_tok_per_s: 0.0,
            min_stream_tok_per_s: 0.0,
            max_stream_tok_per_s: 0.0,
            peak_rss_mb: rss_mb().unwrap_or(0.0),
            est_ram_gb,
            loss: None,
            success: false,
            error: Some(panic_payload_to_string(payload)),
        },
    }
}

fn write_summary(results: &[StreamScenarioResult]) {
    let mut summary = String::new();
    summary.push_str("# PR #22 Multi-stream Forward Summary\n\n");
    summary.push_str("| scale | streams | aggregate tok/s | mean per-stream tok/s | min | max | total wall(ms) | peak RSS(MB) |\n");
    summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for r in results {
        if r.success {
            let _ = writeln!(
                summary,
                "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} |",
                r.name,
                r.streams,
                r.aggregate_tok_per_s,
                r.mean_stream_tok_per_s,
                r.min_stream_tok_per_s,
                r.max_stream_tok_per_s,
                r.total_wall_ms,
                r.peak_rss_mb
            );
        } else {
            let _ = writeln!(
                summary,
                "| {} | {} | FAIL | FAIL | FAIL | FAIL | FAIL | {:.0} |",
                r.name, r.streams, r.peak_rss_mb
            );
        }
    }

    summary.push_str("\n## Scaling\n\n");
    summary.push_str("| scale | streams | scaling vs 1-stream |\n");
    summary.push_str("| --- | ---: | ---: |\n");
    for scale in ["5B", "13B", "20B"] {
        let base = results
            .iter()
            .find(|r| r.name == scale && r.streams == 1 && r.success);
        for r in results.iter().filter(|r| r.name == scale && r.success) {
            let scale_ratio = base
                .map(|b| r.aggregate_tok_per_s / b.aggregate_tok_per_s)
                .unwrap_or(0.0);
            let _ = writeln!(
                summary,
                "| {} | {} | {:.2}x |",
                r.name, r.streams, scale_ratio
            );
        }
    }

    let any_superlinear = results
        .iter()
        .any(|r| r.streams > 1 && r.success && r.aggregate_tok_per_s > 0.0);
    let recommendation = if any_superlinear {
        "both"
    } else {
        "more FFN optimization"
    };
    let _ = writeln!(
        summary,
        "\n## Recommendation\n\nFocus next on **{}**.\n",
        recommendation
    );

    ensure_results_dir();
    fs::write(summary_path(), summary).expect("write summary");
}

fn load_results() -> Vec<StreamScenarioResult> {
    if let Ok(text) = fs::read_to_string(json_path()) {
        serde_json::from_str(&text).unwrap_or_default()
    } else {
        Vec::new()
    }
}

fn write_results(results: &[StreamScenarioResult]) {
    let json = serde_json::to_string_pretty(results).expect("serialize json");
    fs::write(json_path(), json).expect("write json");
    write_summary(results);
}

fn run_named_set(configs: Vec<(ModelConfig, &'static str)>) {
    let _guard = run_lock().lock().unwrap();
    ensure_results_dir();
    let physical_gb = physical_mem_gb();
    let streams = stream_count_from_env();

    let mut results = load_results();
    let mut submit_candidate: Option<(ModelConfig, &'static str, StreamScenarioResult)> = None;
    for (cfg, name) in configs {
        results.retain(|r| !(r.name == name && r.streams == streams));
        let result = run_scenario(&cfg, name, streams, physical_gb);
        if result.success {
            submit_candidate = Some((cfg.clone(), name, result.clone()));
        }
        results.push(result);
    }
    results.sort_by(|a, b| a.name.cmp(&b.name).then(a.streams.cmp(&b.streams)));
    write_results(&results);
    if let Some((cfg, name, result)) = submit_candidate {
        write_submit_artifact(&cfg, name, streams, &result);
    }
}

#[test]
#[ignore]
fn bench_forward_multistream() {
    let _guard = run_lock().lock().unwrap();
    ensure_results_dir();
    let physical_gb = physical_mem_gb();
    let mut results = Vec::new();
    for (cfg, name, stream_counts) in scenarios() {
        for streams in stream_counts {
            results.push(run_scenario(&cfg, name, streams, physical_gb));
            write_results(&results);
        }
    }
    assert!(!results.is_empty(), "expected multistream results");
}

#[test]
#[ignore]
fn bench_forward_multistream_20b_x4() {
    let _guard = run_lock().lock().unwrap();
    ensure_results_dir();
    let physical_gb = physical_mem_gb();
    let cfg = custom_config(5120, 13824, 40, 64, 512);
    let result = run_scenario(&cfg, "20B", 4, physical_gb);

    let mut results = load_results();
    results.retain(|r| !(r.name == "20B" && r.streams == 4));
    results.push(result);
    results.sort_by(|a, b| a.name.cmp(&b.name).then(a.streams.cmp(&b.streams)));
    write_results(&results);

    assert!(
        results.iter().any(|r| r.name == "20B" && r.streams == 4),
        "expected 20B x4 result"
    );
}

#[test]
#[ignore]
fn diagnose_30b_x4_bringup() {
    let _guard = run_lock().lock().unwrap();
    let cfg = custom_config(6144, 16384, 48, 64, 512);
    let weights = Arc::new(ModelWeights::random(&cfg));
    println!("diagnose 30B x4 sequential bring-up");
    println!(
        "rss after weights: {:.0} MB",
        current_rss_mb().unwrap_or(0.0)
    );

    let mut workers = Vec::new();
    for worker_idx in 0..4usize {
        println!("worker {}: compile kernels", worker_idx + 1);
        let result = std::panic::catch_unwind(|| {
            let kernels = CompiledKernels::compile(&cfg);
            println!(
                "worker {}: compiled, rss {:.0} MB",
                worker_idx + 1,
                current_rss_mb().unwrap_or(0.0)
            );
            let mut ws = ModelForwardWorkspace::new_lean(&cfg);
            println!(
                "worker {}: workspace ready, rss {:.0} MB",
                worker_idx + 1,
                current_rss_mb().unwrap_or(0.0)
            );
            let (tokens, targets) = deterministic_tokens(&cfg);
            let _ = full_model::forward_only_ws(
                &cfg,
                &kernels,
                &weights,
                &tokens,
                &targets,
                TrainConfig::default().softcap,
                &mut ws,
            );
            println!(
                "worker {}: warmup ok, rss {:.0} MB",
                worker_idx + 1,
                current_rss_mb().unwrap_or(0.0)
            );
            (kernels, ws)
        });

        match result {
            Ok(worker) => workers.push(worker),
            Err(payload) => {
                println!(
                    "worker {} failed: {} (rss {:.0} MB)",
                    worker_idx + 1,
                    panic_payload_to_string(payload),
                    current_rss_mb().unwrap_or(0.0)
                );
                return;
            }
        }
    }

    println!("all 4 workers brought up successfully");
}

#[test]
#[ignore]
fn forward_7b_multistream() {
    run_named_set(vec![(custom_config(4096, 11008, 32, 32, 512), "7B")]);
}

#[test]
#[ignore]
fn forward_10b_multistream() {
    run_named_set(vec![(custom_config(4096, 11008, 32, 48, 512), "10B")]);
}

#[test]
#[ignore]
fn forward_30b_multistream() {
    run_named_set(vec![(custom_config(5120, 13824, 40, 96, 512), "30B")]);
}

#[test]
#[ignore]
fn forward_ladder_multistream() {
    run_named_set(vec![
        (custom_config(3072, 8192, 24, 44, 512), "5B"),
        (custom_config(4096, 11008, 32, 32, 512), "7B"),
        (custom_config(4096, 11008, 32, 48, 512), "10B"),
        (custom_config(5120, 13824, 40, 40, 512), "13B"),
        (custom_config(5120, 13824, 40, 48, 512), "15B"),
        (custom_config(5120, 13824, 40, 64, 512), "20B"),
    ]);
}

#[test]
#[ignore]
fn forward_ceiling_multistream() {
    run_named_set(vec![
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
    ]);
}
