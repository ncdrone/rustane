//! PR #22 head-only addendum: lean forward-only utilization probe.
//!
//! Run:
//!   cargo test -p engine --test bench_forward_utilization_head --release -- --ignored --nocapture pr22_head_lean_forward_only

use engine::full_model::{self, ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

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

fn compare_scales() -> Vec<(ModelConfig, &'static str)> {
    vec![
        (custom_config(3072, 8192, 24, 44, 512), "5B"),
        (custom_config(5120, 13824, 40, 40, 512), "13B"),
        (custom_config(5120, 13824, 40, 64, 512), "20B"),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeanScaleResult {
    name: String,
    params_b: f64,
    compile_s: f32,
    alloc_s: f32,
    median_fwd_ms: f32,
    p5_fwd_ms: f32,
    p95_fwd_ms: f32,
    tok_per_s: f32,
    loss: f32,
    est_ram_gb: f64,
    rss_mb_after_compile: f32,
    rss_mb_after_alloc: f32,
    rss_mb_after_warmup: f32,
    rss_mb_peak_timed: f32,
    success: bool,
    error: Option<String>,
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/pr22_compare/head_lean")
}

fn json_path() -> PathBuf {
    results_dir().join("scale_ladder.json")
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

fn mem_estimate_gb(cfg: &ModelConfig) -> f64 {
    let params_b = cfg.param_count() as f64 / 1e9;
    let per_layer_cache_mb = (cfg.dim * cfg.seq * 4 * 5
        + cfg.hidden * cfg.seq * 4 * 3
        + cfg.heads * cfg.seq * cfg.seq * 4) as f64
        / 1e6;
    params_b * 4.0 + 2.0 * per_layer_cache_mb / 1000.0 + 0.5
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

fn percentile(values: &[f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() as f32 * pct / 100.0) as usize).min(values.len() - 1);
    values[idx]
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

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with non-string payload".to_string()
    }
}

fn run_probe(cfg: &ModelConfig, name: &str, physical_gb: Option<f64>) -> LeanScaleResult {
    let params_b = cfg.param_count() as f64 / 1e9;
    let est_ram_gb = mem_estimate_gb(cfg);
    if let Some(total_gb) = physical_gb {
        if est_ram_gb > total_gb * 0.90 {
            return LeanScaleResult {
                name: name.to_string(),
                params_b,
                compile_s: 0.0,
                alloc_s: 0.0,
                median_fwd_ms: 0.0,
                p5_fwd_ms: 0.0,
                p95_fwd_ms: 0.0,
                tok_per_s: 0.0,
                loss: 0.0,
                est_ram_gb,
                rss_mb_after_compile: rss_mb().unwrap_or(0.0),
                rss_mb_after_alloc: rss_mb().unwrap_or(0.0),
                rss_mb_after_warmup: rss_mb().unwrap_or(0.0),
                rss_mb_peak_timed: rss_mb().unwrap_or(0.0),
                success: false,
                error: Some(format!(
                    "skipped for safety: estimated {:.1}GB exceeds 90% of physical {:.1}GB",
                    est_ram_gb, total_gb
                )),
            };
        }
    }

    let tc = TrainConfig::default();
    let (tokens, targets) = deterministic_tokens(cfg);
    let mut result = LeanScaleResult {
        name: name.to_string(),
        params_b,
        compile_s: 0.0,
        alloc_s: 0.0,
        median_fwd_ms: 0.0,
        p5_fwd_ms: 0.0,
        p95_fwd_ms: 0.0,
        tok_per_s: 0.0,
        loss: 0.0,
        est_ram_gb,
        rss_mb_after_compile: 0.0,
        rss_mb_after_alloc: 0.0,
        rss_mb_after_warmup: 0.0,
        rss_mb_peak_timed: 0.0,
        success: false,
        error: None,
    };

    let run = std::panic::catch_unwind(|| {
        let t0 = Instant::now();
        let kernels = CompiledKernels::compile(cfg);
        let compile_s = t0.elapsed().as_secs_f32();
        let rss_after_compile = rss_mb().unwrap_or(0.0);

        let t0 = Instant::now();
        let weights = ModelWeights::random(cfg);
        let mut fwd_ws = ModelForwardWorkspace::new_lean(cfg);
        let alloc_s = t0.elapsed().as_secs_f32();
        let rss_after_alloc = rss_mb().unwrap_or(0.0);

        let warmup_loss = full_model::forward_only_ws(
            cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
        );
        assert!(warmup_loss.is_finite(), "warmup loss is not finite");
        let rss_after_warmup = rss_mb().unwrap_or(0.0);

        let mut times = Vec::with_capacity(5);
        let mut loss = warmup_loss;
        let mut rss_peak = rss_after_warmup;
        for _ in 0..5 {
            let t0 = Instant::now();
            loss = full_model::forward_only_ws(
                cfg,
                &kernels,
                &weights,
                &tokens,
                &targets,
                tc.softcap,
                &mut fwd_ws,
            );
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            assert!(loss.is_finite(), "timed loss is not finite");
            times.push(ms);
            rss_peak = rss_peak.max(rss_mb().unwrap_or(rss_peak));
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        (
            compile_s,
            alloc_s,
            rss_after_compile,
            rss_after_alloc,
            rss_after_warmup,
            rss_peak,
            times,
            loss,
        )
    });

    match run {
        Ok((
            compile_s,
            alloc_s,
            rss_after_compile,
            rss_after_alloc,
            rss_after_warmup,
            rss_peak,
            times,
            loss,
        )) => {
            result.compile_s = compile_s;
            result.alloc_s = alloc_s;
            result.rss_mb_after_compile = rss_after_compile;
            result.rss_mb_after_alloc = rss_after_alloc;
            result.rss_mb_after_warmup = rss_after_warmup;
            result.rss_mb_peak_timed = rss_peak;
            result.p5_fwd_ms = percentile(&times, 5.0);
            result.median_fwd_ms = percentile(&times, 50.0);
            result.p95_fwd_ms = percentile(&times, 95.0);
            result.loss = loss;
            result.tok_per_s = cfg.seq as f32 * 1000.0 / result.median_fwd_ms;
            result.success = true;
        }
        Err(payload) => {
            result.error = Some(panic_payload_to_string(payload));
            result.rss_mb_after_compile = rss_mb().unwrap_or(0.0);
            result.rss_mb_after_alloc = rss_mb().unwrap_or(0.0);
            result.rss_mb_after_warmup = rss_mb().unwrap_or(0.0);
            result.rss_mb_peak_timed = rss_mb().unwrap_or(0.0);
        }
    }

    result
}

fn write_summary(results: &[LeanScaleResult]) {
    let mut summary = String::new();
    summary.push_str("# PR #22 Head Lean Forward Summary\n\n");
    summary.push_str("| name | params(B) | compile(s) | alloc(s) | median fwd(ms) | tok/s | est RAM(GB) | rss peak(MB) |\n");
    summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for r in results {
        if r.success {
            let _ = writeln!(
                summary,
                "| {} | {:.2} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} |",
                r.name,
                r.params_b,
                r.compile_s,
                r.alloc_s,
                r.median_fwd_ms,
                r.tok_per_s,
                r.est_ram_gb,
                r.rss_mb_peak_timed
            );
        } else {
            let _ = writeln!(
                summary,
                "| {} | {:.2} | FAIL | FAIL | FAIL | FAIL | {:.1} | {:.0} |",
                r.name, r.params_b, r.est_ram_gb, r.rss_mb_peak_timed
            );
        }
    }
    ensure_results_dir();
    fs::write(summary_path(), summary).expect("write summary");
}

#[test]
#[ignore]
fn pr22_head_lean_forward_only() {
    let _guard = run_lock().lock().unwrap();
    ensure_results_dir();
    let physical_gb = physical_mem_gb();
    let mut results = Vec::new();
    for (cfg, name) in compare_scales() {
        let result = run_probe(&cfg, name, physical_gb);
        let stop = !result.success;
        results.push(result);
        let json = serde_json::to_string_pretty(&results).expect("serialize json");
        fs::write(json_path(), json).expect("write json");
        write_summary(&results);
        if stop {
            break;
        }
    }
    assert!(!results.is_empty(), "expected lean head results");
}
