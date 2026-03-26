//! Forward utilization diagnostics: full-model scale ladder + layer breakdown + kernel wall-vs-hw timing.
//!
//! Run build check:
//!   cargo test -p engine --test bench_forward_utilization --no-run
//!
//! Run single-stream scale ladder:
//!   cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture forward_scale_single_stream
//!
//! Run layer breakdown:
//!   cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture forward_scale_layer_breakdown
//!
//! Run kernel wall-vs-hw:
//!   cargo test -p engine --test bench_forward_utilization --release -- --ignored --nocapture forward_scale_kernel_hw_vs_wall

use ane_bridge::ane::{Graph, Shape, TensorData};
use engine::cpu::rmsnorm;
use engine::full_model::{self, ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::kernels::{dyn_matmul, ffn_fused, sdpa_fwd};
use engine::layer::{self, CompiledKernels, ForwardTimings, LayerWeights};
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const PR22_BASE_SHA: &str = "9a568eca5340492b9b5bac54fb4e5211225996b1";
const ANE_MAX_SPATIAL_WIDTH: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BranchFlavor {
    Base,
    Head,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultsMode {
    Default,
    Compare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForwardMode {
    Common,
}

#[derive(Debug, Clone, Copy)]
struct DispatchInfo {
    ffn_mode: &'static str,
    ffn_dispatches_per_layer: usize,
    forward_dispatches_per_layer: usize,
}

fn run_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn branch_flavor() -> BranchFlavor {
    static FLAVOR: OnceLock<BranchFlavor> = OnceLock::new();
    *FLAVOR.get_or_init(|| {
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output();
        let sha = output
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()
                } else {
                    None
                }
            })
            .unwrap_or_default();
        if sha.trim() == PR22_BASE_SHA {
            BranchFlavor::Base
        } else {
            BranchFlavor::Head
        }
    })
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

fn scale_ladder() -> Vec<(ModelConfig, &'static str)> {
    vec![
        (custom_config(2048, 5632, 16, 28, 512), "1b-A"),
        (custom_config(2304, 6144, 18, 32, 512), "1.5b-A"),
        (custom_config(2560, 6912, 20, 40, 512), "3b-A"),
        (custom_config(3072, 8192, 24, 44, 512), "5B"),
        (custom_config(4096, 11008, 32, 32, 512), "7B"),
        (custom_config(4096, 11008, 32, 48, 512), "10B"),
        (custom_config(5120, 13824, 40, 40, 512), "13B"),
        (custom_config(5120, 13824, 40, 64, 512), "20B"),
    ]
}

fn representative_scales() -> Vec<(ModelConfig, &'static str)> {
    vec![
        (custom_config(2048, 5632, 16, 28, 512), "1b-A"),
        (custom_config(3072, 8192, 24, 44, 512), "5B"),
        (custom_config(5120, 13824, 40, 40, 512), "13B"),
    ]
}

fn compare_scales() -> Vec<(ModelConfig, &'static str)> {
    vec![
        (custom_config(3072, 8192, 24, 44, 512), "5B"),
        (custom_config(5120, 13824, 40, 40, 512), "13B"),
        (custom_config(5120, 13824, 40, 64, 512), "20B"),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScaleLadderResult {
    name: String,
    params_b: f64,
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
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
    ffn_mode: String,
    ffn_dispatches_per_layer: usize,
    forward_dispatches_per_layer: usize,
    forward_dispatches_per_token: f32,
    dispatches_per_forward: usize,
    dispatches_per_token: f32,
    success: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerBreakdownResult {
    name: String,
    rmsnorm1_ms: f32,
    stage_sdpa_ms: f32,
    ane_sdpa_ms: f32,
    read_sdpa_ms: f32,
    stage_wo_ms: f32,
    ane_wo_ms: f32,
    read_wo_ms: f32,
    residual_rmsnorm2_ms: f32,
    stage_ffn_ms: f32,
    ane_ffn_ms: f32,
    read_ffn_ms: f32,
    total_ms: f32,
    ane_total_ms: f32,
    stage_total_ms: f32,
    read_total_ms: f32,
    cpu_total_ms: f32,
    ane_pct: f32,
    stage_pct: f32,
    read_pct: f32,
    cpu_pct: f32,
    success: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KernelHwWallResult {
    scale: String,
    kernel: String,
    wall_median_us: f64,
    wall_p5_us: f64,
    wall_p95_us: f64,
    hw_median_ns: f64,
    hw_p5_ns: f64,
    hw_p95_ns: f64,
    overhead_us: f64,
    overhead_pct: f64,
    success: bool,
    error: Option<String>,
}

fn results_dir() -> PathBuf {
    if let Ok(root) = env::var("RUSTANE_FORWARD_UTIL_RESULTS") {
        PathBuf::from(root)
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/forward_utilization")
    }
}

fn results_root_for(mode: ResultsMode) -> PathBuf {
    match mode {
        ResultsMode::Default => {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/forward_utilization")
        }
        ResultsMode::Compare => {
            let side = match branch_flavor() {
                BranchFlavor::Base => "base",
                BranchFlavor::Head => "head",
            };
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../results/pr22_compare/{side}"))
        }
    }
}

fn scale_json_path() -> PathBuf {
    results_dir().join("scale_ladder.json")
}

fn layer_json_path() -> PathBuf {
    results_dir().join("layer_breakdown.json")
}

fn kernel_json_path() -> PathBuf {
    results_dir().join("kernel_hw_vs_wall.json")
}

fn summary_path() -> PathBuf {
    results_dir().join("summary.md")
}

fn ensure_results_dir() {
    fs::create_dir_all(results_dir()).expect("create results dir");
}

fn with_results_root<T>(root: PathBuf, f: impl FnOnce() -> T) -> T {
    // SAFETY: all callers take the global test mutex first, so there is no
    // concurrent environment mutation across benchmark tests in this process.
    unsafe { env::set_var("RUSTANE_FORWARD_UTIL_RESULTS", &root) };
    let out = f();
    // SAFETY: same reasoning as above.
    unsafe { env::remove_var("RUSTANE_FORWARD_UTIL_RESULTS") };
    out
}

fn write_json<T: Serialize>(path: &Path, value: &T) {
    ensure_results_dir();
    let json = serde_json::to_string_pretty(value).expect("serialize json");
    fs::write(path, json).expect("write json");
}

fn maybe_read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn percentile_f32(values: &[f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() as f32 * pct / 100.0) as usize).min(values.len() - 1);
    values[idx]
}

fn percentile_f64(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() as f64 * pct / 100.0) as usize).min(values.len() - 1);
    values[idx]
}

fn mem_estimate_gb(cfg: &ModelConfig) -> f64 {
    let params_b = cfg.param_count() as f64 / 1e9;
    let per_layer_cache_mb = (cfg.dim * cfg.seq * 4 * 5
        + cfg.hidden * cfg.seq * 4 * 3
        + cfg.heads * cfg.seq * cfg.seq * 4) as f64
        / 1e6;
    params_b * 4.0 + cfg.nlayers as f64 * per_layer_cache_mb / 1000.0 + 0.5
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

fn deterministic_tokens(cfg: &ModelConfig) -> (Vec<u32>, Vec<u32>) {
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    (tokens, targets)
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i.wrapping_mul(17).wrapping_add(11)) % 251) as f32 * 0.001 - 0.125)
        .collect()
}

fn ffn_needs_decomposition(cfg: &ModelConfig) -> bool {
    2 * cfg.seq + 3 * cfg.hidden > ANE_MAX_SPATIAL_WIDTH
}

fn ffn_can_use_dual_w13(cfg: &ModelConfig) -> bool {
    cfg.seq + 2 * cfg.hidden <= ANE_MAX_SPATIAL_WIDTH
}

fn dual_separate_spatial_width(seq: usize, oc: usize) -> usize {
    seq + 2 * oc
}

fn build_dual_separate(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + 2 * oc;
    let mut g = Graph::new();

    let input = g.placeholder(Shape {
        batch: 1,
        channels: ic,
        height: 1,
        width: sp,
    });
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let acts_r = g.reshape(
        acts,
        Shape {
            batch: 1,
            channels: 1,
            height: ic,
            width: seq,
        },
    );
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);

    let wts1 = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let wts1_r = g.reshape(
        wts1,
        Shape {
            batch: 1,
            channels: 1,
            height: ic,
            width: oc,
        },
    );
    let mm1 = g.matrix_multiplication(acts_t, wts1_r, false, false);
    let mm1_t = g.transpose(mm1, [0, 1, 3, 2]);
    let out1 = g.reshape(
        mm1_t,
        Shape {
            batch: 1,
            channels: oc,
            height: 1,
            width: seq,
        },
    );

    let wts2 = g.slice(input, [0, 0, 0, seq + oc], [1, ic, 1, oc]);
    let wts2_r = g.reshape(
        wts2,
        Shape {
            batch: 1,
            channels: 1,
            height: ic,
            width: oc,
        },
    );
    let mm2 = g.matrix_multiplication(acts_t, wts2_r, false, false);
    let mm2_t = g.transpose(mm2, [0, 1, 3, 2]);
    let out2 = g.reshape(
        mm2_t,
        Shape {
            batch: 1,
            channels: oc,
            height: 1,
            width: seq,
        },
    );

    let _out = g.concat(&[out1, out2], 1);
    g
}

fn dispatch_info(cfg: &ModelConfig, flavor: BranchFlavor) -> DispatchInfo {
    if flavor == BranchFlavor::Head && ffn_needs_decomposition(cfg) {
        if ffn_can_use_dual_w13(cfg) {
            DispatchInfo {
                ffn_mode: "dual_w13_plus_w2",
                ffn_dispatches_per_layer: 2,
                forward_dispatches_per_layer: 4,
            }
        } else {
            DispatchInfo {
                ffn_mode: "w1_w3_w2",
                ffn_dispatches_per_layer: 3,
                forward_dispatches_per_layer: 5,
            }
        }
    } else {
        DispatchInfo {
            ffn_mode: "fused",
            ffn_dispatches_per_layer: 1,
            forward_dispatches_per_layer: 3,
        }
    }
}

fn stage_spatial(
    dst: &mut [f32],
    channels: usize,
    sp_width: usize,
    src: &[f32],
    src_width: usize,
    sp_offset: usize,
) {
    for c in 0..channels {
        let dst_row = c * sp_width;
        let src_row = c * src_width;
        dst[dst_row + sp_offset..dst_row + sp_offset + src_width]
            .copy_from_slice(&src[src_row..src_row + src_width]);
    }
}

fn write_summary() {
    let scale_results: Vec<ScaleLadderResult> =
        maybe_read_json(&scale_json_path()).unwrap_or_default();
    let layer_results: Vec<LayerBreakdownResult> =
        maybe_read_json(&layer_json_path()).unwrap_or_default();
    let kernel_results: Vec<KernelHwWallResult> =
        maybe_read_json(&kernel_json_path()).unwrap_or_default();

    let mut summary = String::new();
    summary.push_str("# Forward Utilization Summary\n\n");

    if !scale_results.is_empty() {
        summary.push_str("## Full-model scale ladder\n\n");
        summary.push_str("| name | params(B) | compile(s) | alloc(s) | median fwd(ms) | tok/s | est RAM(GB) | rss compile(MB) | rss alloc(MB) | rss warm(MB) | rss peak(MB) | dispatches |\n");
        summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &scale_results {
            if r.success {
                let _ = writeln!(
                    summary,
                    "| {} | {:.2} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} | {:.0} | {:.0} | {} |",
                    r.name,
                    r.params_b,
                    r.compile_s,
                    r.alloc_s,
                    r.median_fwd_ms,
                    r.tok_per_s,
                    r.est_ram_gb,
                    r.rss_mb_after_compile,
                    r.rss_mb_after_alloc,
                    r.rss_mb_after_warmup,
                    r.rss_mb_peak_timed,
                    r.dispatches_per_forward
                );
            } else {
                let _ = writeln!(
                    summary,
                    "| {} | {:.2} | FAIL | FAIL | FAIL | FAIL | {:.1} | {:.0} | {:.0} | {:.0} | {:.0} | {} |",
                    r.name,
                    r.params_b,
                    r.est_ram_gb,
                    r.rss_mb_after_compile,
                    r.rss_mb_after_alloc,
                    r.rss_mb_after_warmup,
                    r.rss_mb_peak_timed,
                    r.dispatches_per_forward
                );
            }
        }
        summary.push('\n');

        summary.push_str("## RSS checkpoints\n\n");
        summary.push_str("| name | after compile | after alloc | after warmup | peak timed |\n");
        summary.push_str("| --- | ---: | ---: | ---: | ---: |\n");
        for r in &scale_results {
            let _ = writeln!(
                summary,
                "| {} | {:.0} MB | {:.0} MB | {:.0} MB | {:.0} MB |",
                r.name,
                r.rss_mb_after_compile,
                r.rss_mb_after_alloc,
                r.rss_mb_after_warmup,
                r.rss_mb_peak_timed
            );
        }
        summary.push('\n');
    }

    if !layer_results.is_empty() {
        summary.push_str("## Single-layer bucket shares\n\n");
        summary.push_str("| scale | total(ms) | ane(ms) | stage(ms) | read(ms) | cpu(ms) | ane% | stage% | read% | cpu% |\n");
        summary
            .push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &layer_results {
            if r.success {
                let _ = writeln!(
                    summary,
                    "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1}% | {:.1}% | {:.1}% | {:.1}% |",
                    r.name,
                    r.total_ms,
                    r.ane_total_ms,
                    r.stage_total_ms,
                    r.read_total_ms,
                    r.cpu_total_ms,
                    r.ane_pct,
                    r.stage_pct,
                    r.read_pct,
                    r.cpu_pct
                );
            } else {
                let _ = writeln!(
                    summary,
                    "| {} | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |",
                    r.name
                );
            }
        }
        summary.push('\n');
    }

    if !kernel_results.is_empty() {
        summary.push_str("## Per-kernel wall vs hardware\n\n");
        summary.push_str(
            "| scale | kernel | wall median(us) | hw median(ns) | overhead(us) | overhead% |\n",
        );
        summary.push_str("| --- | --- | ---: | ---: | ---: | ---: |\n");
        for r in &kernel_results {
            if r.success {
                let _ = writeln!(
                    summary,
                    "| {} | {} | {:.1} | {:.0} | {:.1} | {:.1}% |",
                    r.scale,
                    r.kernel,
                    r.wall_median_us,
                    r.hw_median_ns,
                    r.overhead_us,
                    r.overhead_pct
                );
            } else {
                let _ = writeln!(
                    summary,
                    "| {} | {} | FAIL | FAIL | FAIL | FAIL |",
                    r.scale, r.kernel
                );
            }
        }
        summary.push('\n');
    }

    let failures = collect_failures(&scale_results, &layer_results, &kernel_results);
    if !failures.is_empty() {
        summary.push_str("## Recorded failures\n\n");
        for failure in failures {
            let _ = writeln!(summary, "- {}", failure);
        }
        summary.push('\n');
    }

    let bottleneck = classify_bottleneck(&scale_results, &layer_results, &kernel_results);
    let recommendation =
        recommend_next_action(&scale_results, &layer_results, &kernel_results, &bottleneck);
    let _ = writeln!(
        summary,
        "## Conclusion\n\nCurrent classification: **{}**.\n",
        bottleneck
    );
    let _ = writeln!(
        summary,
        "## Recommendation\n\nNext action: **{}**.\n",
        recommendation
    );

    ensure_results_dir();
    fs::write(summary_path(), summary).expect("write summary");
}

fn classify_bottleneck(
    scale_results: &[ScaleLadderResult],
    layer_results: &[LayerBreakdownResult],
    kernel_results: &[KernelHwWallResult],
) -> &'static str {
    if scale_results.iter().any(|r| !r.success)
        || layer_results.iter().any(|r| !r.success)
        || kernel_results.iter().any(|r| !r.success)
    {
        return "mixed";
    }

    let layer_1b = layer_results.iter().find(|r| r.name == "1b-A");
    let layer_13b = layer_results.iter().find(|r| r.name == "13B");
    if matches!((layer_1b, layer_13b), (Some(a), Some(b)) if a.ane_pct < 50.0 && b.ane_pct < 60.0) {
        return "staging/readback dominated";
    }

    let high_scale_overheads: Vec<f64> = kernel_results
        .iter()
        .filter(|r| {
            (r.scale == "5B" || r.scale == "13B")
                && (r.kernel == "wo_fwd" || r.kernel.starts_with("ffn_"))
        })
        .map(|r| r.overhead_pct)
        .collect();
    if !high_scale_overheads.is_empty()
        && high_scale_overheads.iter().copied().sum::<f64>() / high_scale_overheads.len() as f64
            > 25.0
    {
        return "staging/readback dominated";
    }

    let avg_ane_pct = if layer_results.is_empty() {
        0.0
    } else {
        layer_results.iter().map(|r| r.ane_pct).sum::<f32>() / layer_results.len() as f32
    };
    let avg_overhead_pct = if kernel_results.is_empty() {
        100.0
    } else {
        kernel_results.iter().map(|r| r.overhead_pct).sum::<f64>() / kernel_results.len() as f64
    };
    if avg_ane_pct > 65.0 && avg_overhead_pct < 15.0 {
        return "ANE-compute dominated";
    }

    if let (Some(first), Some(last)) = (scale_results.first(), scale_results.last()) {
        let tok_scale = if first.tok_per_s > 0.0 {
            last.tok_per_s / first.tok_per_s
        } else {
            0.0
        };
        let rss_scale = if first.rss_mb_peak_timed > 0.0 {
            last.rss_mb_peak_timed / first.rss_mb_peak_timed
        } else {
            0.0
        };
        if tok_scale < 1.5 && rss_scale > 2.0 {
            return "mixed";
        }
    }

    "mixed"
}

fn recommend_next_action(
    scale_results: &[ScaleLadderResult],
    _layer_results: &[LayerBreakdownResult],
    _kernel_results: &[KernelHwWallResult],
    bottleneck: &str,
) -> &'static str {
    if scale_results.iter().any(|r| !r.success)
        || _layer_results.iter().any(|r| !r.success)
        || _kernel_results.iter().any(|r| !r.success)
    {
        return "graph shaping";
    }

    match bottleneck {
        "staging/readback dominated" => "request/dispatch reduction",
        "ANE-compute dominated" => "graph shaping",
        _ => {
            if let (Some(first), Some(last)) = (scale_results.first(), scale_results.last()) {
                let tok_scale = if first.tok_per_s > 0.0 {
                    last.tok_per_s / first.tok_per_s
                } else {
                    0.0
                };
                let rss_scale = if first.rss_mb_peak_timed > 0.0 {
                    last.rss_mb_peak_timed / first.rss_mb_peak_timed
                } else {
                    0.0
                };
                if tok_scale < 1.5 && rss_scale > 2.0 {
                    "memory/workspace reduction"
                } else {
                    "graph shaping"
                }
            } else {
                "graph shaping"
            }
        }
    }
}

fn collect_failures(
    scale_results: &[ScaleLadderResult],
    layer_results: &[LayerBreakdownResult],
    kernel_results: &[KernelHwWallResult],
) -> Vec<String> {
    let mut failures = Vec::new();
    for r in scale_results.iter().filter(|r| !r.success) {
        failures.push(format!(
            "scale {}: {}",
            r.name,
            r.error.as_deref().unwrap_or("unknown error")
        ));
    }
    for r in layer_results.iter().filter(|r| !r.success) {
        failures.push(format!(
            "layer {}: {}",
            r.name,
            r.error.as_deref().unwrap_or("unknown error")
        ));
    }
    for r in kernel_results.iter().filter(|r| !r.success) {
        failures.push(format!(
            "kernel {} / {}: {}",
            r.scale,
            r.kernel,
            r.error.as_deref().unwrap_or("unknown error")
        ));
    }
    failures
}

fn scale_probe(cfg: &ModelConfig, name: &str, safety_ram_gb: Option<f64>) -> ScaleLadderResult {
    let params_b = cfg.param_count() as f64 / 1e9;
    let est_ram_gb = mem_estimate_gb(cfg);
    let dispatch = dispatch_info(cfg, branch_flavor());
    if let Some(physical_gb) = safety_ram_gb {
        if est_ram_gb > physical_gb * 0.90 {
            return ScaleLadderResult {
                name: name.to_string(),
                params_b,
                dim: cfg.dim,
                hidden: cfg.hidden,
                heads: cfg.heads,
                nlayers: cfg.nlayers,
                seq: cfg.seq,
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
                ffn_mode: dispatch.ffn_mode.to_string(),
                ffn_dispatches_per_layer: dispatch.ffn_dispatches_per_layer,
                forward_dispatches_per_layer: dispatch.forward_dispatches_per_layer,
                forward_dispatches_per_token: dispatch.forward_dispatches_per_layer as f32
                    / cfg.seq as f32,
                dispatches_per_forward: dispatch.forward_dispatches_per_layer * cfg.nlayers,
                dispatches_per_token: (dispatch.forward_dispatches_per_layer * cfg.nlayers) as f32
                    / cfg.seq as f32,
                success: false,
                error: Some(format!(
                    "skipped for safety: estimated {:.1}GB exceeds 90% of physical {:.1}GB",
                    est_ram_gb, physical_gb
                )),
            };
        }
    }

    let mut result = ScaleLadderResult {
        name: name.to_string(),
        params_b,
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
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
        ffn_mode: dispatch.ffn_mode.to_string(),
        ffn_dispatches_per_layer: dispatch.ffn_dispatches_per_layer,
        forward_dispatches_per_layer: dispatch.forward_dispatches_per_layer,
        forward_dispatches_per_token: dispatch.forward_dispatches_per_layer as f32 / cfg.seq as f32,
        dispatches_per_forward: dispatch.forward_dispatches_per_layer * cfg.nlayers,
        dispatches_per_token: (dispatch.forward_dispatches_per_layer * cfg.nlayers) as f32
            / cfg.seq as f32,
        success: false,
        error: None,
    };

    let tc = TrainConfig::default();
    let (tokens, targets) = deterministic_tokens(cfg);

    let run = std::panic::catch_unwind(|| {
        let t0 = Instant::now();
        let kernels = CompiledKernels::compile(cfg);
        let compile_s = t0.elapsed().as_secs_f32();
        let rss_after_compile = rss_mb().unwrap_or(0.0);

        let t0 = Instant::now();
        let weights = ModelWeights::random(cfg);
        let mut fwd_ws = ModelForwardWorkspace::new(cfg);
        let alloc_s = t0.elapsed().as_secs_f32();
        let rss_after_alloc = rss_mb().unwrap_or(0.0);

        let warmup_loss = full_model::forward_ws(
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
            loss = full_model::forward_ws(
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
            result.p5_fwd_ms = percentile_f32(&times, 5.0);
            result.median_fwd_ms = percentile_f32(&times, 50.0);
            result.p95_fwd_ms = percentile_f32(&times, 95.0);
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

fn layer_breakdown_probe(cfg: &ModelConfig, name: &str) -> LayerBreakdownResult {
    let run = std::panic::catch_unwind(|| {
        let kernels = CompiledKernels::compile(cfg);
        let weights = LayerWeights::random(cfg);
        let x = deterministic_input(cfg.dim * cfg.seq);

        let _ = layer::forward_timed(cfg, &kernels, &weights, &x);

        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let (_, _, timings) = layer::forward_timed(cfg, &kernels, &weights, &x);
            samples.push(timings);
        }
        median_forward_timings(&samples)
    });

    match run {
        Ok(median) => {
            let ane_total_ms = median.ane_sdpa_ms + median.ane_wo_ms + median.ane_ffn_ms;
            let stage_total_ms = median.stage_sdpa_ms + median.stage_wo_ms + median.stage_ffn_ms;
            let read_total_ms = median.read_sdpa_ms + median.read_wo_ms + median.read_ffn_ms;
            let cpu_total_ms = median.rmsnorm1_ms + median.residual_rmsnorm2_ms;
            let total_ms = median.total_ms.max(1e-6);

            LayerBreakdownResult {
                name: name.to_string(),
                rmsnorm1_ms: median.rmsnorm1_ms,
                stage_sdpa_ms: median.stage_sdpa_ms,
                ane_sdpa_ms: median.ane_sdpa_ms,
                read_sdpa_ms: median.read_sdpa_ms,
                stage_wo_ms: median.stage_wo_ms,
                ane_wo_ms: median.ane_wo_ms,
                read_wo_ms: median.read_wo_ms,
                residual_rmsnorm2_ms: median.residual_rmsnorm2_ms,
                stage_ffn_ms: median.stage_ffn_ms,
                ane_ffn_ms: median.ane_ffn_ms,
                read_ffn_ms: median.read_ffn_ms,
                total_ms: median.total_ms,
                ane_total_ms,
                stage_total_ms,
                read_total_ms,
                cpu_total_ms,
                ane_pct: ane_total_ms / total_ms * 100.0,
                stage_pct: stage_total_ms / total_ms * 100.0,
                read_pct: read_total_ms / total_ms * 100.0,
                cpu_pct: cpu_total_ms / total_ms * 100.0,
                success: true,
                error: None,
            }
        }
        Err(payload) => LayerBreakdownResult {
            name: name.to_string(),
            rmsnorm1_ms: 0.0,
            stage_sdpa_ms: 0.0,
            ane_sdpa_ms: 0.0,
            read_sdpa_ms: 0.0,
            stage_wo_ms: 0.0,
            ane_wo_ms: 0.0,
            read_wo_ms: 0.0,
            residual_rmsnorm2_ms: 0.0,
            stage_ffn_ms: 0.0,
            ane_ffn_ms: 0.0,
            read_ffn_ms: 0.0,
            total_ms: 0.0,
            ane_total_ms: 0.0,
            stage_total_ms: 0.0,
            read_total_ms: 0.0,
            cpu_total_ms: 0.0,
            ane_pct: 0.0,
            stage_pct: 0.0,
            read_pct: 0.0,
            cpu_pct: 0.0,
            success: false,
            error: Some(panic_payload_to_string(payload)),
        },
    }
}

fn median_forward_timings(samples: &[ForwardTimings]) -> ForwardTimings {
    fn med<F: Fn(&ForwardTimings) -> f32>(samples: &[ForwardTimings], get: F) -> f32 {
        let mut vals: Vec<f32> = samples.iter().map(get).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals[vals.len() / 2]
    }

    ForwardTimings {
        rmsnorm1_ms: med(samples, |s| s.rmsnorm1_ms),
        stage_sdpa_ms: med(samples, |s| s.stage_sdpa_ms),
        ane_sdpa_ms: med(samples, |s| s.ane_sdpa_ms),
        read_sdpa_ms: med(samples, |s| s.read_sdpa_ms),
        stage_wo_ms: med(samples, |s| s.stage_wo_ms),
        ane_wo_ms: med(samples, |s| s.ane_wo_ms),
        read_wo_ms: med(samples, |s| s.read_wo_ms),
        residual_rmsnorm2_ms: med(samples, |s| s.residual_rmsnorm2_ms),
        stage_ffn_ms: med(samples, |s| s.stage_ffn_ms),
        ane_ffn_ms: med(samples, |s| s.ane_ffn_ms),
        read_ffn_ms: med(samples, |s| s.read_ffn_ms),
        total_ms: med(samples, |s| s.total_ms),
    }
}

fn kernel_hw_probe(cfg: &ModelConfig, scale: &str) -> Vec<KernelHwWallResult> {
    let weights = LayerWeights::random(cfg);
    let x = deterministic_input(cfg.dim * cfg.seq);
    let mut xnorm = vec![0.0f32; cfg.dim * cfg.seq];
    let mut rms_inv1 = vec![0.0f32; cfg.seq];
    rmsnorm::forward_channel_first(
        &x,
        &weights.gamma1,
        &mut xnorm,
        &mut rms_inv1,
        cfg.dim,
        cfg.seq,
    );
    let attn_out = deterministic_input(cfg.q_dim * cfg.seq);
    let x2 = deterministic_input(cfg.dim * cfg.seq);
    let mut x2norm = vec![0.0f32; cfg.dim * cfg.seq];
    let mut rms_inv2 = vec![0.0f32; cfg.seq];
    rmsnorm::forward_channel_first(
        &x2,
        &weights.gamma2,
        &mut x2norm,
        &mut rms_inv2,
        cfg.dim,
        cfg.seq,
    );

    let mut results = Vec::new();

    let sdpa_xnorm_input = TensorData::with_f32(&xnorm, sdpa_fwd::xnorm_shape(cfg));
    let sdpa_wq_input = TensorData::with_f32(&weights.wq, sdpa_fwd::wq_shape(cfg));
    let sdpa_wk_input = TensorData::with_f32(&weights.wk, sdpa_fwd::wk_shape(cfg));
    let sdpa_wv_input = TensorData::with_f32(&weights.wv, sdpa_fwd::wv_shape(cfg));
    let sdpa_attn_output = TensorData::new(sdpa_fwd::attn_out_shape(cfg));
    let sdpa_q_output = TensorData::new(sdpa_fwd::q_rope_shape(cfg));
    let sdpa_k_output = TensorData::new(sdpa_fwd::k_rope_shape(cfg));
    let sdpa_v_output = TensorData::new(sdpa_fwd::v_shape(cfg));
    results.push(try_bench_kernel(scale, "sdpa_fwd", || {
        let exe = sdpa_fwd::build(cfg)
            .compile(NSQualityOfService::UserInteractive)
            .expect("sdpa_fwd compile");
        bench_kernel_multi(
            scale,
            "sdpa_fwd",
            &exe,
            &[
                &sdpa_xnorm_input,
                &sdpa_wq_input,
                &sdpa_wk_input,
                &sdpa_wv_input,
            ],
            &[
                &sdpa_attn_output,
                &sdpa_q_output,
                &sdpa_k_output,
                &sdpa_v_output,
            ],
        )
    }));

    let wo_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    let mut wo_in = vec![0.0f32; cfg.q_dim * wo_sp];
    stage_spatial(&mut wo_in, cfg.q_dim, wo_sp, &attn_out, cfg.seq, 0);
    stage_spatial(&mut wo_in, cfg.q_dim, wo_sp, &weights.wo, cfg.dim, cfg.seq);
    let wo_input = TensorData::with_f32(
        &wo_in,
        Shape {
            batch: 1,
            channels: cfg.q_dim,
            height: 1,
            width: wo_sp,
        },
    );
    let wo_output = TensorData::new(Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.seq,
    });
    results.push(try_bench_kernel(scale, "wo_fwd", || {
        let exe = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(NSQualityOfService::UserInteractive)
            .expect("wo_fwd compile");
        bench_kernel(scale, "wo_fwd", &exe, &wo_input, &wo_output)
    }));

    let ffn_sp = ffn_fused::input_spatial_width(cfg);
    let mut ffn_in = vec![0.0f32; cfg.dim * ffn_sp];
    stage_spatial(&mut ffn_in, cfg.dim, ffn_sp, &x2norm, cfg.seq, 0);
    stage_spatial(&mut ffn_in, cfg.dim, ffn_sp, &x2, cfg.seq, cfg.seq);
    stage_spatial(
        &mut ffn_in,
        cfg.dim,
        ffn_sp,
        &weights.w1,
        cfg.hidden,
        2 * cfg.seq,
    );
    stage_spatial(
        &mut ffn_in,
        cfg.dim,
        ffn_sp,
        &weights.w3,
        cfg.hidden,
        2 * cfg.seq + cfg.hidden,
    );
    stage_spatial(
        &mut ffn_in,
        cfg.dim,
        ffn_sp,
        &weights.w2,
        cfg.hidden,
        2 * cfg.seq + 2 * cfg.hidden,
    );
    let ffn_input = TensorData::with_f32(
        &ffn_in,
        Shape {
            batch: 1,
            channels: cfg.dim,
            height: 1,
            width: ffn_sp,
        },
    );
    let ffn_output = TensorData::new(Shape {
        batch: 1,
        channels: ffn_fused::output_channels(cfg),
        height: 1,
        width: cfg.seq,
    });
    if branch_flavor() == BranchFlavor::Head && ffn_needs_decomposition(cfg) {
        let gate = deterministic_input(cfg.hidden * cfg.seq);
        let mut w2t = vec![0.0f32; cfg.hidden * cfg.dim];
        for r in 0..cfg.dim {
            for c in 0..cfg.hidden {
                w2t[c * cfg.dim + r] = weights.w2[r * cfg.hidden + c];
            }
        }

        if ffn_can_use_dual_w13(cfg) {
            let w13_sp = dual_separate_spatial_width(cfg.seq, cfg.hidden);
            let mut w13_in = vec![0.0f32; cfg.dim * w13_sp];
            stage_spatial(&mut w13_in, cfg.dim, w13_sp, &x2norm, cfg.seq, 0);
            stage_spatial(
                &mut w13_in,
                cfg.dim,
                w13_sp,
                &weights.w1,
                cfg.hidden,
                cfg.seq,
            );
            stage_spatial(
                &mut w13_in,
                cfg.dim,
                w13_sp,
                &weights.w3,
                cfg.hidden,
                cfg.seq + cfg.hidden,
            );
            let w13_input = TensorData::with_f32(
                &w13_in,
                Shape {
                    batch: 1,
                    channels: cfg.dim,
                    height: 1,
                    width: w13_sp,
                },
            );
            let w13_output = TensorData::new(Shape {
                batch: 1,
                channels: 2 * cfg.hidden,
                height: 1,
                width: cfg.seq,
            });
            results.push(try_bench_kernel(scale, "ffn_w13_fwd", || {
                let exe = build_dual_separate(cfg.dim, cfg.hidden, cfg.seq)
                    .compile(NSQualityOfService::UserInteractive)
                    .expect("ffn_w13 dual compile");
                bench_kernel(scale, "ffn_w13_fwd", &exe, &w13_input, &w13_output)
            }));
        } else {
            let w13_sp = dyn_matmul::spatial_width(cfg.seq, cfg.hidden);
            let mut w1_in = vec![0.0f32; cfg.dim * w13_sp];
            stage_spatial(&mut w1_in, cfg.dim, w13_sp, &x2norm, cfg.seq, 0);
            stage_spatial(
                &mut w1_in,
                cfg.dim,
                w13_sp,
                &weights.w1,
                cfg.hidden,
                cfg.seq,
            );
            let w1_input = TensorData::with_f32(
                &w1_in,
                Shape {
                    batch: 1,
                    channels: cfg.dim,
                    height: 1,
                    width: w13_sp,
                },
            );
            let w1_output = TensorData::new(Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: cfg.seq,
            });
            results.push(try_bench_kernel(scale, "ffn_w1_fwd", || {
                let exe = dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
                    .compile(NSQualityOfService::UserInteractive)
                    .expect("ffn_w1 compile");
                bench_kernel(scale, "ffn_w1_fwd", &exe, &w1_input, &w1_output)
            }));

            let mut w3_in = vec![0.0f32; cfg.dim * w13_sp];
            stage_spatial(&mut w3_in, cfg.dim, w13_sp, &x2norm, cfg.seq, 0);
            stage_spatial(
                &mut w3_in,
                cfg.dim,
                w13_sp,
                &weights.w3,
                cfg.hidden,
                cfg.seq,
            );
            let w3_input = TensorData::with_f32(
                &w3_in,
                Shape {
                    batch: 1,
                    channels: cfg.dim,
                    height: 1,
                    width: w13_sp,
                },
            );
            let w3_output = TensorData::new(Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: cfg.seq,
            });
            results.push(try_bench_kernel(scale, "ffn_w3_fwd", || {
                let exe = dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
                    .compile(NSQualityOfService::UserInteractive)
                    .expect("ffn_w3 compile");
                bench_kernel(scale, "ffn_w3_fwd", &exe, &w3_input, &w3_output)
            }));
        }

        let w2_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
        let mut w2_in = vec![0.0f32; cfg.hidden * w2_sp];
        stage_spatial(&mut w2_in, cfg.hidden, w2_sp, &gate, cfg.seq, 0);
        stage_spatial(&mut w2_in, cfg.hidden, w2_sp, &w2t, cfg.dim, cfg.seq);
        let w2_input = TensorData::with_f32(
            &w2_in,
            Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: w2_sp,
            },
        );
        let w2_output = TensorData::new(Shape {
            batch: 1,
            channels: cfg.dim,
            height: 1,
            width: cfg.seq,
        });
        results.push(try_bench_kernel(scale, "ffn_w2_fwd", || {
            let exe = dyn_matmul::build(cfg.hidden, cfg.dim, cfg.seq)
                .compile(NSQualityOfService::UserInteractive)
                .expect("ffn_w2 compile");
            bench_kernel(scale, "ffn_w2_fwd", &exe, &w2_input, &w2_output)
        }));
    } else {
        results.push(try_bench_kernel(scale, "ffn_fused", || {
            let exe = ffn_fused::build(cfg)
                .compile(NSQualityOfService::UserInteractive)
                .expect("ffn_fused compile");
            bench_kernel(scale, "ffn_fused", &exe, &ffn_input, &ffn_output)
        }));
    }

    results
}

fn try_bench_kernel<F>(scale: &str, kernel: &str, f: F) -> KernelHwWallResult
where
    F: FnOnce() -> KernelHwWallResult,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => KernelHwWallResult {
            scale: scale.to_string(),
            kernel: kernel.to_string(),
            wall_median_us: 0.0,
            wall_p5_us: 0.0,
            wall_p95_us: 0.0,
            hw_median_ns: 0.0,
            hw_p5_ns: 0.0,
            hw_p95_ns: 0.0,
            overhead_us: 0.0,
            overhead_pct: 0.0,
            success: false,
            error: Some(panic_payload_to_string(payload)),
        },
    }
}

fn bench_kernel(
    scale: &str,
    kernel: &str,
    exe: &ane_bridge::ane::Executable,
    input: &TensorData,
    output: &TensorData,
) -> KernelHwWallResult {
    bench_kernel_multi(scale, kernel, exe, &[input], &[output])
}

fn bench_kernel_multi(
    scale: &str,
    kernel: &str,
    exe: &ane_bridge::ane::Executable,
    inputs: &[&TensorData],
    outputs: &[&TensorData],
) -> KernelHwWallResult {
    for _ in 0..5 {
        exe.run_cached(inputs, outputs)
            .expect("kernel warmup failed");
    }

    let mut wall_us = Vec::with_capacity(50);
    for _ in 0..50 {
        let t0 = Instant::now();
        exe.run_cached(inputs, outputs)
            .expect("kernel wall eval failed");
        wall_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let mut hw_ns = Vec::with_capacity(50);
    for _ in 0..50 {
        let hw = exe
            .run_cached_with_stats(inputs, outputs)
            .expect("kernel perf eval failed");
        hw_ns.push(hw as f64);
    }

    wall_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hw_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let wall_median_us = percentile_f64(&wall_us, 50.0);
    let hw_median_ns = percentile_f64(&hw_ns, 50.0);
    let overhead_us = wall_median_us - hw_median_ns / 1000.0;
    let overhead_pct = if wall_median_us > 0.0 {
        overhead_us / wall_median_us * 100.0
    } else {
        0.0
    };

    KernelHwWallResult {
        scale: scale.to_string(),
        kernel: kernel.to_string(),
        wall_median_us,
        wall_p5_us: percentile_f64(&wall_us, 5.0),
        wall_p95_us: percentile_f64(&wall_us, 95.0),
        hw_median_ns,
        hw_p5_ns: percentile_f64(&hw_ns, 5.0),
        hw_p95_ns: percentile_f64(&hw_ns, 95.0),
        overhead_us,
        overhead_pct,
        success: true,
        error: None,
    }
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

fn run_scale_suite(
    configs: Vec<(ModelConfig, &'static str)>,
    mode: ResultsMode,
    _forward_mode: ForwardMode,
) {
    let _guard = run_lock().lock().unwrap();
    let physical_gb = physical_mem_gb();
    with_results_root(results_root_for(mode), || {
        let mut results = Vec::new();
        for (cfg, name) in configs {
            let result = scale_probe(&cfg, name, physical_gb);
            let stop = !result.success;
            results.push(result);
            write_json(&scale_json_path(), &results);
            write_summary();
            if stop {
                break;
            }
        }
        assert!(!results.is_empty(), "no scale ladder results recorded");
    });
}

fn run_layer_suite(configs: Vec<(ModelConfig, &'static str)>, mode: ResultsMode) {
    let _guard = run_lock().lock().unwrap();
    with_results_root(results_root_for(mode), || {
        let mut results = Vec::new();
        for (cfg, name) in configs {
            results.push(layer_breakdown_probe(&cfg, name));
        }
        write_json(&layer_json_path(), &results);
        write_summary();
        assert!(!results.is_empty(), "expected layer breakdown results");
    });
}

fn run_kernel_suite(configs: Vec<(ModelConfig, &'static str)>, mode: ResultsMode) {
    let _guard = run_lock().lock().unwrap();
    with_results_root(results_root_for(mode), || {
        let mut results = Vec::new();
        for (cfg, scale) in configs {
            results.extend(kernel_hw_probe(&cfg, scale));
        }
        write_json(&kernel_json_path(), &results);
        write_summary();
        assert!(!results.is_empty(), "expected kernel timing results");
    });
}

#[test]
#[ignore]
fn forward_scale_single_stream() {
    run_scale_suite(scale_ladder(), ResultsMode::Default, ForwardMode::Common);
}

#[test]
#[ignore]
fn forward_scale_layer_breakdown() {
    run_layer_suite(representative_scales(), ResultsMode::Default);
}

#[test]
#[ignore]
fn forward_scale_kernel_hw_vs_wall() {
    run_kernel_suite(representative_scales(), ResultsMode::Default);
}

#[test]
#[ignore]
fn pr22_compare_single_stream() {
    run_scale_suite(compare_scales(), ResultsMode::Compare, ForwardMode::Common);
}

#[test]
#[ignore]
fn pr22_compare_layer_breakdown() {
    run_layer_suite(compare_scales(), ResultsMode::Compare);
}

#[test]
#[ignore]
fn pr22_compare_kernel_overhead() {
    run_kernel_suite(compare_scales(), ResultsMode::Compare);
}
