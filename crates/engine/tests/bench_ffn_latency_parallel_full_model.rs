//! Benchmark-only full-model FFN latency parallel experiment.
//!
//! Run build check:
//!   cargo test -p engine --test bench_ffn_latency_parallel_full_model --no-run
//!
//! Run smoke validation:
//!   cargo test -p engine --test bench_ffn_latency_parallel_full_model --release -- --ignored --nocapture ffn_shard_full_model_smoke_30b_matches_baseline
//!
//! Run 30B benchmark:
//!   cargo test -p engine --test bench_ffn_latency_parallel_full_model --release -- --ignored --nocapture bench_ffn_shard_full_model_30b
//!
//! Run 50B equal-shard benchmark:
//!   cargo test -p engine --test bench_ffn_latency_parallel_full_model --release -- --ignored --nocapture bench_ffn_shard_full_model_50b_equal
//!
//! Run 80B equal-shard benchmark:
//!   cargo test -p engine --test bench_ffn_latency_parallel_full_model --release -- --ignored --nocapture bench_ffn_shard_full_model_80b_equal

use ane_bridge::ane::{Executable, TensorData};
use engine::cpu::{cross_entropy, embedding, rmsnorm, silu, vdsp};
use engine::full_model::{ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::kernels::{dyn_matmul, ffn_fused, sdpa_fwd};
use engine::layer::{self, CompiledKernels, LayerWeights};
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;
use serde::Serialize;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::Instant;

const TIMED_RUNS: usize = 2;
const ANE_STATS_RUNS: usize = 1;
const MAX_ABS_DIFF_TOL: f32 = 2e-2;
const MEAN_ABS_DIFF_TOL: f32 = 5e-3;
const COSINE_SIM_TOL: f32 = 0.999;
const LOSS_ABS_DIFF_TOL: f32 = 1e-4;
const PERF_WIN_PCT_TOL: f32 = 5.0;

fn run_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/latency_parallel_ffn")
}

fn json_path(scale_name: &str) -> PathBuf {
    results_dir().join(format!(
        "full_model_{}_compare.json",
        scale_name.to_lowercase()
    ))
}

fn summary_path(scale_name: &str) -> PathBuf {
    results_dir().join(format!(
        "full_model_{}_summary.md",
        scale_name.to_lowercase()
    ))
}

fn ensure_results_dir() {
    fs::create_dir_all(results_dir()).expect("create results dir");
}

fn write_json<T: Serialize>(path: &Path, value: &T) {
    ensure_results_dir();
    let json = serde_json::to_string_pretty(value).expect("serialize json");
    fs::write(path, json).expect("write json");
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

fn cfg_30b() -> ModelConfig {
    custom_config(5120, 13824, 40, 96, 512)
}

fn cfg_50b() -> ModelConfig {
    custom_config(5120, 13824, 40, 160, 512)
}

fn cfg_80b() -> ModelConfig {
    custom_config(5120, 13824, 40, 256, 512)
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

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
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

fn read_channels_into(
    src: &[f32],
    total_ch: usize,
    seq: usize,
    ch_start: usize,
    ch_count: usize,
    dst: &mut [f32],
) {
    let start = ch_start * seq;
    let end = (ch_start + ch_count).min(total_ch) * seq;
    assert_eq!(
        dst.len(),
        end - start,
        "destination length mismatch in read_channels_into"
    );
    dst.copy_from_slice(&src[start..end]);
}

fn stage_weight_columns(
    dst: &mut [f32],
    rows: usize,
    sp_width: usize,
    src: &[f32],
    total_cols: usize,
    col_start: usize,
    col_count: usize,
    sp_offset: usize,
) {
    for r in 0..rows {
        let src_row = r * total_cols + col_start;
        let dst_row = r * sp_width + sp_offset;
        dst[dst_row..dst_row + col_count].copy_from_slice(&src[src_row..src_row + col_count]);
    }
}

fn stage_transposed_weight_columns(
    dst: &mut [f32],
    src: &[f32],
    rows: usize,
    total_cols: usize,
    col_start: usize,
    col_count: usize,
    sp_width: usize,
    sp_offset: usize,
) {
    for c in 0..col_count {
        let dst_row = c * sp_width + sp_offset;
        for r in 0..rows {
            dst[dst_row + r] = src[r * total_cols + col_start + c];
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct DiffMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    cosine_similarity: f32,
    loss_abs_diff: f32,
}

impl DiffMetrics {
    fn passes(&self) -> bool {
        self.max_abs_diff <= MAX_ABS_DIFF_TOL
            && self.mean_abs_diff <= MEAN_ABS_DIFF_TOL
            && self.cosine_similarity >= COSINE_SIM_TOL
            && self.loss_abs_diff <= LOSS_ABS_DIFF_TOL
    }
}

fn compare_logits_and_loss(
    actual_logits: &[f32],
    actual_loss: f32,
    expected_logits: &[f32],
    expected_loss: f32,
) -> DiffMetrics {
    assert_eq!(
        actual_logits.len(),
        expected_logits.len(),
        "logits length mismatch"
    );

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&a, &b) in actual_logits.iter().zip(expected_logits.iter()) {
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        dot += a as f64 * b as f64;
        norm_a += a as f64 * a as f64;
        norm_b += b as f64 * b as f64;
    }

    let cosine_similarity = if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    } else {
        1.0
    };

    DiffMetrics {
        max_abs_diff: max_abs,
        mean_abs_diff: sum_abs / actual_logits.len() as f32,
        cosine_similarity,
        loss_abs_diff: (actual_loss - expected_loss).abs(),
    }
}

#[derive(Debug, Clone, Serialize)]
struct Thresholds {
    max_abs_diff_tol: f32,
    mean_abs_diff_tol: f32,
    cosine_similarity_tol: f32,
    loss_abs_diff_tol: f32,
    perf_win_pct_tol: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ModeResult {
    mode: String,
    shard_count: usize,
    compile_s: f32,
    warmup_wall_ms: f32,
    timed_samples_ms: Vec<f32>,
    mean_full_forward_ms: f32,
    tok_per_s: f32,
    peak_rss_mb: f32,
    ane_hw_ms: f32,
    ane_busy_pct: f32,
    avg_layer_ms: f32,
    total_ffn_ms: f32,
    total_non_ffn_ms: f32,
    ffn_pct_of_wall: f32,
    diff_vs_baseline: DiffMetrics,
    correctness_pass: bool,
    full_forward_win_pct_vs_baseline: f32,
    meets_perf_gate: bool,
    bucket_note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ExperimentResult {
    config_name: String,
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
    baseline: ModeResult,
    shard2: ModeResult,
    shard4: ModeResult,
    shard6: ModeResult,
    shard8: ModeResult,
    shard12: ModeResult,
    shard16: ModeResult,
    primary_success: bool,
    winning_modes: Vec<String>,
    thresholds: Thresholds,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedExperimentResult {
    config_name: String,
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
    baseline: ModeResult,
    modes: Vec<ModeResult>,
    primary_success: bool,
    winning_modes: Vec<String>,
    thresholds: Thresholds,
}

#[derive(Debug, Clone, Copy)]
struct AneSample {
    hw_ms: f32,
}

#[derive(Debug, Clone, Copy)]
struct ExecOptions {
    collect_stats: bool,
}

impl ExecOptions {
    const DIRECT: Self = Self {
        collect_stats: false,
    };
    const STATS: Self = Self {
        collect_stats: true,
    };
}

impl Default for AneSample {
    fn default() -> Self {
        Self { hw_ms: 0.0 }
    }
}

struct FullModelOutputs {
    logits: Vec<f32>,
    loss: f32,
    total_ffn_ms: f32,
}

struct SdpaRunner {
    exe: Executable,
    xnorm_in: TensorData,
    wq_in: TensorData,
    wk_in: TensorData,
    wv_in: TensorData,
    attn_out: TensorData,
    q_rope_out: TensorData,
    k_rope_out: TensorData,
    v_out: TensorData,
}

struct WoRunner {
    exe: Executable,
    input: TensorData,
    output: TensorData,
}

#[derive(Clone, Copy)]
struct ShardLayout {
    col_start: usize,
    col_count: usize,
}

struct ShardWorker {
    w13_exe: Executable,
    w2_exe: Executable,
    w13_in: TensorData,
    w13_out: TensorData,
    w2_in: TensorData,
    w2_out: TensorData,
    h1: Vec<f32>,
    h3: Vec<f32>,
    gate: Vec<f32>,
}

struct LayerScratch {
    xnorm: Vec<f32>,
    rms_inv1: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    x2: Vec<f32>,
    x2norm: Vec<f32>,
    rms_inv2: Vec<f32>,
    ffn_out: Vec<f32>,
    merge_tmp: Vec<f32>,
}

impl LayerScratch {
    fn new(cfg: &ModelConfig) -> Self {
        Self {
            xnorm: vec![0.0; cfg.dim * cfg.seq],
            rms_inv1: vec![0.0; cfg.seq],
            attn_out: vec![0.0; cfg.q_dim * cfg.seq],
            o_out: vec![0.0; cfg.dim * cfg.seq],
            x2: vec![0.0; cfg.dim * cfg.seq],
            x2norm: vec![0.0; cfg.dim * cfg.seq],
            rms_inv2: vec![0.0; cfg.seq],
            ffn_out: vec![0.0; cfg.dim * cfg.seq],
            merge_tmp: vec![0.0; cfg.dim * cfg.seq],
        }
    }
}

struct ShardedFfnRunner {
    cfg: ModelConfig,
    shard_count: usize,
    sdpa: SdpaRunner,
    wo: WoRunner,
    workers: Vec<ShardWorker>,
    layer_scratch: LayerScratch,
    shard_layouts: Vec<ShardLayout>,
}

struct BaselineRunner {
    cfg: ModelConfig,
    use_dual_w13: bool,
    sdpa: SdpaRunner,
    wo: WoRunner,
    w13_exe: Executable,
    w13_in: TensorData,
    w13_out: TensorData,
    w2_exe: Executable,
    w2_in: TensorData,
    w2_out: TensorData,
    h1: Vec<f32>,
    h3: Vec<f32>,
    gate: Vec<f32>,
    w2t: Vec<f32>,
    layer_scratch: LayerScratch,
}

fn compile_sdpa_runner(cfg: &ModelConfig) -> SdpaRunner {
    let qos = NSQualityOfService::UserInteractive;
    let exe = sdpa_fwd::build(cfg).compile(qos).expect("sdpa_fwd compile");
    SdpaRunner {
        exe,
        xnorm_in: TensorData::new(sdpa_fwd::xnorm_shape(cfg)),
        wq_in: TensorData::new(sdpa_fwd::wq_shape(cfg)),
        wk_in: TensorData::new(sdpa_fwd::wk_shape(cfg)),
        wv_in: TensorData::new(sdpa_fwd::wv_shape(cfg)),
        attn_out: TensorData::new(sdpa_fwd::attn_out_shape(cfg)),
        q_rope_out: TensorData::new(sdpa_fwd::q_rope_shape(cfg)),
        k_rope_out: TensorData::new(sdpa_fwd::k_rope_shape(cfg)),
        v_out: TensorData::new(sdpa_fwd::v_shape(cfg)),
    }
}

fn compile_wo_runner(cfg: &ModelConfig) -> WoRunner {
    let qos = NSQualityOfService::UserInteractive;
    let exe = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
        .compile(qos)
        .expect("wo_fwd compile");
    let sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    WoRunner {
        exe,
        input: TensorData::new(ane_bridge::ane::Shape {
            batch: 1,
            channels: cfg.q_dim,
            height: 1,
            width: sp,
        }),
        output: TensorData::new(ane_bridge::ane::Shape {
            batch: 1,
            channels: cfg.dim,
            height: 1,
            width: cfg.seq,
        }),
    }
}

fn shard_layouts(cfg: &ModelConfig, shard_count: usize) -> Vec<ShardLayout> {
    let shard_hidden = cfg.hidden / shard_count;
    (0..shard_count)
        .map(|i| ShardLayout {
            col_start: i * shard_hidden,
            col_count: shard_hidden,
        })
        .collect()
}

fn compile_sharded_runner(cfg: &ModelConfig, shard_count: usize) -> (ShardedFfnRunner, f32) {
    assert!(
        cfg.hidden % shard_count == 0,
        "hidden must be divisible by shard count"
    );
    let shard_hidden = cfg.hidden / shard_count;
    let qos = NSQualityOfService::UserInteractive;
    let t0 = Instant::now();
    let sdpa = compile_sdpa_runner(cfg);
    let wo = compile_wo_runner(cfg);
    let mut workers = Vec::with_capacity(shard_count);
    for _ in 0..shard_count {
        let w13_exe = dyn_matmul::build_dual_separate(cfg.dim, shard_hidden, cfg.seq)
            .compile(qos)
            .expect("w13 shard compile");
        let w2_exe = dyn_matmul::build(shard_hidden, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("w2 shard compile");
        let w13_sp = dyn_matmul::dual_separate_spatial_width(cfg.seq, shard_hidden);
        let w2_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
        workers.push(ShardWorker {
            w13_exe,
            w2_exe,
            w13_in: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: w13_sp,
            }),
            w13_out: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: 2 * shard_hidden,
                height: 1,
                width: cfg.seq,
            }),
            w2_in: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: shard_hidden,
                height: 1,
                width: w2_sp,
            }),
            w2_out: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: cfg.seq,
            }),
            h1: vec![0.0; shard_hidden * cfg.seq],
            h3: vec![0.0; shard_hidden * cfg.seq],
            gate: vec![0.0; shard_hidden * cfg.seq],
        });
    }
    let compile_s = t0.elapsed().as_secs_f32();
    (
        ShardedFfnRunner {
            cfg: cfg.clone(),
            shard_count,
            sdpa,
            wo,
            workers,
            layer_scratch: LayerScratch::new(cfg),
            shard_layouts: shard_layouts(cfg, shard_count),
        },
        compile_s,
    )
}

fn compile_baseline_runner(cfg: &ModelConfig) -> (BaselineRunner, f32) {
    let qos = NSQualityOfService::UserInteractive;
    let use_dual_w13 = ffn_fused::can_use_dual_w13(cfg);
    let t0 = Instant::now();
    let sdpa = compile_sdpa_runner(cfg);
    let wo = compile_wo_runner(cfg);
    let w13_exe = if use_dual_w13 {
        dyn_matmul::build_dual_separate(cfg.dim, cfg.hidden, cfg.seq)
            .compile(qos)
            .expect("baseline w13 dual compile")
    } else {
        dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
            .compile(qos)
            .expect("baseline w13 compile")
    };
    let w13_sp = if use_dual_w13 {
        dyn_matmul::dual_separate_spatial_width(cfg.seq, cfg.hidden)
    } else {
        dyn_matmul::spatial_width(cfg.seq, cfg.hidden)
    };
    let w13_out_channels = if use_dual_w13 {
        2 * cfg.hidden
    } else {
        cfg.hidden
    };
    let w2_exe = dyn_matmul::build(cfg.hidden, cfg.dim, cfg.seq)
        .compile(qos)
        .expect("baseline w2 compile");
    let w2_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    let compile_s = t0.elapsed().as_secs_f32();

    (
        BaselineRunner {
            cfg: cfg.clone(),
            use_dual_w13,
            sdpa,
            wo,
            w13_exe,
            w13_in: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: w13_sp,
            }),
            w13_out: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: w13_out_channels,
                height: 1,
                width: cfg.seq,
            }),
            w2_exe,
            w2_in: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: w2_sp,
            }),
            w2_out: TensorData::new(ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: cfg.seq,
            }),
            h1: vec![0.0; cfg.hidden * cfg.seq],
            h3: vec![0.0; cfg.hidden * cfg.seq],
            gate: vec![0.0; cfg.hidden * cfg.seq],
            w2t: vec![0.0; cfg.hidden * cfg.dim],
            layer_scratch: LayerScratch::new(cfg),
        },
        compile_s,
    )
}

fn run_sdpa(
    sdpa: &mut SdpaRunner,
    xnorm: &[f32],
    weights: &LayerWeights,
    attn_out: &mut [f32],
    options: ExecOptions,
) -> u64 {
    sdpa.xnorm_in.copy_from_f32(xnorm);
    sdpa.wq_in.copy_from_f32(&weights.wq);
    sdpa.wk_in.copy_from_f32(&weights.wk);
    sdpa.wv_in.copy_from_f32(&weights.wv);
    let hw_ns = if options.collect_stats {
        sdpa.exe
            .run_cached_with_stats(
                &[&sdpa.xnorm_in, &sdpa.wq_in, &sdpa.wk_in, &sdpa.wv_in],
                &[
                    &sdpa.attn_out,
                    &sdpa.q_rope_out,
                    &sdpa.k_rope_out,
                    &sdpa.v_out,
                ],
            )
            .expect("sdpa run")
    } else {
        sdpa.exe
            .run_cached_direct(
                &[&sdpa.xnorm_in, &sdpa.wq_in, &sdpa.wk_in, &sdpa.wv_in],
                &[
                    &sdpa.attn_out,
                    &sdpa.q_rope_out,
                    &sdpa.k_rope_out,
                    &sdpa.v_out,
                ],
            )
            .expect("sdpa run");
        0
    };
    let locked = sdpa.attn_out.as_f32_slice();
    attn_out.copy_from_slice(&locked[..attn_out.len()]);
    hw_ns
}

fn run_wo(
    wo: &mut WoRunner,
    cfg: &ModelConfig,
    attn_out: &[f32],
    weights: &LayerWeights,
    o_out: &mut [f32],
    options: ExecOptions,
) -> u64 {
    let sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    {
        let mut locked = wo.input.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, cfg.q_dim, sp, attn_out, cfg.seq, 0);
        stage_spatial(buf, cfg.q_dim, sp, &weights.wo, cfg.dim, cfg.seq);
    }
    let hw_ns = if options.collect_stats {
        wo.exe
            .run_cached_with_stats(&[&wo.input], &[&wo.output])
            .expect("wo run")
    } else {
        wo.exe
            .run_cached_direct(&[&wo.input], &[&wo.output])
            .expect("wo run");
        0
    };
    let locked = wo.output.as_f32_slice();
    o_out.copy_from_slice(&locked[..o_out.len()]);
    hw_ns
}

#[allow(dead_code)]
fn run_sharded_layer_forward_into(
    runner: &mut ShardedFfnRunner,
    layer_weights: &LayerWeights,
    shard_layouts: &[ShardLayout],
    x: &[f32],
    x_next: &mut [f32],
) -> (f32, u64) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
    let scratch = &mut runner.layer_scratch;

    rmsnorm::forward_channel_first(
        x,
        &layer_weights.gamma1,
        &mut scratch.xnorm,
        &mut scratch.rms_inv1,
        dim,
        seq,
    );
    let mut total_hw_ns = 0u64;
    total_hw_ns += run_sdpa(
        &mut runner.sdpa,
        &scratch.xnorm,
        layer_weights,
        &mut scratch.attn_out,
        ExecOptions::DIRECT,
    );
    total_hw_ns += run_wo(
        &mut runner.wo,
        cfg,
        &scratch.attn_out,
        layer_weights,
        &mut scratch.o_out,
        ExecOptions::DIRECT,
    );

    vdsp::vsma(&scratch.o_out, alpha, x, &mut scratch.x2);
    rmsnorm::forward_channel_first(
        &scratch.x2,
        &layer_weights.gamma2,
        &mut scratch.x2norm,
        &mut scratch.rms_inv2,
        dim,
        seq,
    );

    let ffn_t0 = Instant::now();
    let barrier = Arc::new(Barrier::new(runner.shard_count + 1));
    let x2norm_ref: &[f32] = &scratch.x2norm;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(runner.shard_count);
        for (worker, shard) in runner.workers.iter_mut().zip(shard_layouts.iter()) {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                let shard_hidden = shard.col_count;
                let w13_sp = dyn_matmul::dual_separate_spatial_width(seq, shard_hidden);
                let w2_sp = dyn_matmul::spatial_width(seq, dim);

                barrier.wait();

                {
                    let mut locked = worker.w13_in.as_f32_slice_mut();
                    let buf = &mut *locked;
                    stage_spatial(buf, dim, w13_sp, x2norm_ref, seq, 0);
                    stage_weight_columns(
                        buf,
                        dim,
                        w13_sp,
                        &layer_weights.w1,
                        cfg.hidden,
                        shard.col_start,
                        shard_hidden,
                        seq,
                    );
                    stage_weight_columns(
                        buf,
                        dim,
                        w13_sp,
                        &layer_weights.w3,
                        cfg.hidden,
                        shard.col_start,
                        shard_hidden,
                        seq + shard_hidden,
                    );
                }
                worker
                    .w13_exe
                    .run_cached_direct(&[&worker.w13_in], &[&worker.w13_out])
                    .expect("w13 shard run");
                {
                    let locked = worker.w13_out.as_f32_slice();
                    read_channels_into(
                        &locked,
                        2 * shard_hidden,
                        seq,
                        0,
                        shard_hidden,
                        &mut worker.h1,
                    );
                    read_channels_into(
                        &locked,
                        2 * shard_hidden,
                        seq,
                        shard_hidden,
                        shard_hidden,
                        &mut worker.h3,
                    );
                }

                silu::silu_gate(&worker.h1, &worker.h3, &mut worker.gate);

                {
                    let mut locked = worker.w2_in.as_f32_slice_mut();
                    let buf = &mut *locked;
                    stage_spatial(buf, shard_hidden, w2_sp, &worker.gate, seq, 0);
                    stage_transposed_weight_columns(
                        buf,
                        &layer_weights.w2,
                        dim,
                        cfg.hidden,
                        shard.col_start,
                        shard_hidden,
                        w2_sp,
                        seq,
                    );
                }
                worker
                    .w2_exe
                    .run_cached_direct(&[&worker.w2_in], &[&worker.w2_out])
                    .expect("w2 shard run");
            }));
        }

        barrier.wait();
        for handle in handles {
            handle.join().expect("shard worker panicked");
        }
    });

    scratch.ffn_out.fill(0.0);
    for worker in &runner.workers {
        let locked = worker.w2_out.as_f32_slice();
        vdsp::vadd(
            &scratch.ffn_out,
            &locked[..scratch.ffn_out.len()],
            &mut scratch.merge_tmp,
        );
        scratch.ffn_out.copy_from_slice(&scratch.merge_tmp);
    }
    vdsp::vsma(&scratch.ffn_out, alpha, &scratch.x2, x_next);
    (ffn_t0.elapsed().as_secs_f32() * 1000.0, total_hw_ns)
}

fn run_sharded_layer_forward_into_stats(
    runner: &mut ShardedFfnRunner,
    layer_weights: &LayerWeights,
    shard_layouts: &[ShardLayout],
    x: &[f32],
    x_next: &mut [f32],
) -> (f32, u64) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
    let scratch = &mut runner.layer_scratch;

    rmsnorm::forward_channel_first(
        x,
        &layer_weights.gamma1,
        &mut scratch.xnorm,
        &mut scratch.rms_inv1,
        dim,
        seq,
    );
    let mut total_hw_ns = 0u64;
    total_hw_ns += run_sdpa(
        &mut runner.sdpa,
        &scratch.xnorm,
        layer_weights,
        &mut scratch.attn_out,
        ExecOptions::STATS,
    );
    total_hw_ns += run_wo(
        &mut runner.wo,
        cfg,
        &scratch.attn_out,
        layer_weights,
        &mut scratch.o_out,
        ExecOptions::STATS,
    );

    vdsp::vsma(&scratch.o_out, alpha, x, &mut scratch.x2);
    rmsnorm::forward_channel_first(
        &scratch.x2,
        &layer_weights.gamma2,
        &mut scratch.x2norm,
        &mut scratch.rms_inv2,
        dim,
        seq,
    );

    let ffn_t0 = Instant::now();
    let barrier = Arc::new(Barrier::new(runner.shard_count + 1));
    let x2norm_ref: &[f32] = &scratch.x2norm;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(runner.shard_count);
        for (worker, shard) in runner.workers.iter_mut().zip(shard_layouts.iter()) {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || -> u64 {
                let shard_hidden = shard.col_count;
                let w13_sp = dyn_matmul::dual_separate_spatial_width(seq, shard_hidden);
                let w2_sp = dyn_matmul::spatial_width(seq, dim);

                barrier.wait();

                {
                    let mut locked = worker.w13_in.as_f32_slice_mut();
                    let buf = &mut *locked;
                    stage_spatial(buf, dim, w13_sp, x2norm_ref, seq, 0);
                    stage_weight_columns(
                        buf,
                        dim,
                        w13_sp,
                        &layer_weights.w1,
                        cfg.hidden,
                        shard.col_start,
                        shard_hidden,
                        seq,
                    );
                    stage_weight_columns(
                        buf,
                        dim,
                        w13_sp,
                        &layer_weights.w3,
                        cfg.hidden,
                        shard.col_start,
                        shard_hidden,
                        seq + shard_hidden,
                    );
                }
                let mut shard_hw_ns = worker
                    .w13_exe
                    .run_cached_with_stats(&[&worker.w13_in], &[&worker.w13_out])
                    .expect("w13 shard stats run");
                {
                    let locked = worker.w13_out.as_f32_slice();
                    read_channels_into(
                        &locked,
                        2 * shard_hidden,
                        seq,
                        0,
                        shard_hidden,
                        &mut worker.h1,
                    );
                    read_channels_into(
                        &locked,
                        2 * shard_hidden,
                        seq,
                        shard_hidden,
                        shard_hidden,
                        &mut worker.h3,
                    );
                }

                silu::silu_gate(&worker.h1, &worker.h3, &mut worker.gate);

                {
                    let mut locked = worker.w2_in.as_f32_slice_mut();
                    let buf = &mut *locked;
                    stage_spatial(buf, shard_hidden, w2_sp, &worker.gate, seq, 0);
                    stage_transposed_weight_columns(
                        buf,
                        &layer_weights.w2,
                        dim,
                        cfg.hidden,
                        shard.col_start,
                        shard_hidden,
                        w2_sp,
                        seq,
                    );
                }
                shard_hw_ns += worker
                    .w2_exe
                    .run_cached_with_stats(&[&worker.w2_in], &[&worker.w2_out])
                    .expect("w2 shard stats run");
                shard_hw_ns
            }));
        }

        barrier.wait();
        for handle in handles {
            total_hw_ns += handle.join().expect("stats shard worker panicked");
        }
    });

    scratch.ffn_out.fill(0.0);
    for worker in &runner.workers {
        let locked = worker.w2_out.as_f32_slice();
        vdsp::vadd(
            &scratch.ffn_out,
            &locked[..scratch.ffn_out.len()],
            &mut scratch.merge_tmp,
        );
        scratch.ffn_out.copy_from_slice(&scratch.merge_tmp);
    }
    vdsp::vsma(&scratch.ffn_out, alpha, &scratch.x2, x_next);
    (ffn_t0.elapsed().as_secs_f32() * 1000.0, total_hw_ns)
}

fn run_baseline_layer_forward_into(
    runner: &mut BaselineRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    x_next: &mut [f32],
    options: ExecOptions,
) -> (f32, u64) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
    let scratch = &mut runner.layer_scratch;

    rmsnorm::forward_channel_first(
        x,
        &layer_weights.gamma1,
        &mut scratch.xnorm,
        &mut scratch.rms_inv1,
        dim,
        seq,
    );
    let mut total_hw_ns = 0u64;
    total_hw_ns += run_sdpa(
        &mut runner.sdpa,
        &scratch.xnorm,
        layer_weights,
        &mut scratch.attn_out,
        options,
    );
    total_hw_ns += run_wo(
        &mut runner.wo,
        cfg,
        &scratch.attn_out,
        layer_weights,
        &mut scratch.o_out,
        options,
    );

    vdsp::vsma(&scratch.o_out, alpha, x, &mut scratch.x2);
    rmsnorm::forward_channel_first(
        &scratch.x2,
        &layer_weights.gamma2,
        &mut scratch.x2norm,
        &mut scratch.rms_inv2,
        dim,
        seq,
    );

    let ffn_t0 = Instant::now();
    let w13_sp = if runner.use_dual_w13 {
        dyn_matmul::dual_separate_spatial_width(seq, cfg.hidden)
    } else {
        dyn_matmul::spatial_width(seq, cfg.hidden)
    };
    let w2_sp = dyn_matmul::spatial_width(seq, dim);

    {
        let mut locked = runner.w13_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w13_sp, &scratch.x2norm, seq, 0);
        stage_spatial(buf, dim, w13_sp, &layer_weights.w1, cfg.hidden, seq);
        if runner.use_dual_w13 {
            stage_spatial(
                buf,
                dim,
                w13_sp,
                &layer_weights.w3,
                cfg.hidden,
                seq + cfg.hidden,
            );
        }
    }

    if runner.use_dual_w13 {
        total_hw_ns += if options.collect_stats {
            runner
                .w13_exe
                .run_cached_with_stats(&[&runner.w13_in], &[&runner.w13_out])
                .expect("baseline w13 dual stats run")
        } else {
            runner
                .w13_exe
                .run_cached_direct(&[&runner.w13_in], &[&runner.w13_out])
                .expect("baseline w13 dual run");
            0
        };
        {
            let locked = runner.w13_out.as_f32_slice();
            read_channels_into(&locked, 2 * cfg.hidden, seq, 0, cfg.hidden, &mut runner.h1);
            read_channels_into(
                &locked,
                2 * cfg.hidden,
                seq,
                cfg.hidden,
                cfg.hidden,
                &mut runner.h3,
            );
        }
    } else {
        total_hw_ns += if options.collect_stats {
            runner
                .w13_exe
                .run_cached_with_stats(&[&runner.w13_in], &[&runner.w13_out])
                .expect("baseline w1 stats run")
        } else {
            runner
                .w13_exe
                .run_cached_direct(&[&runner.w13_in], &[&runner.w13_out])
                .expect("baseline w1 run");
            0
        };
        {
            let locked = runner.w13_out.as_f32_slice();
            runner.h1.copy_from_slice(&locked[..cfg.hidden * seq]);
        }
        {
            let mut locked = runner.w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &scratch.x2norm, seq, 0);
            stage_spatial(buf, dim, w13_sp, &layer_weights.w3, cfg.hidden, seq);
        }
        total_hw_ns += if options.collect_stats {
            runner
                .w13_exe
                .run_cached_with_stats(&[&runner.w13_in], &[&runner.w13_out])
                .expect("baseline w3 stats run")
        } else {
            runner
                .w13_exe
                .run_cached_direct(&[&runner.w13_in], &[&runner.w13_out])
                .expect("baseline w3 run");
            0
        };
        {
            let locked = runner.w13_out.as_f32_slice();
            runner.h3.copy_from_slice(&locked[..cfg.hidden * seq]);
        }
    }

    silu::silu_gate(&runner.h1, &runner.h3, &mut runner.gate);
    vdsp::mtrans(
        &layer_weights.w2,
        cfg.hidden,
        &mut runner.w2t,
        dim,
        dim,
        cfg.hidden,
    );
    {
        let mut locked = runner.w2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, cfg.hidden, w2_sp, &runner.gate, seq, 0);
        stage_spatial(buf, cfg.hidden, w2_sp, &runner.w2t, dim, seq);
    }
    total_hw_ns += if options.collect_stats {
        runner
            .w2_exe
            .run_cached_with_stats(&[&runner.w2_in], &[&runner.w2_out])
            .expect("baseline w2 stats run")
    } else {
        runner
            .w2_exe
            .run_cached_direct(&[&runner.w2_in], &[&runner.w2_out])
            .expect("baseline w2 run");
        0
    };

    {
        let locked = runner.w2_out.as_f32_slice();
        let len = scratch.ffn_out.len();
        scratch.ffn_out.copy_from_slice(&locked[..len]);
    }
    vdsp::vsma(&scratch.ffn_out, alpha, &scratch.x2, x_next);
    (ffn_t0.elapsed().as_secs_f32() * 1000.0, total_hw_ns)
}

fn finalize_logits_and_loss(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ws: &mut ModelForwardWorkspace,
    targets: &[u32],
    softcap: f32,
) -> (Vec<f32>, f32) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    ws.x_prenorm.copy_from_slice(&ws.x_buf);
    rmsnorm::forward_channel_first(
        &ws.x_prenorm,
        &weights.gamma_final,
        &mut ws.x_final,
        &mut ws.rms_inv_final,
        dim,
        seq,
    );
    vdsp::mtrans(&ws.x_final, seq, &mut ws.x_final_row, dim, dim, seq);
    ws.logits.fill(0.0);
    vdsp::sgemm_at(
        &ws.x_final_row,
        seq,
        dim,
        &weights.embed,
        vocab,
        &mut ws.logits,
    );

    if softcap > 0.0 {
        vdsp::sscal(&mut ws.logits, 1.0 / softcap);
        vdsp::tanhf(&ws.logits, &mut ws.logits_capped);
        vdsp::vsmul(&ws.logits_capped, softcap, &mut ws.logits);
    }

    let total_loss = cross_entropy::forward_backward_batch(
        &ws.logits,
        targets,
        vocab,
        &mut ws.dlogits,
        1.0 / seq as f32,
    );

    (ws.logits.clone(), total_loss / seq as f32)
}

#[allow(dead_code)]
fn run_baseline_reference_once(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> FullModelOutputs {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let mut x_row = vec![0.0f32; seq * dim];
    let mut x_buf = vec![0.0f32; dim * seq];
    let mut total_ffn_ms = 0.0f32;

    embedding::forward(&weights.embed, dim, tokens, &mut x_row);
    vdsp::mtrans(&x_row, dim, &mut x_buf, seq, seq, dim);

    for layer_weights in &weights.layers {
        let (x_next, _, timings) = layer::forward_timed(cfg, kernels, layer_weights, &x_buf);
        total_ffn_ms += timings.stage_ffn_ms + timings.ane_ffn_ms + timings.read_ffn_ms;
        x_buf = x_next;
    }

    let mut ws = ModelForwardWorkspace::new_lean(cfg);
    ws.x_buf.copy_from_slice(&x_buf);
    let (logits, loss) = finalize_logits_and_loss(cfg, weights, &mut ws, targets, softcap);
    FullModelOutputs {
        logits,
        loss,
        total_ffn_ms,
    }
}

#[allow(dead_code)]
fn run_sharded_full_forward_once(
    runner: &mut ShardedFfnRunner,
    ws: &mut ModelForwardWorkspace,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> FullModelOutputs {
    let cfg = runner.cfg.clone();
    let dim = cfg.dim;
    let seq = cfg.seq;

    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    let mut total_ffn_ms = 0.0f32;
    for layer_weights in &weights.layers {
        let shard_layouts = runner.shard_layouts.clone();
        let x_buf = ws.x_buf.clone();
        let (ffn_ms, _) = run_sharded_layer_forward_into(
            runner,
            layer_weights,
            &shard_layouts,
            &x_buf,
            &mut ws.x_next_buf,
        );
        total_ffn_ms += ffn_ms;
        std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
    }

    let (logits, loss) = finalize_logits_and_loss(&cfg, weights, ws, targets, softcap);
    FullModelOutputs {
        logits,
        loss,
        total_ffn_ms,
    }
}

fn run_sharded_full_forward_once_with_stats(
    runner: &mut ShardedFfnRunner,
    ws: &mut ModelForwardWorkspace,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> (FullModelOutputs, u64) {
    let cfg = runner.cfg.clone();
    let dim = cfg.dim;
    let seq = cfg.seq;

    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    let mut total_ffn_ms = 0.0f32;
    let mut total_hw_ns = 0u64;
    for layer_weights in &weights.layers {
        let shard_layouts = runner.shard_layouts.clone();
        let x_buf = ws.x_buf.clone();
        let (ffn_ms, hw_ns) = run_sharded_layer_forward_into_stats(
            runner,
            layer_weights,
            &shard_layouts,
            &x_buf,
            &mut ws.x_next_buf,
        );
        total_ffn_ms += ffn_ms;
        total_hw_ns += hw_ns;
        std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
    }

    let (logits, loss) = finalize_logits_and_loss(&cfg, weights, ws, targets, softcap);
    (
        FullModelOutputs {
            logits,
            loss,
            total_ffn_ms,
        },
        total_hw_ns,
    )
}

fn run_baseline_full_forward_once(
    runner: &mut BaselineRunner,
    ws: &mut ModelForwardWorkspace,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
    options: ExecOptions,
) -> (FullModelOutputs, u64) {
    let cfg = runner.cfg.clone();
    let dim = cfg.dim;
    let seq = cfg.seq;

    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    let mut total_ffn_ms = 0.0f32;
    let mut total_hw_ns = 0u64;
    for layer_weights in &weights.layers {
        let x_buf = ws.x_buf.clone();
        let (ffn_ms, hw_ns) = run_baseline_layer_forward_into(
            runner,
            layer_weights,
            &x_buf,
            &mut ws.x_next_buf,
            options,
        );
        total_ffn_ms += ffn_ms;
        total_hw_ns += hw_ns;
        std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
    }

    let (logits, loss) = finalize_logits_and_loss(&cfg, weights, ws, targets, softcap);
    (
        FullModelOutputs {
            logits,
            loss,
            total_ffn_ms,
        },
        total_hw_ns,
    )
}

#[allow(dead_code)]
fn measure_baseline_ane_stats(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    baseline_logits: &[f32],
    baseline_loss: f32,
) -> AneSample {
    let softcap = TrainConfig::default().softcap;
    let (mut runner, _) = compile_baseline_runner(cfg);
    let mut ws = ModelForwardWorkspace::new_lean(cfg);

    let (warmup, _) = run_baseline_full_forward_once(
        &mut runner,
        &mut ws,
        weights,
        tokens,
        targets,
        softcap,
        ExecOptions::STATS,
    );
    let diff = compare_logits_and_loss(&warmup.logits, warmup.loss, baseline_logits, baseline_loss);
    assert!(
        diff.passes(),
        "baseline stats pass diverged from baseline: {:?}",
        diff
    );

    let mut hw_samples_ms = Vec::with_capacity(ANE_STATS_RUNS);
    for _ in 0..ANE_STATS_RUNS {
        let t0 = Instant::now();
        let (out, hw_ns) = run_baseline_full_forward_once(
            &mut runner,
            &mut ws,
            weights,
            tokens,
            targets,
            softcap,
            ExecOptions::STATS,
        );
        let diff = compare_logits_and_loss(&out.logits, out.loss, baseline_logits, baseline_loss);
        assert!(
            diff.passes(),
            "baseline stats sample diverged from baseline: {:?}",
            diff
        );
        let _wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
        hw_samples_ms.push(hw_ns as f32 / 1_000_000.0);
    }

    AneSample {
        hw_ms: mean(&hw_samples_ms),
    }
}

#[allow(dead_code)]
fn measure_sharded_ane_stats(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    baseline_logits: &[f32],
    baseline_loss: f32,
    shard_count: usize,
) -> AneSample {
    let softcap = TrainConfig::default().softcap;
    let (mut runner, _) = compile_sharded_runner(cfg, shard_count);
    let mut ws = ModelForwardWorkspace::new_lean(cfg);

    let (warmup, _) = run_sharded_full_forward_once_with_stats(
        &mut runner,
        &mut ws,
        weights,
        tokens,
        targets,
        softcap,
    );
    let diff = compare_logits_and_loss(&warmup.logits, warmup.loss, baseline_logits, baseline_loss);
    assert!(
        diff.passes(),
        "sharded stats pass diverged from baseline for shard{}: {:?}",
        shard_count,
        diff
    );

    let mut hw_samples_ms = Vec::with_capacity(ANE_STATS_RUNS);
    for _ in 0..ANE_STATS_RUNS {
        let t0 = Instant::now();
        let (out, hw_ns) = run_sharded_full_forward_once_with_stats(
            &mut runner,
            &mut ws,
            weights,
            tokens,
            targets,
            softcap,
        );
        let diff = compare_logits_and_loss(&out.logits, out.loss, baseline_logits, baseline_loss);
        assert!(
            diff.passes(),
            "sharded stats sample diverged from baseline for shard{}: {:?}",
            shard_count,
            diff
        );
        let _wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
        hw_samples_ms.push(hw_ns as f32 / 1_000_000.0);
    }

    AneSample {
        hw_ms: mean(&hw_samples_ms),
    }
}

fn zero_diff_metrics() -> DiffMetrics {
    DiffMetrics {
        max_abs_diff: 0.0,
        mean_abs_diff: 0.0,
        cosine_similarity: 1.0,
        loss_abs_diff: 0.0,
    }
}

fn run_baseline_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
) -> (ModeResult, Vec<f32>, f32) {
    let softcap = TrainConfig::default().softcap;
    let (mut runner, compile_s) = compile_baseline_runner(cfg);

    let mut ws = ModelForwardWorkspace::new_lean(cfg);
    let mut peak_rss_mb = rss_mb().unwrap_or(0.0);

    let t0 = Instant::now();
    let (warmup, _) = run_baseline_full_forward_once(
        &mut runner,
        &mut ws,
        weights,
        tokens,
        targets,
        softcap,
        ExecOptions::STATS,
    );
    let warmup_wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let baseline_logits = warmup.logits.clone();
    let baseline_loss = warmup.loss;
    peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));

    let mut timed_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut total_ffn_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut hw_samples_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let (out, hw_ns) = run_baseline_full_forward_once(
            &mut runner,
            &mut ws,
            weights,
            tokens,
            targets,
            softcap,
            ExecOptions::STATS,
        );
        let diff = compare_logits_and_loss(&out.logits, out.loss, &baseline_logits, baseline_loss);
        assert!(
            diff.passes(),
            "baseline measured sample diverged from warmup baseline: {:?}",
            diff
        );
        timed_samples_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
        total_ffn_samples_ms.push(out.total_ffn_ms);
        hw_samples_ms.push(hw_ns as f32 / 1_000_000.0);
        peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));
    }

    let mean_full_forward_ms = mean(&timed_samples_ms);
    let total_ffn_ms = mean(&total_ffn_samples_ms);
    let ane_sample = AneSample {
        hw_ms: mean(&hw_samples_ms),
    };
    let tok_per_s = cfg.seq as f32 * 1000.0 / mean_full_forward_ms;
    let total_non_ffn_ms = (mean_full_forward_ms - total_ffn_ms).max(0.0);
    let ane_busy_pct = if mean_full_forward_ms > 0.0 {
        ane_sample.hw_ms / mean_full_forward_ms * 100.0
    } else {
        0.0
    };
    let ffn_pct_of_wall = if mean_full_forward_ms > 0.0 {
        total_ffn_ms / mean_full_forward_ms * 100.0
    } else {
        0.0
    };

    (
        ModeResult {
            mode: "baseline".to_string(),
            shard_count: 1,
            compile_s,
            warmup_wall_ms,
            timed_samples_ms,
            mean_full_forward_ms,
            tok_per_s,
            peak_rss_mb,
            ane_hw_ms: ane_sample.hw_ms,
            ane_busy_pct,
            avg_layer_ms: mean_full_forward_ms / cfg.nlayers as f32,
            total_ffn_ms,
            total_non_ffn_ms,
            ffn_pct_of_wall,
            diff_vs_baseline: zero_diff_metrics(),
            correctness_pass: true,
            full_forward_win_pct_vs_baseline: 0.0,
            meets_perf_gate: false,
            bucket_note: Some("1 warmup plus 2 measured stats-enabled full-model passes; wall, FFN, RSS, and ANE usage all come from the same measured samples.".to_string()),
        },
        baseline_logits,
        baseline_loss,
    )
}

fn run_sharded_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    shard_count: usize,
    baseline_logits: &[f32],
    baseline_loss: f32,
    baseline_mean_ms: f32,
) -> ModeResult {
    let softcap = TrainConfig::default().softcap;
    let (mut runner, compile_s) = compile_sharded_runner(cfg, shard_count);
    let mut ws = ModelForwardWorkspace::new_lean(cfg);
    let mut peak_rss_mb = rss_mb().unwrap_or(0.0);

    let t0 = Instant::now();
    let (warmup, _) = run_sharded_full_forward_once_with_stats(
        &mut runner,
        &mut ws,
        weights,
        tokens,
        targets,
        softcap,
    );
    let warmup_wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let diff = compare_logits_and_loss(&warmup.logits, warmup.loss, baseline_logits, baseline_loss);
    peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));

    let mut timed_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut total_ffn_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut hw_samples_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let (out, hw_ns) = run_sharded_full_forward_once_with_stats(
            &mut runner,
            &mut ws,
            weights,
            tokens,
            targets,
            softcap,
        );
        let sample_diff =
            compare_logits_and_loss(&out.logits, out.loss, baseline_logits, baseline_loss);
        assert!(
            sample_diff.passes(),
            "sharded measured sample diverged from baseline for shard{}: {:?}",
            shard_count,
            sample_diff
        );
        let total_ms = t0.elapsed().as_secs_f32() * 1000.0;
        timed_samples_ms.push(total_ms);
        total_ffn_samples_ms.push(out.total_ffn_ms);
        hw_samples_ms.push(hw_ns as f32 / 1_000_000.0);
        peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));
    }

    let mean_full_forward_ms = mean(&timed_samples_ms);
    let total_ffn_ms = mean(&total_ffn_samples_ms);
    let total_non_ffn_ms = (mean_full_forward_ms - total_ffn_ms).max(0.0);
    let tok_per_s = cfg.seq as f32 * 1000.0 / mean_full_forward_ms;
    let ane_sample = AneSample {
        hw_ms: mean(&hw_samples_ms),
    };
    let ane_busy_pct = if mean_full_forward_ms > 0.0 {
        ane_sample.hw_ms / mean_full_forward_ms * 100.0
    } else {
        0.0
    };
    let ffn_pct_of_wall = if mean_full_forward_ms > 0.0 {
        total_ffn_ms / mean_full_forward_ms * 100.0
    } else {
        0.0
    };
    let full_forward_win_pct_vs_baseline =
        (baseline_mean_ms - mean_full_forward_ms) / baseline_mean_ms * 100.0;

    ModeResult {
        mode: format!("shard{shard_count}"),
        shard_count,
        compile_s,
        warmup_wall_ms,
        timed_samples_ms,
        mean_full_forward_ms,
        tok_per_s,
        peak_rss_mb,
        ane_hw_ms: ane_sample.hw_ms,
        ane_busy_pct,
        avg_layer_ms: mean_full_forward_ms / cfg.nlayers as f32,
        total_ffn_ms,
        total_non_ffn_ms,
        ffn_pct_of_wall,
        diff_vs_baseline: diff.clone(),
        correctness_pass: diff.passes(),
        full_forward_win_pct_vs_baseline,
        meets_perf_gate: diff.passes() && full_forward_win_pct_vs_baseline >= PERF_WIN_PCT_TOL,
        bucket_note: Some("1 warmup plus 2 measured stats-enabled full-model passes; wall, FFN, RSS, and ANE usage all come from the same measured samples.".to_string()),
    }
}

fn write_summary(result: &ExperimentResult, path: &Path) {
    let mut summary = String::new();
    summary.push_str("# Full-Model FFN Latency Parallel Comparison\n\n");
    summary.push_str(&format!(
        "Config: `{}` — {}d/{}h/{}L/seq{}\n\n",
        result.config_name, result.dim, result.hidden, result.nlayers, result.seq
    ));

    summary.push_str("| mode | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | ANE hw(ms) | ANE busy % | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |\n");
    summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    for mode in [
        &result.baseline,
        &result.shard2,
        &result.shard4,
        &result.shard6,
        &result.shard8,
        &result.shard12,
        &result.shard16,
    ] {
        let diff = &mode.diff_vs_baseline;
        let _ = writeln!(
            summary,
            "| {} | {:.2} | {:.1} | {:.1} | {:.2} | {:.0} | {:.1} | {:.1}% | {:.3} | {:.1} | {:.1} | {:.1}% | {:.4} | {:.4} | {:.6} | {:.6} | {:+.2}% |",
            mode.mode,
            mode.compile_s,
            mode.warmup_wall_ms,
            mode.mean_full_forward_ms,
            mode.tok_per_s,
            mode.peak_rss_mb,
            mode.ane_hw_ms,
            mode.ane_busy_pct,
            mode.avg_layer_ms,
            mode.total_ffn_ms,
            mode.total_non_ffn_ms,
            mode.ffn_pct_of_wall,
            diff.max_abs_diff,
            diff.mean_abs_diff,
            diff.cosine_similarity,
            diff.loss_abs_diff,
            mode.full_forward_win_pct_vs_baseline,
        );
    }

    summary.push_str("\n## Thresholds\n\n");
    summary.push_str(&format!(
        "- max abs diff <= `{:.3e}`\n- mean abs diff <= `{:.3e}`\n- cosine similarity >= `{:.3}`\n- loss abs diff <= `{:.3e}`\n- full forward win >= `{:.1}%`\n\n",
        result.thresholds.max_abs_diff_tol,
        result.thresholds.mean_abs_diff_tol,
        result.thresholds.cosine_similarity_tol,
        result.thresholds.loss_abs_diff_tol,
        result.thresholds.perf_win_pct_tol
    ));

    summary.push_str(&format!(
        "Primary success: **{}**\n\n",
        if result.primary_success {
            "PASS"
        } else {
            "FAIL"
        }
    ));
    if !result.winning_modes.is_empty() {
        summary.push_str(&format!(
            "Winning modes: `{}`\n\n",
            result.winning_modes.join("`, `")
        ));
    }
    summary.push_str("## Notes\n\n");
    summary
        .push_str("- Each mode runs 1 warmup plus 2 measured stats-enabled full-model passes.\n");
    summary.push_str("- Reported wall time, FFN totals, RSS, and ANE utilization all come from the same measured samples.\n");
    summary.push_str("- This harness uses the mirrored stats-enabled path, not `full_model::forward_only_ws(...)`.\n");

    ensure_results_dir();
    fs::write(path, summary).expect("write summary");
}

fn write_selected_summary(result: &SelectedExperimentResult, path: &Path) {
    let mut summary = String::new();
    summary.push_str("# Full-Model FFN Latency Parallel Comparison\n\n");
    summary.push_str(&format!(
        "Config: `{}` — {}d/{}h/{}L/seq{}\n\n",
        result.config_name, result.dim, result.hidden, result.nlayers, result.seq
    ));

    summary.push_str("| mode | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | ANE hw(ms) | ANE busy % | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |\n");
    summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    let baseline_diff = &result.baseline.diff_vs_baseline;
    let _ = writeln!(
        summary,
        "| {} | {:.2} | {:.1} | {:.1} | {:.2} | {:.0} | {:.1} | {:.1}% | {:.3} | {:.1} | {:.1} | {:.1}% | {:.4} | {:.4} | {:.6} | {:.6} | {:+.2}% |",
        result.baseline.mode,
        result.baseline.compile_s,
        result.baseline.warmup_wall_ms,
        result.baseline.mean_full_forward_ms,
        result.baseline.tok_per_s,
        result.baseline.peak_rss_mb,
        result.baseline.ane_hw_ms,
        result.baseline.ane_busy_pct,
        result.baseline.avg_layer_ms,
        result.baseline.total_ffn_ms,
        result.baseline.total_non_ffn_ms,
        result.baseline.ffn_pct_of_wall,
        baseline_diff.max_abs_diff,
        baseline_diff.mean_abs_diff,
        baseline_diff.cosine_similarity,
        baseline_diff.loss_abs_diff,
        result.baseline.full_forward_win_pct_vs_baseline,
    );

    for mode in &result.modes {
        let diff = &mode.diff_vs_baseline;
        let _ = writeln!(
            summary,
            "| {} | {:.2} | {:.1} | {:.1} | {:.2} | {:.0} | {:.1} | {:.1}% | {:.3} | {:.1} | {:.1} | {:.1}% | {:.4} | {:.4} | {:.6} | {:.6} | {:+.2}% |",
            mode.mode,
            mode.compile_s,
            mode.warmup_wall_ms,
            mode.mean_full_forward_ms,
            mode.tok_per_s,
            mode.peak_rss_mb,
            mode.ane_hw_ms,
            mode.ane_busy_pct,
            mode.avg_layer_ms,
            mode.total_ffn_ms,
            mode.total_non_ffn_ms,
            mode.ffn_pct_of_wall,
            diff.max_abs_diff,
            diff.mean_abs_diff,
            diff.cosine_similarity,
            diff.loss_abs_diff,
            mode.full_forward_win_pct_vs_baseline,
        );
    }

    summary.push_str("\n## Thresholds\n\n");
    summary.push_str(&format!(
        "- max abs diff <= `{:.3e}`\n- mean abs diff <= `{:.3e}`\n- cosine similarity >= `{:.3}`\n- loss abs diff <= `{:.3e}`\n- full forward win >= `{:.1}%`\n\n",
        result.thresholds.max_abs_diff_tol,
        result.thresholds.mean_abs_diff_tol,
        result.thresholds.cosine_similarity_tol,
        result.thresholds.loss_abs_diff_tol,
        result.thresholds.perf_win_pct_tol
    ));

    summary.push_str(&format!(
        "Primary success: **{}**\n\n",
        if result.primary_success {
            "PASS"
        } else {
            "FAIL"
        }
    ));
    if !result.winning_modes.is_empty() {
        summary.push_str(&format!(
            "Winning modes: `{}`\n\n",
            result.winning_modes.join("`, `")
        ));
    }
    summary.push_str("## Notes\n\n");
    summary
        .push_str("- Each mode runs 1 warmup plus 2 measured stats-enabled full-model passes.\n");
    summary.push_str("- Reported wall time, FFN totals, RSS, and ANE utilization all come from the same measured samples.\n");
    summary.push_str("- This harness uses the mirrored stats-enabled path, not `full_model::forward_only_ws(...)`.\n");

    ensure_results_dir();
    fs::write(path, summary).expect("write summary");
}

fn run_experiment(cfg: &ModelConfig, name: &str) -> ExperimentResult {
    let weights = ModelWeights::random(cfg);
    let (tokens, targets) = deterministic_tokens(cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(cfg, &weights, &tokens, &targets);

    let shard2 = run_sharded_mode(
        cfg,
        &weights,
        &tokens,
        &targets,
        2,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard4 = run_sharded_mode(
        cfg,
        &weights,
        &tokens,
        &targets,
        4,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard6 = run_sharded_mode(
        cfg,
        &weights,
        &tokens,
        &targets,
        6,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard8 = run_sharded_mode(
        cfg,
        &weights,
        &tokens,
        &targets,
        8,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard12 = run_sharded_mode(
        cfg,
        &weights,
        &tokens,
        &targets,
        12,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard16 = run_sharded_mode(
        cfg,
        &weights,
        &tokens,
        &targets,
        16,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );

    let winning_modes = [
        (&shard2, "shard2"),
        (&shard4, "shard4"),
        (&shard6, "shard6"),
        (&shard8, "shard8"),
        (&shard12, "shard12"),
        (&shard16, "shard16"),
    ]
    .into_iter()
    .filter_map(|(mode, name)| mode.meets_perf_gate.then_some(name.to_string()))
    .collect::<Vec<_>>();

    ExperimentResult {
        config_name: name.to_string(),
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
        baseline,
        shard2,
        shard4,
        shard6,
        shard8,
        shard12,
        shard16,
        primary_success: !winning_modes.is_empty(),
        winning_modes,
        thresholds: Thresholds {
            max_abs_diff_tol: MAX_ABS_DIFF_TOL,
            mean_abs_diff_tol: MEAN_ABS_DIFF_TOL,
            cosine_similarity_tol: COSINE_SIM_TOL,
            loss_abs_diff_tol: LOSS_ABS_DIFF_TOL,
            perf_win_pct_tol: PERF_WIN_PCT_TOL,
        },
    }
}

#[test]
#[ignore]
fn ffn_shard_full_model_smoke_30b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let cfg = cfg_30b();
    let weights = ModelWeights::random(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(&cfg, &weights, &tokens, &targets);

    let shard2 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        2,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard4 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        4,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard6 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        6,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard8 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        8,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard12 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        12,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard16 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        16,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );

    assert!(
        shard2.correctness_pass,
        "30B shard2 correctness failed: {:?}",
        shard2.diff_vs_baseline
    );
    assert!(
        shard4.correctness_pass,
        "30B shard4 correctness failed: {:?}",
        shard4.diff_vs_baseline
    );
    assert!(
        shard6.correctness_pass,
        "30B shard6 correctness failed: {:?}",
        shard6.diff_vs_baseline
    );
    assert!(
        shard8.correctness_pass,
        "30B shard8 correctness failed: {:?}",
        shard8.diff_vs_baseline
    );
    assert!(
        shard12.correctness_pass,
        "30B shard12 correctness failed: {:?}",
        shard12.diff_vs_baseline
    );
    assert!(
        shard16.correctness_pass,
        "30B shard16 correctness failed: {:?}",
        shard16.diff_vs_baseline
    );
}

#[test]
#[ignore]
fn bench_ffn_shard_full_model_30b() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_30b(), "30B");
    assert!(
        result.shard2.correctness_pass,
        "30B shard2 correctness failed"
    );
    assert!(
        result.shard4.correctness_pass,
        "30B shard4 correctness failed"
    );
    assert!(
        result.shard6.correctness_pass,
        "30B shard6 correctness failed"
    );
    assert!(
        result.shard8.correctness_pass,
        "30B shard8 correctness failed"
    );
    assert!(
        result.shard12.correctness_pass,
        "30B shard12 correctness failed"
    );
    assert!(
        result.shard16.correctness_pass,
        "30B shard16 correctness failed"
    );
    write_json(&json_path("30b"), &result);
    write_summary(&result, &summary_path("30b"));
}

#[test]
#[ignore]
fn ffn_shard_full_model_smoke_50b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let cfg = cfg_50b();
    let weights = ModelWeights::random(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(&cfg, &weights, &tokens, &targets);

    let shard2 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        2,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard4 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        4,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard6 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        6,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard8 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        8,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard12 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        12,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard16 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        16,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );

    assert!(
        shard2.correctness_pass,
        "50B shard2 correctness failed: {:?}",
        shard2.diff_vs_baseline
    );
    assert!(
        shard4.correctness_pass,
        "50B shard4 correctness failed: {:?}",
        shard4.diff_vs_baseline
    );
    assert!(
        shard6.correctness_pass,
        "50B shard6 correctness failed: {:?}",
        shard6.diff_vs_baseline
    );
    assert!(
        shard8.correctness_pass,
        "50B shard8 correctness failed: {:?}",
        shard8.diff_vs_baseline
    );
    assert!(
        shard12.correctness_pass,
        "50B shard12 correctness failed: {:?}",
        shard12.diff_vs_baseline
    );
    assert!(
        shard16.correctness_pass,
        "50B shard16 correctness failed: {:?}",
        shard16.diff_vs_baseline
    );
}

#[test]
#[ignore]
fn bench_ffn_shard_full_model_50b_equal() {
    let _guard = run_lock().lock().unwrap();
    let cfg = cfg_50b();
    let weights = ModelWeights::random(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(&cfg, &weights, &tokens, &targets);

    let shard2 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        2,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard4 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        4,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard6 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        6,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard8 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        8,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard12 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        12,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard16 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        16,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );

    assert!(shard2.correctness_pass, "50B shard2 correctness failed");
    assert!(shard4.correctness_pass, "50B shard4 correctness failed");
    assert!(shard6.correctness_pass, "50B shard6 correctness failed");
    assert!(shard8.correctness_pass, "50B shard8 correctness failed");
    assert!(shard12.correctness_pass, "50B shard12 correctness failed");
    assert!(shard16.correctness_pass, "50B shard16 correctness failed");

    let modes = vec![shard2, shard4, shard6, shard8, shard12, shard16];
    let winning_modes = modes
        .iter()
        .filter_map(|m| m.meets_perf_gate.then_some(m.mode.clone()))
        .collect::<Vec<_>>();

    let result = SelectedExperimentResult {
        config_name: "50B".to_string(),
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
        baseline,
        modes,
        primary_success: !winning_modes.is_empty(),
        winning_modes,
        thresholds: Thresholds {
            max_abs_diff_tol: MAX_ABS_DIFF_TOL,
            mean_abs_diff_tol: MEAN_ABS_DIFF_TOL,
            cosine_similarity_tol: COSINE_SIM_TOL,
            loss_abs_diff_tol: LOSS_ABS_DIFF_TOL,
            perf_win_pct_tol: PERF_WIN_PCT_TOL,
        },
    };

    write_json(&json_path("50b"), &result);
    write_selected_summary(&result, &summary_path("50b"));
}

#[test]
#[ignore]
fn ffn_shard_full_model_smoke_80b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let cfg = cfg_80b();
    let weights = ModelWeights::random(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(&cfg, &weights, &tokens, &targets);

    let shard2 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        2,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard4 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        4,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard6 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        6,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard8 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        8,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard12 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        12,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard16 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        16,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard24 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        24,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard32 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        32,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );

    assert!(
        shard2.correctness_pass,
        "80B shard2 correctness failed: {:?}",
        shard2.diff_vs_baseline
    );
    assert!(
        shard4.correctness_pass,
        "80B shard4 correctness failed: {:?}",
        shard4.diff_vs_baseline
    );
    assert!(
        shard6.correctness_pass,
        "80B shard6 correctness failed: {:?}",
        shard6.diff_vs_baseline
    );
    assert!(
        shard8.correctness_pass,
        "80B shard8 correctness failed: {:?}",
        shard8.diff_vs_baseline
    );
    assert!(
        shard12.correctness_pass,
        "80B shard12 correctness failed: {:?}",
        shard12.diff_vs_baseline
    );
    assert!(
        shard16.correctness_pass,
        "80B shard16 correctness failed: {:?}",
        shard16.diff_vs_baseline
    );
    assert!(
        shard24.correctness_pass,
        "80B shard24 correctness failed: {:?}",
        shard24.diff_vs_baseline
    );
    assert!(
        shard32.correctness_pass,
        "80B shard32 correctness failed: {:?}",
        shard32.diff_vs_baseline
    );
}

#[test]
#[ignore]
fn bench_ffn_shard_full_model_80b_equal() {
    let _guard = run_lock().lock().unwrap();
    let cfg = cfg_80b();
    let weights = ModelWeights::random(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(&cfg, &weights, &tokens, &targets);

    let shard2 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        2,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard4 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        4,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard6 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        6,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard8 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        8,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard12 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        12,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard16 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        16,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard24 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        24,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );
    let shard32 = run_sharded_mode(
        &cfg,
        &weights,
        &tokens,
        &targets,
        32,
        &baseline_logits,
        baseline_loss,
        baseline.mean_full_forward_ms,
    );

    assert!(shard2.correctness_pass, "80B shard2 correctness failed");
    assert!(shard4.correctness_pass, "80B shard4 correctness failed");
    assert!(shard6.correctness_pass, "80B shard6 correctness failed");
    assert!(shard8.correctness_pass, "80B shard8 correctness failed");
    assert!(shard12.correctness_pass, "80B shard12 correctness failed");
    assert!(shard16.correctness_pass, "80B shard16 correctness failed");
    assert!(shard24.correctness_pass, "80B shard24 correctness failed");
    assert!(shard32.correctness_pass, "80B shard32 correctness failed");

    let modes = vec![
        shard2, shard4, shard6, shard8, shard12, shard16, shard24, shard32,
    ];
    let winning_modes = modes
        .iter()
        .filter_map(|m| m.meets_perf_gate.then_some(m.mode.clone()))
        .collect::<Vec<_>>();

    let result = SelectedExperimentResult {
        config_name: "80B".to_string(),
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
        baseline,
        modes,
        primary_success: !winning_modes.is_empty(),
        winning_modes,
        thresholds: Thresholds {
            max_abs_diff_tol: MAX_ABS_DIFF_TOL,
            mean_abs_diff_tol: MEAN_ABS_DIFF_TOL,
            cosine_similarity_tol: COSINE_SIM_TOL,
            loss_abs_diff_tol: LOSS_ABS_DIFF_TOL,
            perf_win_pct_tol: PERF_WIN_PCT_TOL,
        },
    };

    write_json(&json_path("80b"), &result);
    write_selected_summary(&result, &summary_path("80b"));
}
