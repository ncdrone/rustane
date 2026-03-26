//! Benchmark-only full-model attention+FFN parallel experiment.
//!
//! Run build check:
//!   cargo test -p engine --test bench_attn_ffn_parallel_full_model --no-run
//!
//! Run 30B-geometry/16L smoke validation:
//!   cargo test -p engine --test bench_attn_ffn_parallel_full_model --release -- --ignored --nocapture attn_ffn_full_model_smoke_30bgeom_16l_matches_baseline
//!
//! Run full 30B-geometry/16L matrix benchmark:
//!   cargo test -p engine --test bench_attn_ffn_parallel_full_model --release -- --ignored --nocapture bench_attn_ffn_full_model_30bgeom_16l_matrix
//!
//! Run full 30B matrix benchmark:
//!   cargo test -p engine --test bench_attn_ffn_parallel_full_model --release -- --ignored --nocapture bench_attn_ffn_full_model_30b_matrix
//!
//! Run full 50B matrix benchmark:
//!   cargo test -p engine --test bench_attn_ffn_parallel_full_model --release -- --ignored --nocapture bench_attn_ffn_full_model_50b_matrix
//!
//! Run full 80B matrix benchmark:
//!   cargo test -p engine --test bench_attn_ffn_parallel_full_model --release -- --ignored --nocapture bench_attn_ffn_full_model_80b_matrix

use ane_bridge::ane::{Executable, TensorData};
use engine::cpu::{cross_entropy, embedding, rmsnorm, silu, vdsp};
use engine::full_model::{ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::kernels::{dyn_matmul, ffn_fused, sdpa_fwd};
use engine::layer::LayerWeights;
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/latency_parallel_attn_ffn")
}

fn json_path(stem: &str) -> PathBuf {
    results_dir().join(format!("full_model_{stem}_compare.json"))
}

fn summary_path(stem: &str) -> PathBuf {
    results_dir().join(format!("full_model_{stem}_summary.md"))
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

fn cfg_30bgeom_16l() -> ModelConfig {
    custom_config(5120, 13824, 40, 16, 512)
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

fn attention_group_cfg(cfg: &ModelConfig, head_count: usize) -> ModelConfig {
    ModelConfig {
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: head_count,
        kv_heads: head_count,
        hd: cfg.hd,
        seq: cfg.seq,
        nlayers: cfg.nlayers,
        vocab: cfg.vocab,
        q_dim: head_count * cfg.hd,
        kv_dim: head_count * cfg.hd,
        gqa_ratio: 1,
    }
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

fn layer_alpha(cfg: &ModelConfig) -> f32 {
    1.0 / (2.0 * cfg.nlayers as f32).sqrt()
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
    assert_eq!(dst.len(), end - start, "destination length mismatch");
    dst.copy_from_slice(&src[start..end]);
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

fn zero_diff_metrics() -> DiffMetrics {
    DiffMetrics {
        max_abs_diff: 0.0,
        mean_abs_diff: 0.0,
        cosine_similarity: 1.0,
        loss_abs_diff: 0.0,
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
    attn_shards: usize,
    ffn_shards: usize,
    compile_s: f32,
    warmup_wall_ms: f32,
    timed_samples_ms: Vec<f32>,
    mean_full_forward_ms: f32,
    tok_per_s: f32,
    peak_rss_mb: f32,
    total_attn_ms: f32,
    total_ffn_ms: f32,
    total_other_ms: f32,
    attn_pct_of_wall: f32,
    ffn_pct_of_wall: f32,
    diff_vs_baseline: DiffMetrics,
    correctness_pass: bool,
    full_forward_win_pct_vs_baseline: f32,
    meets_perf_gate: bool,
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
    modes: Vec<ModeResult>,
    best_attention_only_mode: Option<String>,
    best_ffn_only_mode: Option<String>,
    best_combined_mode: Option<String>,
    primary_success: bool,
    winning_modes: Vec<String>,
    thresholds: Thresholds,
}

#[derive(Debug, Clone)]
struct ModeSpec {
    name: String,
    attn_shards: usize,
    ffn_shards: usize,
}

impl ModeSpec {
    fn baseline() -> Self {
        Self {
            name: "baseline".to_string(),
            attn_shards: 1,
            ffn_shards: 1,
        }
    }

    fn attention_only(attn_shards: usize) -> Self {
        Self {
            name: format!("attn{attn_shards}"),
            attn_shards,
            ffn_shards: 1,
        }
    }

    fn ffn_only(ffn_shards: usize) -> Self {
        Self {
            name: format!("ffn{ffn_shards}"),
            attn_shards: 1,
            ffn_shards,
        }
    }

    fn combined(ffn_shards: usize, attn_shards: usize) -> Self {
        Self {
            name: format!("ffn{ffn_shards}_attn{attn_shards}"),
            attn_shards,
            ffn_shards,
        }
    }
}

fn mode_specs() -> Vec<ModeSpec> {
    let mut specs = vec![ModeSpec::baseline()];
    for attn_shards in [2usize, 4, 5, 8] {
        specs.push(ModeSpec::attention_only(attn_shards));
    }
    for ffn_shards in [8usize, 12, 16] {
        specs.push(ModeSpec::ffn_only(ffn_shards));
    }
    for ffn_shards in [8usize, 12, 16] {
        for attn_shards in [2usize, 4, 5, 8] {
            specs.push(ModeSpec::combined(ffn_shards, attn_shards));
        }
    }
    specs
}

struct FullModelOutputs {
    logits: Vec<f32>,
    loss: f32,
    total_attn_ms: f32,
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
struct AttentionShardLayout {
    head_count: usize,
    q_col_start: usize,
    q_col_count: usize,
}

#[derive(Clone, Copy)]
struct FfnShardLayout {
    col_start: usize,
    col_count: usize,
}

struct AttentionShardWorker {
    layout: AttentionShardLayout,
    cfg: ModelConfig,
    sdpa: SdpaRunner,
    wo: WoRunner,
}

struct FfnShardWorker {
    layout: FfnShardLayout,
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
            o_out: vec![0.0; cfg.dim * cfg.seq],
            x2: vec![0.0; cfg.dim * cfg.seq],
            x2norm: vec![0.0; cfg.dim * cfg.seq],
            rms_inv2: vec![0.0; cfg.seq],
            ffn_out: vec![0.0; cfg.dim * cfg.seq],
            merge_tmp: vec![0.0; cfg.dim * cfg.seq],
        }
    }
}

struct BaselineAttentionRunner {
    cfg: ModelConfig,
    sdpa: SdpaRunner,
    wo: WoRunner,
}

struct ShardedAttentionRunner {
    cfg: ModelConfig,
    workers: Vec<AttentionShardWorker>,
}

struct BaselineFfnRunner {
    cfg: ModelConfig,
    use_dual_w13: bool,
    w13_exe: Executable,
    w13_in: TensorData,
    w13_out: TensorData,
    w2_exe: Executable,
    w2_in: TensorData,
    w2_out: TensorData,
    h1: Vec<f32>,
    h3: Vec<f32>,
    gate: Vec<f32>,
}

struct ShardedFfnRunner {
    cfg: ModelConfig,
    workers: Vec<FfnShardWorker>,
}

enum AttentionRunner {
    Baseline(BaselineAttentionRunner),
    Sharded(ShardedAttentionRunner),
}

enum FfnRunner {
    Baseline(BaselineFfnRunner),
    Sharded(ShardedFfnRunner),
}

struct ModeRunner {
    attn: AttentionRunner,
    ffn: FfnRunner,
    scratch: LayerScratch,
}

fn attention_shard_layouts(cfg: &ModelConfig, shard_count: usize) -> Vec<AttentionShardLayout> {
    assert!(
        cfg.heads % shard_count == 0,
        "heads must be divisible by attention shard count"
    );
    let head_count = cfg.heads / shard_count;
    let q_col_count = head_count * cfg.hd;
    (0..shard_count)
        .map(|i| AttentionShardLayout {
            head_count,
            q_col_start: i * q_col_count,
            q_col_count,
        })
        .collect()
}

fn ffn_shard_layouts(cfg: &ModelConfig, shard_count: usize) -> Vec<FfnShardLayout> {
    assert!(
        cfg.hidden % shard_count == 0,
        "hidden must be divisible by ffn shard count"
    );
    let col_count = cfg.hidden / shard_count;
    (0..shard_count)
        .map(|i| FfnShardLayout {
            col_start: i * col_count,
            col_count,
        })
        .collect()
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

fn compile_baseline_attention_runner(cfg: &ModelConfig) -> (BaselineAttentionRunner, f32) {
    let t0 = Instant::now();
    let runner = BaselineAttentionRunner {
        cfg: cfg.clone(),
        sdpa: compile_sdpa_runner(cfg),
        wo: compile_wo_runner(cfg),
    };
    (runner, t0.elapsed().as_secs_f32())
}

fn compile_sharded_attention_runner(
    cfg: &ModelConfig,
    shard_count: usize,
) -> (ShardedAttentionRunner, f32) {
    let t0 = Instant::now();
    let workers = attention_shard_layouts(cfg, shard_count)
        .into_iter()
        .map(|layout| {
            let worker_cfg = attention_group_cfg(cfg, layout.head_count);
            AttentionShardWorker {
                layout,
                cfg: worker_cfg.clone(),
                sdpa: compile_sdpa_runner(&worker_cfg),
                wo: compile_wo_runner(&worker_cfg),
            }
        })
        .collect();
    (
        ShardedAttentionRunner {
            cfg: cfg.clone(),
            workers,
        },
        t0.elapsed().as_secs_f32(),
    )
}

fn compile_baseline_ffn_runner(cfg: &ModelConfig) -> (BaselineFfnRunner, f32) {
    let qos = NSQualityOfService::UserInteractive;
    let use_dual_w13 = ffn_fused::can_use_dual_w13(cfg);
    let t0 = Instant::now();
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
    (
        BaselineFfnRunner {
            cfg: cfg.clone(),
            use_dual_w13,
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
        },
        t0.elapsed().as_secs_f32(),
    )
}

fn compile_sharded_ffn_runner(cfg: &ModelConfig, shard_count: usize) -> (ShardedFfnRunner, f32) {
    let qos = NSQualityOfService::UserInteractive;
    let t0 = Instant::now();
    let workers = ffn_shard_layouts(cfg, shard_count)
        .into_iter()
        .map(|layout| {
            let shard_hidden = layout.col_count;
            let w13_exe = dyn_matmul::build_dual_separate(cfg.dim, shard_hidden, cfg.seq)
                .compile(qos)
                .expect("sharded w13 compile");
            let w2_exe = dyn_matmul::build(shard_hidden, cfg.dim, cfg.seq)
                .compile(qos)
                .expect("sharded w2 compile");
            let w13_sp = dyn_matmul::dual_separate_spatial_width(cfg.seq, shard_hidden);
            let w2_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
            FfnShardWorker {
                layout,
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
            }
        })
        .collect();
    (
        ShardedFfnRunner {
            cfg: cfg.clone(),
            workers,
        },
        t0.elapsed().as_secs_f32(),
    )
}

fn compile_mode_runner(cfg: &ModelConfig, mode: &ModeSpec) -> (ModeRunner, f32) {
    let (attn_runner, attn_compile_s) = if mode.attn_shards == 1 {
        let (runner, compile_s) = compile_baseline_attention_runner(cfg);
        (AttentionRunner::Baseline(runner), compile_s)
    } else {
        let (runner, compile_s) = compile_sharded_attention_runner(cfg, mode.attn_shards);
        (AttentionRunner::Sharded(runner), compile_s)
    };

    let (ffn_runner, ffn_compile_s) = if mode.ffn_shards == 1 {
        let (runner, compile_s) = compile_baseline_ffn_runner(cfg);
        (FfnRunner::Baseline(runner), compile_s)
    } else {
        let (runner, compile_s) = compile_sharded_ffn_runner(cfg, mode.ffn_shards);
        (FfnRunner::Sharded(runner), compile_s)
    };

    (
        ModeRunner {
            attn: attn_runner,
            ffn: ffn_runner,
            scratch: LayerScratch::new(cfg),
        },
        attn_compile_s + ffn_compile_s,
    )
}

fn run_baseline_attention_into(
    runner: &mut BaselineAttentionRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    scratch: &mut LayerScratch,
) -> f32 {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);
    let t0 = Instant::now();

    rmsnorm::forward_channel_first(
        x,
        &layer_weights.gamma1,
        &mut scratch.xnorm,
        &mut scratch.rms_inv1,
        dim,
        seq,
    );
    runner.sdpa.xnorm_in.copy_from_f32(&scratch.xnorm);
    runner.sdpa.wq_in.copy_from_f32(&layer_weights.wq);
    runner.sdpa.wk_in.copy_from_f32(&layer_weights.wk);
    runner.sdpa.wv_in.copy_from_f32(&layer_weights.wv);
    runner
        .sdpa
        .exe
        .run_cached_direct(
            &[
                &runner.sdpa.xnorm_in,
                &runner.sdpa.wq_in,
                &runner.sdpa.wk_in,
                &runner.sdpa.wv_in,
            ],
            &[
                &runner.sdpa.attn_out,
                &runner.sdpa.q_rope_out,
                &runner.sdpa.k_rope_out,
                &runner.sdpa.v_out,
            ],
        )
        .expect("baseline sdpa run");

    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let attn_locked = runner.sdpa.attn_out.as_f32_slice();
        let mut wo_locked = runner.wo.input.as_f32_slice_mut();
        let buf = &mut *wo_locked;
        stage_spatial(
            buf,
            cfg.q_dim,
            wo_sp,
            &attn_locked[..cfg.q_dim * seq],
            seq,
            0,
        );
        stage_spatial(buf, cfg.q_dim, wo_sp, &layer_weights.wo, dim, seq);
    }
    runner
        .wo
        .exe
        .run_cached_direct(&[&runner.wo.input], &[&runner.wo.output])
        .expect("baseline wo run");
    {
        let locked = runner.wo.output.as_f32_slice();
        let len = scratch.o_out.len();
        scratch.o_out.copy_from_slice(&locked[..len]);
    }

    vdsp::vsma(&scratch.o_out, alpha, x, &mut scratch.x2);
    rmsnorm::forward_channel_first(
        &scratch.x2,
        &layer_weights.gamma2,
        &mut scratch.x2norm,
        &mut scratch.rms_inv2,
        dim,
        seq,
    );
    t0.elapsed().as_secs_f32() * 1000.0
}

fn run_sharded_attention_into(
    runner: &mut ShardedAttentionRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    scratch: &mut LayerScratch,
) -> f32 {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);
    let t0 = Instant::now();

    rmsnorm::forward_channel_first(
        x,
        &layer_weights.gamma1,
        &mut scratch.xnorm,
        &mut scratch.rms_inv1,
        dim,
        seq,
    );
    let barrier = Arc::new(Barrier::new(runner.workers.len() + 1));
    let xnorm_ref = &scratch.xnorm;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(runner.workers.len());
        for worker in &mut runner.workers {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                barrier.wait();

                worker.sdpa.xnorm_in.copy_from_f32(xnorm_ref);
                {
                    let mut locked = worker.sdpa.wq_in.as_f32_slice_mut();
                    stage_weight_columns(
                        &mut locked,
                        cfg.dim,
                        worker.cfg.q_dim,
                        &layer_weights.wq,
                        cfg.q_dim,
                        worker.layout.q_col_start,
                        worker.layout.q_col_count,
                        0,
                    );
                }
                {
                    let mut locked = worker.sdpa.wk_in.as_f32_slice_mut();
                    stage_weight_columns(
                        &mut locked,
                        cfg.dim,
                        worker.cfg.kv_dim,
                        &layer_weights.wk,
                        cfg.kv_dim,
                        worker.layout.q_col_start,
                        worker.layout.q_col_count,
                        0,
                    );
                }
                {
                    let mut locked = worker.sdpa.wv_in.as_f32_slice_mut();
                    stage_weight_columns(
                        &mut locked,
                        cfg.dim,
                        worker.cfg.kv_dim,
                        &layer_weights.wv,
                        cfg.kv_dim,
                        worker.layout.q_col_start,
                        worker.layout.q_col_count,
                        0,
                    );
                }

                worker
                    .sdpa
                    .exe
                    .run_cached_direct(
                        &[
                            &worker.sdpa.xnorm_in,
                            &worker.sdpa.wq_in,
                            &worker.sdpa.wk_in,
                            &worker.sdpa.wv_in,
                        ],
                        &[
                            &worker.sdpa.attn_out,
                            &worker.sdpa.q_rope_out,
                            &worker.sdpa.k_rope_out,
                            &worker.sdpa.v_out,
                        ],
                    )
                    .expect("sharded sdpa run");

                let wo_sp = dyn_matmul::spatial_width(seq, cfg.dim);
                let wo_row_start = worker.layout.q_col_start * cfg.dim;
                let wo_row_end = wo_row_start + worker.layout.q_col_count * cfg.dim;
                {
                    let attn_locked = worker.sdpa.attn_out.as_f32_slice();
                    let mut wo_locked = worker.wo.input.as_f32_slice_mut();
                    let buf = &mut *wo_locked;
                    stage_spatial(
                        buf,
                        worker.cfg.q_dim,
                        wo_sp,
                        &attn_locked[..worker.cfg.q_dim * seq],
                        seq,
                        0,
                    );
                    stage_spatial(
                        buf,
                        worker.cfg.q_dim,
                        wo_sp,
                        &layer_weights.wo[wo_row_start..wo_row_end],
                        cfg.dim,
                        seq,
                    );
                }
                worker
                    .wo
                    .exe
                    .run_cached_direct(&[&worker.wo.input], &[&worker.wo.output])
                    .expect("sharded wo run");
            }));
        }

        barrier.wait();
        for handle in handles {
            handle.join().expect("attention shard worker panicked");
        }
    });

    scratch.o_out.fill(0.0);
    for worker in &runner.workers {
        let locked = worker.wo.output.as_f32_slice();
        vdsp::vadd(
            &scratch.o_out,
            &locked[..scratch.o_out.len()],
            &mut scratch.merge_tmp,
        );
        scratch.o_out.copy_from_slice(&scratch.merge_tmp);
    }

    vdsp::vsma(&scratch.o_out, alpha, x, &mut scratch.x2);
    rmsnorm::forward_channel_first(
        &scratch.x2,
        &layer_weights.gamma2,
        &mut scratch.x2norm,
        &mut scratch.rms_inv2,
        dim,
        seq,
    );
    t0.elapsed().as_secs_f32() * 1000.0
}

fn run_baseline_ffn_into(
    runner: &mut BaselineFfnRunner,
    layer_weights: &LayerWeights,
    scratch: &mut LayerScratch,
    x_next: &mut [f32],
) -> f32 {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);
    let t0 = Instant::now();

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
        stage_weight_columns(
            buf,
            dim,
            w13_sp,
            &layer_weights.w1,
            cfg.hidden,
            0,
            cfg.hidden,
            seq,
        );
        if runner.use_dual_w13 {
            stage_weight_columns(
                buf,
                dim,
                w13_sp,
                &layer_weights.w3,
                cfg.hidden,
                0,
                cfg.hidden,
                seq + cfg.hidden,
            );
        }
    }

    if runner.use_dual_w13 {
        runner
            .w13_exe
            .run_cached_direct(&[&runner.w13_in], &[&runner.w13_out])
            .expect("baseline w13 dual run");
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
        runner
            .w13_exe
            .run_cached_direct(&[&runner.w13_in], &[&runner.w13_out])
            .expect("baseline w1 run");
        {
            let locked = runner.w13_out.as_f32_slice();
            runner.h1.copy_from_slice(&locked[..cfg.hidden * seq]);
        }
        {
            let mut locked = runner.w13_in.as_f32_slice_mut();
            let buf = &mut *locked;
            stage_spatial(buf, dim, w13_sp, &scratch.x2norm, seq, 0);
            stage_weight_columns(
                buf,
                dim,
                w13_sp,
                &layer_weights.w3,
                cfg.hidden,
                0,
                cfg.hidden,
                seq,
            );
        }
        runner
            .w13_exe
            .run_cached_direct(&[&runner.w13_in], &[&runner.w13_out])
            .expect("baseline w3 run");
        {
            let locked = runner.w13_out.as_f32_slice();
            runner.h3.copy_from_slice(&locked[..cfg.hidden * seq]);
        }
    }

    silu::silu_gate(&runner.h1, &runner.h3, &mut runner.gate);
    {
        let mut locked = runner.w2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, cfg.hidden, w2_sp, &runner.gate, seq, 0);
        stage_transposed_weight_columns(
            buf,
            &layer_weights.w2,
            dim,
            cfg.hidden,
            0,
            cfg.hidden,
            w2_sp,
            seq,
        );
    }
    runner
        .w2_exe
        .run_cached_direct(&[&runner.w2_in], &[&runner.w2_out])
        .expect("baseline w2 run");
    {
        let locked = runner.w2_out.as_f32_slice();
        let len = scratch.ffn_out.len();
        scratch.ffn_out.copy_from_slice(&locked[..len]);
    }

    vdsp::vsma(&scratch.ffn_out, alpha, &scratch.x2, x_next);
    t0.elapsed().as_secs_f32() * 1000.0
}

fn run_sharded_ffn_into(
    runner: &mut ShardedFfnRunner,
    layer_weights: &LayerWeights,
    scratch: &mut LayerScratch,
    x_next: &mut [f32],
) -> f32 {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);
    let t0 = Instant::now();

    let barrier = Arc::new(Barrier::new(runner.workers.len() + 1));
    let x2norm_ref = &scratch.x2norm;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(runner.workers.len());
        for worker in &mut runner.workers {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                let shard_hidden = worker.layout.col_count;
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
                        worker.layout.col_start,
                        shard_hidden,
                        seq,
                    );
                    stage_weight_columns(
                        buf,
                        dim,
                        w13_sp,
                        &layer_weights.w3,
                        cfg.hidden,
                        worker.layout.col_start,
                        shard_hidden,
                        seq + shard_hidden,
                    );
                }
                worker
                    .w13_exe
                    .run_cached_direct(&[&worker.w13_in], &[&worker.w13_out])
                    .expect("sharded ffn w13 run");
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
                        worker.layout.col_start,
                        shard_hidden,
                        w2_sp,
                        seq,
                    );
                }
                worker
                    .w2_exe
                    .run_cached_direct(&[&worker.w2_in], &[&worker.w2_out])
                    .expect("sharded ffn w2 run");
            }));
        }

        barrier.wait();
        for handle in handles {
            handle.join().expect("ffn shard worker panicked");
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
    t0.elapsed().as_secs_f32() * 1000.0
}

fn run_mode_layer_forward_into(
    runner: &mut ModeRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    x_next: &mut [f32],
) -> (f32, f32) {
    let attn_ms = match &mut runner.attn {
        AttentionRunner::Baseline(attn) => {
            run_baseline_attention_into(attn, layer_weights, x, &mut runner.scratch)
        }
        AttentionRunner::Sharded(attn) => {
            run_sharded_attention_into(attn, layer_weights, x, &mut runner.scratch)
        }
    };
    let ffn_ms = match &mut runner.ffn {
        FfnRunner::Baseline(ffn) => {
            run_baseline_ffn_into(ffn, layer_weights, &mut runner.scratch, x_next)
        }
        FfnRunner::Sharded(ffn) => {
            run_sharded_ffn_into(ffn, layer_weights, &mut runner.scratch, x_next)
        }
    };
    (attn_ms, ffn_ms)
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

fn run_mode_full_forward_once(
    cfg: &ModelConfig,
    runner: &mut ModeRunner,
    ws: &mut ModelForwardWorkspace,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> FullModelOutputs {
    let dim = cfg.dim;
    let seq = cfg.seq;

    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    let mut total_attn_ms = 0.0f32;
    let mut total_ffn_ms = 0.0f32;
    for layer_weights in &weights.layers {
        let x_buf = ws.x_buf.clone();
        let (attn_ms, ffn_ms) =
            run_mode_layer_forward_into(runner, layer_weights, &x_buf, &mut ws.x_next_buf);
        total_attn_ms += attn_ms;
        total_ffn_ms += ffn_ms;
        std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
    }

    let (logits, loss) = finalize_logits_and_loss(cfg, weights, ws, targets, softcap);
    FullModelOutputs {
        logits,
        loss,
        total_attn_ms,
        total_ffn_ms,
    }
}

fn run_baseline_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
) -> (ModeResult, Vec<f32>, f32) {
    let softcap = TrainConfig::default().softcap;
    let baseline_spec = ModeSpec::baseline();
    let (mut runner, compile_s) = compile_mode_runner(cfg, &baseline_spec);
    let mut ws = ModelForwardWorkspace::new_lean(cfg);
    let mut peak_rss_mb = rss_mb().unwrap_or(0.0);

    let t0 = Instant::now();
    let warmup =
        run_mode_full_forward_once(cfg, &mut runner, &mut ws, weights, tokens, targets, softcap);
    let warmup_wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let baseline_logits = warmup.logits.clone();
    let baseline_loss = warmup.loss;
    peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));

    let mut timed_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut attn_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut ffn_samples_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let out = run_mode_full_forward_once(
            cfg,
            &mut runner,
            &mut ws,
            weights,
            tokens,
            targets,
            softcap,
        );
        let diff = compare_logits_and_loss(&out.logits, out.loss, &baseline_logits, baseline_loss);
        assert!(
            diff.passes(),
            "baseline measured sample diverged from baseline warmup: {:?}",
            diff
        );
        timed_samples_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
        attn_samples_ms.push(out.total_attn_ms);
        ffn_samples_ms.push(out.total_ffn_ms);
        peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));
    }

    let mean_full_forward_ms = mean(&timed_samples_ms);
    let total_attn_ms = mean(&attn_samples_ms);
    let total_ffn_ms = mean(&ffn_samples_ms);
    let total_other_ms = (mean_full_forward_ms - total_attn_ms - total_ffn_ms).max(0.0);
    let tok_per_s = cfg.seq as f32 * 1000.0 / mean_full_forward_ms;
    let attn_pct_of_wall = if mean_full_forward_ms > 0.0 {
        total_attn_ms / mean_full_forward_ms * 100.0
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
            mode: baseline_spec.name,
            attn_shards: 1,
            ffn_shards: 1,
            compile_s,
            warmup_wall_ms,
            timed_samples_ms,
            mean_full_forward_ms,
            tok_per_s,
            peak_rss_mb,
            total_attn_ms,
            total_ffn_ms,
            total_other_ms,
            attn_pct_of_wall,
            ffn_pct_of_wall,
            diff_vs_baseline: zero_diff_metrics(),
            correctness_pass: true,
            full_forward_win_pct_vs_baseline: 0.0,
            meets_perf_gate: false,
        },
        baseline_logits,
        baseline_loss,
    )
}

fn run_candidate_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    mode: &ModeSpec,
    baseline_logits: &[f32],
    baseline_loss: f32,
    baseline_mean_ms: f32,
) -> ModeResult {
    let softcap = TrainConfig::default().softcap;
    let (mut runner, compile_s) = compile_mode_runner(cfg, mode);
    let mut ws = ModelForwardWorkspace::new_lean(cfg);
    let mut peak_rss_mb = rss_mb().unwrap_or(0.0);

    let t0 = Instant::now();
    let warmup =
        run_mode_full_forward_once(cfg, &mut runner, &mut ws, weights, tokens, targets, softcap);
    let warmup_wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let diff = compare_logits_and_loss(&warmup.logits, warmup.loss, baseline_logits, baseline_loss);
    peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));

    let mut timed_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut attn_samples_ms = Vec::with_capacity(TIMED_RUNS);
    let mut ffn_samples_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let out = run_mode_full_forward_once(
            cfg,
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
            "measured sample diverged from baseline for mode {}: {:?}",
            mode.name,
            sample_diff
        );
        timed_samples_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
        attn_samples_ms.push(out.total_attn_ms);
        ffn_samples_ms.push(out.total_ffn_ms);
        peak_rss_mb = peak_rss_mb.max(rss_mb().unwrap_or(peak_rss_mb));
    }

    let mean_full_forward_ms = mean(&timed_samples_ms);
    let total_attn_ms = mean(&attn_samples_ms);
    let total_ffn_ms = mean(&ffn_samples_ms);
    let total_other_ms = (mean_full_forward_ms - total_attn_ms - total_ffn_ms).max(0.0);
    let tok_per_s = cfg.seq as f32 * 1000.0 / mean_full_forward_ms;
    let attn_pct_of_wall = if mean_full_forward_ms > 0.0 {
        total_attn_ms / mean_full_forward_ms * 100.0
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
        mode: mode.name.clone(),
        attn_shards: mode.attn_shards,
        ffn_shards: mode.ffn_shards,
        compile_s,
        warmup_wall_ms,
        timed_samples_ms,
        mean_full_forward_ms,
        tok_per_s,
        peak_rss_mb,
        total_attn_ms,
        total_ffn_ms,
        total_other_ms,
        attn_pct_of_wall,
        ffn_pct_of_wall,
        diff_vs_baseline: diff.clone(),
        correctness_pass: diff.passes(),
        full_forward_win_pct_vs_baseline,
        meets_perf_gate: diff.passes() && full_forward_win_pct_vs_baseline >= PERF_WIN_PCT_TOL,
    }
}

fn best_mode_name<'a>(modes: impl Iterator<Item = &'a ModeResult>) -> Option<String> {
    modes
        .min_by(|a, b| {
            a.mean_full_forward_ms
                .partial_cmp(&b.mean_full_forward_ms)
                .unwrap()
        })
        .map(|mode| mode.mode.clone())
}

fn run_experiment(cfg: &ModelConfig, config_name: &str) -> ExperimentResult {
    let weights = ModelWeights::random(cfg);
    let (tokens, targets) = deterministic_tokens(cfg);
    let (baseline, baseline_logits, baseline_loss) =
        run_baseline_mode(cfg, &weights, &tokens, &targets);

    let modes = mode_specs()
        .into_iter()
        .skip(1)
        .map(|mode| {
            run_candidate_mode(
                cfg,
                &weights,
                &tokens,
                &targets,
                &mode,
                &baseline_logits,
                baseline_loss,
                baseline.mean_full_forward_ms,
            )
        })
        .collect::<Vec<_>>();

    let winning_modes = modes
        .iter()
        .filter_map(|mode| mode.meets_perf_gate.then_some(mode.mode.clone()))
        .collect::<Vec<_>>();

    let best_attention_only_mode = best_mode_name(
        modes
            .iter()
            .filter(|mode| mode.attn_shards > 1 && mode.ffn_shards == 1),
    );
    let best_ffn_only_mode = best_mode_name(
        modes
            .iter()
            .filter(|mode| mode.attn_shards == 1 && mode.ffn_shards > 1),
    );
    let best_combined_mode = best_mode_name(
        modes
            .iter()
            .filter(|mode| mode.attn_shards > 1 && mode.ffn_shards > 1),
    );

    ExperimentResult {
        config_name: config_name.to_string(),
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
        baseline,
        modes,
        best_attention_only_mode,
        best_ffn_only_mode,
        best_combined_mode,
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

fn write_summary(result: &ExperimentResult, path: &Path) {
    let mut summary = String::new();
    summary.push_str("# Full-Model Attention+FFN Latency Parallel Comparison\n\n");
    summary.push_str(&format!(
        "Config: `{}` — {}d/{}h/{}L/seq{}\n\n",
        result.config_name, result.dim, result.hidden, result.nlayers, result.seq
    ));

    summary.push_str("| mode | attn shards | ffn shards | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | total attn(ms) | total ffn(ms) | total other(ms) | attn % | ffn % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |\n");
    summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    for mode in std::iter::once(&result.baseline).chain(result.modes.iter()) {
        let diff = &mode.diff_vs_baseline;
        let _ = writeln!(
            summary,
            "| {} | {} | {} | {:.2} | {:.1} | {:.1} | {:.2} | {:.0} | {:.1} | {:.1} | {:.1} | {:.1}% | {:.1}% | {:.4} | {:.4} | {:.6} | {:.6} | {:+.2}% |",
            mode.mode,
            mode.attn_shards,
            mode.ffn_shards,
            mode.compile_s,
            mode.warmup_wall_ms,
            mode.mean_full_forward_ms,
            mode.tok_per_s,
            mode.peak_rss_mb,
            mode.total_attn_ms,
            mode.total_ffn_ms,
            mode.total_other_ms,
            mode.attn_pct_of_wall,
            mode.ffn_pct_of_wall,
            diff.max_abs_diff,
            diff.mean_abs_diff,
            diff.cosine_similarity,
            diff.loss_abs_diff,
            mode.full_forward_win_pct_vs_baseline,
        );
    }

    summary.push_str("\n## Best Modes\n\n");
    summary.push_str(&format!(
        "- Best attention-only: `{}`\n",
        result.best_attention_only_mode.as_deref().unwrap_or("n/a"),
    ));
    summary.push_str(&format!(
        "- Best FFN-only: `{}`\n",
        result.best_ffn_only_mode.as_deref().unwrap_or("n/a"),
    ));
    summary.push_str(&format!(
        "- Best combined: `{}`\n\n",
        result.best_combined_mode.as_deref().unwrap_or("n/a"),
    ));

    summary.push_str("## Thresholds\n\n");
    summary.push_str(&format!(
        "- max abs diff <= `{:.3e}`\n- mean abs diff <= `{:.3e}`\n- cosine similarity >= `{:.3}`\n- loss abs diff <= `{:.3e}`\n- full forward win >= `{:.1}%`\n\n",
        result.thresholds.max_abs_diff_tol,
        result.thresholds.mean_abs_diff_tol,
        result.thresholds.cosine_similarity_tol,
        result.thresholds.loss_abs_diff_tol,
        result.thresholds.perf_win_pct_tol,
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
    summary.push_str("- Each mode runs 1 warmup plus 2 measured full-model passes.\n");
    summary.push_str("- Attention totals include RMSNorm1, SDPA, WO, residual, and RMSNorm2.\n");
    summary.push_str("- ANE hardware-time reporting is intentionally omitted from this harness.\n");

    ensure_results_dir();
    fs::write(path, summary).expect("write summary");
}

#[test]
#[ignore]
fn attn_ffn_full_model_smoke_30bgeom_16l_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_30bgeom_16l(), "30Bgeom-16L");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed: {:?}",
            mode.mode, mode.diff_vs_baseline
        );
    }
}

#[test]
#[ignore]
fn bench_attn_ffn_full_model_30bgeom_16l_matrix() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_30bgeom_16l(), "30Bgeom-16L");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed",
            mode.mode
        );
    }
    write_json(&json_path("30bgeom_16l"), &result);
    write_summary(&result, &summary_path("30bgeom_16l"));
}

#[test]
#[ignore]
fn attn_ffn_full_model_smoke_30b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_30b(), "30B");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed: {:?}",
            mode.mode, mode.diff_vs_baseline
        );
    }
}

#[test]
#[ignore]
fn bench_attn_ffn_full_model_30b_matrix() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_30b(), "30B");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed",
            mode.mode
        );
    }
    write_json(&json_path("30b"), &result);
    write_summary(&result, &summary_path("30b"));
}

#[test]
#[ignore]
fn attn_ffn_full_model_smoke_50b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_50b(), "50B");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed: {:?}",
            mode.mode, mode.diff_vs_baseline
        );
    }
}

#[test]
#[ignore]
fn bench_attn_ffn_full_model_50b_matrix() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_50b(), "50B");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed",
            mode.mode
        );
    }
    write_json(&json_path("50b"), &result);
    write_summary(&result, &summary_path("50b"));
}

#[test]
#[ignore]
fn attn_ffn_full_model_smoke_80b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_80b(), "80B");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed: {:?}",
            mode.mode, mode.diff_vs_baseline
        );
    }
}

#[test]
#[ignore]
fn bench_attn_ffn_full_model_80b_matrix() {
    let _guard = run_lock().lock().unwrap();
    let result = run_experiment(&cfg_80b(), "80B");
    assert!(
        result.baseline.correctness_pass,
        "baseline correctness failed"
    );
    for mode in &result.modes {
        assert!(
            mode.correctness_pass,
            "mode {} correctness failed",
            mode.mode
        );
    }
    write_json(&json_path("80b"), &result);
    write_summary(&result, &summary_path("80b"));
}
