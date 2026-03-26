//! Benchmark-only FFN latency parallel experiment.
//!
//! Run build check:
//!   cargo test -p engine --test bench_ffn_latency_parallel --no-run
//!
//! Run smoke validation:
//!   cargo test -p engine --test bench_ffn_latency_parallel --release -- --ignored --nocapture ffn_shard_smoke_5b_matches_baseline
//!
//! Run scale suite benchmark:
//!   cargo test -p engine --test bench_ffn_latency_parallel --release -- --ignored --nocapture bench_ffn_shard_latency_scales

use ane_bridge::ane::{Executable, TensorData};
use engine::cpu::{rmsnorm, silu, vdsp};
use engine::kernels::{dyn_matmul, sdpa_fwd};
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

const WARMUP_RUNS: usize = 1;
const TIMED_RUNS: usize = 5;
const MAX_ABS_DIFF_TOL: f32 = 2e-2;
const MEAN_ABS_DIFF_TOL: f32 = 5e-3;
const COSINE_SIM_TOL: f32 = 0.999;
const PERF_WIN_PCT_TOL: f32 = 5.0;

fn run_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/latency_parallel_ffn")
}

fn json_path() -> PathBuf {
    results_dir().join("scale_shard_compare.json")
}

fn summary_path() -> PathBuf {
    results_dir().join("summary.md")
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

fn cfg_5b() -> ModelConfig {
    custom_config(3072, 8192, 24, 44, 512)
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

fn cfg_100b() -> ModelConfig {
    custom_config(5120, 13824, 40, 320, 512)
}

fn scale_configs() -> Vec<(ModelConfig, &'static str)> {
    vec![
        (cfg_30b(), "30B"),
        (cfg_50b(), "50B"),
        (cfg_80b(), "80B"),
        (cfg_100b(), "100B"),
    ]
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i.wrapping_mul(17).wrapping_add(11)) % 251) as f32 * 0.001 - 0.125)
        .collect()
}

fn percentile(values: &[f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f32 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn median(values: &[f32]) -> f32 {
    percentile(values, 50.0)
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

fn slice_weight_columns(
    src: &[f32],
    rows: usize,
    cols: usize,
    col_start: usize,
    col_count: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * col_count];
    for r in 0..rows {
        let src_row = r * cols + col_start;
        let dst_row = r * col_count;
        out[dst_row..dst_row + col_count].copy_from_slice(&src[src_row..src_row + col_count]);
    }
    out
}

fn compare_outputs(actual: &[f32], expected: &[f32]) -> DiffMetrics {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&a, &b) in actual.iter().zip(expected.iter()) {
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
        mean_abs_diff: sum_abs / actual.len() as f32,
        cosine_similarity,
    }
}

#[derive(Debug, Clone, Serialize)]
struct DiffMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    cosine_similarity: f32,
}

impl DiffMetrics {
    fn passes(&self) -> bool {
        self.max_abs_diff <= MAX_ABS_DIFF_TOL
            && self.mean_abs_diff <= MEAN_ABS_DIFF_TOL
            && self.cosine_similarity >= COSINE_SIM_TOL
    }
}

#[derive(Debug, Clone, Serialize)]
struct Thresholds {
    max_abs_diff_tol: f32,
    mean_abs_diff_tol: f32,
    cosine_similarity_tol: f32,
    perf_win_pct_tol: f32,
}

#[derive(Debug, Clone, Serialize)]
struct RunSample {
    full_layer_ms: f32,
    ffn_only_ms: f32,
    w13_stage_ms: f32,
    w13_ane_ms: f32,
    w13_read_ms: f32,
    gate_cpu_ms: f32,
    w2_stage_ms: f32,
    w2_ane_ms: f32,
    w2_read_ms: f32,
    merge_cpu_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ModeResult {
    mode: String,
    shard_count: usize,
    compile_s: f32,
    samples: Vec<RunSample>,
    median_full_layer_ms: f32,
    median_ffn_only_ms: f32,
    max_rss_mb: f32,
    diff_vs_baseline: Option<DiffMetrics>,
    correctness_pass: bool,
    full_layer_win_pct_vs_baseline: Option<f32>,
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
    primary_success: bool,
    winning_modes: Vec<String>,
    thresholds: Thresholds,
}

#[derive(Debug, Clone, Serialize)]
struct ExperimentSuiteResult {
    experiments: Vec<ExperimentResult>,
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

struct ShardWeights {
    w1: Vec<f32>,
    w3: Vec<f32>,
    w2: Vec<f32>, // [dim, shard_hidden]
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
    w2t: Vec<f32>,
    partial_out: Vec<f32>,
}

struct ShardedFfnRunner {
    cfg: ModelConfig,
    shard_count: usize,
    sdpa: SdpaRunner,
    wo: WoRunner,
    workers: Vec<ShardWorker>,
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
            w2t: vec![0.0; shard_hidden * cfg.dim],
            partial_out: vec![0.0; cfg.dim * cfg.seq],
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
        },
        compile_s,
    )
}

fn shard_weights(
    weights: &LayerWeights,
    cfg: &ModelConfig,
    shard_count: usize,
) -> Vec<ShardWeights> {
    let shard_hidden = cfg.hidden / shard_count;
    (0..shard_count)
        .map(|i| {
            let start = i * shard_hidden;
            ShardWeights {
                w1: slice_weight_columns(&weights.w1, cfg.dim, cfg.hidden, start, shard_hidden),
                w3: slice_weight_columns(&weights.w3, cfg.dim, cfg.hidden, start, shard_hidden),
                w2: slice_weight_columns(&weights.w2, cfg.dim, cfg.hidden, start, shard_hidden),
            }
        })
        .collect()
}

fn run_sdpa(sdpa: &mut SdpaRunner, xnorm: &[f32], weights: &LayerWeights, attn_out: &mut [f32]) {
    sdpa.xnorm_in.copy_from_f32(xnorm);
    sdpa.wq_in.copy_from_f32(&weights.wq);
    sdpa.wk_in.copy_from_f32(&weights.wk);
    sdpa.wv_in.copy_from_f32(&weights.wv);
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
    let locked = sdpa.attn_out.as_f32_slice();
    attn_out.copy_from_slice(&locked[..attn_out.len()]);
}

fn run_wo(
    wo: &mut WoRunner,
    cfg: &ModelConfig,
    attn_out: &[f32],
    weights: &LayerWeights,
    o_out: &mut [f32],
) {
    let sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    {
        let mut locked = wo.input.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, cfg.q_dim, sp, attn_out, cfg.seq, 0);
        stage_spatial(buf, cfg.q_dim, sp, &weights.wo, cfg.dim, cfg.seq);
    }
    wo.exe
        .run_cached_direct(&[&wo.input], &[&wo.output])
        .expect("wo run");
    let locked = wo.output.as_f32_slice();
    o_out.copy_from_slice(&locked[..o_out.len()]);
}

fn run_sharded_layer_forward(
    runner: &mut ShardedFfnRunner,
    weights: &LayerWeights,
    shard_weights: &[ShardWeights],
    x: &[f32],
) -> (Vec<f32>, RunSample) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    let t_total = Instant::now();

    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(x, &weights.gamma1, &mut xnorm, &mut rms_inv1, dim, seq);

    let mut attn_out = vec![0.0f32; cfg.q_dim * seq];
    run_sdpa(&mut runner.sdpa, &xnorm, weights, &mut attn_out);

    let mut o_out = vec![0.0f32; dim * seq];
    run_wo(&mut runner.wo, cfg, &attn_out, weights, &mut o_out);

    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(&x2, &weights.gamma2, &mut x2norm, &mut rms_inv2, dim, seq);

    let ffn_t0 = Instant::now();
    let barrier = Arc::new(Barrier::new(runner.shard_count + 1));
    let mut shard_samples = Vec::with_capacity(runner.shard_count);
    let x2norm_ref: &[f32] = &x2norm;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(runner.shard_count);
        for (worker, shard_w) in runner.workers.iter_mut().zip(shard_weights.iter()) {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                let shard_hidden = shard_w.w1.len() / dim;
                let w13_sp = dyn_matmul::dual_separate_spatial_width(seq, shard_hidden);
                let w2_sp = dyn_matmul::spatial_width(seq, dim);

                barrier.wait();

                let t = Instant::now();
                {
                    let mut locked = worker.w13_in.as_f32_slice_mut();
                    let buf = &mut *locked;
                    stage_spatial(buf, dim, w13_sp, x2norm_ref, seq, 0);
                    stage_spatial(buf, dim, w13_sp, &shard_w.w1, shard_hidden, seq);
                    stage_spatial(
                        buf,
                        dim,
                        w13_sp,
                        &shard_w.w3,
                        shard_hidden,
                        seq + shard_hidden,
                    );
                }
                let w13_stage_ms = t.elapsed().as_secs_f32() * 1000.0;

                let t = Instant::now();
                worker
                    .w13_exe
                    .run_cached_direct(&[&worker.w13_in], &[&worker.w13_out])
                    .expect("w13 shard run");
                let w13_ane_ms = t.elapsed().as_secs_f32() * 1000.0;

                let t = Instant::now();
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
                let w13_read_ms = t.elapsed().as_secs_f32() * 1000.0;

                let t = Instant::now();
                silu::silu_gate(&worker.h1, &worker.h3, &mut worker.gate);
                let gate_cpu_ms = t.elapsed().as_secs_f32() * 1000.0;

                let t = Instant::now();
                vdsp::mtrans(
                    &shard_w.w2,
                    shard_hidden,
                    &mut worker.w2t,
                    dim,
                    dim,
                    shard_hidden,
                );
                {
                    let mut locked = worker.w2_in.as_f32_slice_mut();
                    let buf = &mut *locked;
                    stage_spatial(buf, shard_hidden, w2_sp, &worker.gate, seq, 0);
                    stage_spatial(buf, shard_hidden, w2_sp, &worker.w2t, dim, seq);
                }
                let w2_stage_ms = t.elapsed().as_secs_f32() * 1000.0;

                let t = Instant::now();
                worker
                    .w2_exe
                    .run_cached_direct(&[&worker.w2_in], &[&worker.w2_out])
                    .expect("w2 shard run");
                let w2_ane_ms = t.elapsed().as_secs_f32() * 1000.0;

                let t = Instant::now();
                {
                    let locked = worker.w2_out.as_f32_slice();
                    let len = worker.partial_out.len();
                    worker.partial_out.copy_from_slice(&locked[..len]);
                }
                let w2_read_ms = t.elapsed().as_secs_f32() * 1000.0;

                (
                    w13_stage_ms,
                    w13_ane_ms,
                    w13_read_ms,
                    gate_cpu_ms,
                    w2_stage_ms,
                    w2_ane_ms,
                    w2_read_ms,
                )
            }));
        }

        barrier.wait();
        for handle in handles {
            shard_samples.push(handle.join().expect("shard worker panicked"));
        }
    });

    let t = Instant::now();
    let mut ffn_out = vec![0.0f32; dim * seq];
    for worker in &runner.workers {
        // vDSP wrapper does not support in-place add aliasing safely; use a temp.
        let mut tmp = vec![0.0f32; dim * seq];
        vdsp::vadd(&ffn_out, &worker.partial_out, &mut tmp);
        ffn_out = tmp;
    }
    let mut x_next = vec![0.0f32; dim * seq];
    vdsp::vsma(&ffn_out, alpha, &x2, &mut x_next);
    let merge_cpu_ms = t.elapsed().as_secs_f32() * 1000.0;

    let ffn_only_ms = ffn_t0.elapsed().as_secs_f32() * 1000.0;
    let full_layer_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let w13_stage_ms = shard_samples.iter().map(|s| s.0).fold(0.0f32, f32::max);
    let w13_ane_ms = shard_samples.iter().map(|s| s.1).fold(0.0f32, f32::max);
    let w13_read_ms = shard_samples.iter().map(|s| s.2).fold(0.0f32, f32::max);
    let gate_cpu_ms = shard_samples.iter().map(|s| s.3).fold(0.0f32, f32::max);
    let w2_stage_ms = shard_samples.iter().map(|s| s.4).fold(0.0f32, f32::max);
    let w2_ane_ms = shard_samples.iter().map(|s| s.5).fold(0.0f32, f32::max);
    let w2_read_ms = shard_samples.iter().map(|s| s.6).fold(0.0f32, f32::max);

    (
        x_next,
        RunSample {
            full_layer_ms,
            ffn_only_ms,
            w13_stage_ms,
            w13_ane_ms,
            w13_read_ms,
            gate_cpu_ms,
            w2_stage_ms,
            w2_ane_ms,
            w2_read_ms,
            merge_cpu_ms,
        },
    )
}

fn run_baseline_mode(
    cfg: &ModelConfig,
    weights: &LayerWeights,
    x: &[f32],
) -> (ModeResult, Vec<f32>) {
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(cfg);
    let compile_s = t0.elapsed().as_secs_f32();

    let mut samples = Vec::with_capacity(TIMED_RUNS);
    let mut max_rss = rss_mb().unwrap_or(0.0);

    let mut baseline_x_next = Vec::new();
    for i in 0..WARMUP_RUNS {
        let (x_next, _, _) = layer::forward_timed(cfg, &kernels, weights, x);
        if i + 1 == WARMUP_RUNS {
            baseline_x_next = x_next;
        }
        max_rss = max_rss.max(rss_mb().unwrap_or(max_rss));
    }

    for _ in 0..TIMED_RUNS {
        let (_, _, timings) = layer::forward_timed(cfg, &kernels, weights, x);
        samples.push(RunSample {
            full_layer_ms: timings.total_ms,
            ffn_only_ms: timings.stage_ffn_ms + timings.ane_ffn_ms + timings.read_ffn_ms,
            w13_stage_ms: timings.stage_ffn_ms,
            w13_ane_ms: timings.ane_ffn_ms,
            w13_read_ms: timings.read_ffn_ms,
            gate_cpu_ms: 0.0,
            w2_stage_ms: 0.0,
            w2_ane_ms: 0.0,
            w2_read_ms: 0.0,
            merge_cpu_ms: 0.0,
        });
        max_rss = max_rss.max(rss_mb().unwrap_or(max_rss));
    }

    let full_layer: Vec<f32> = samples.iter().map(|s| s.full_layer_ms).collect();
    let ffn_only: Vec<f32> = samples.iter().map(|s| s.ffn_only_ms).collect();
    let baseline_diff = compare_outputs(&baseline_x_next, &baseline_x_next);

    (
        ModeResult {
            mode: "baseline".to_string(),
            shard_count: 1,
            compile_s,
            samples,
            median_full_layer_ms: median(&full_layer),
            median_ffn_only_ms: median(&ffn_only),
            max_rss_mb: max_rss,
            diff_vs_baseline: Some(baseline_diff),
            correctness_pass: true,
            full_layer_win_pct_vs_baseline: Some(0.0),
            meets_perf_gate: false,
            bucket_note: Some(
                "Baseline uses coarse FFN timer buckets from layer::forward_timed; W13/W2/gate/merge are not separately exposed."
                    .to_string(),
            ),
        },
        baseline_x_next,
    )
}

fn run_sharded_mode(
    cfg: &ModelConfig,
    weights: &LayerWeights,
    x: &[f32],
    shard_count: usize,
    baseline_x_next: &[f32],
    baseline_full_layer_median: f32,
) -> ModeResult {
    let (mut runner, compile_s) = compile_sharded_runner(cfg, shard_count);
    let shard_weights = shard_weights(weights, cfg, shard_count);

    let mut max_rss = rss_mb().unwrap_or(0.0);
    let mut warmup_x_next = Vec::new();
    for i in 0..WARMUP_RUNS {
        let (x_next, _) = run_sharded_layer_forward(&mut runner, weights, &shard_weights, x);
        if i + 1 == WARMUP_RUNS {
            warmup_x_next = x_next;
        }
        max_rss = max_rss.max(rss_mb().unwrap_or(max_rss));
    }
    let diff = compare_outputs(&warmup_x_next, baseline_x_next);
    max_rss = max_rss.max(rss_mb().unwrap_or(max_rss));

    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let (_, sample) = run_sharded_layer_forward(&mut runner, weights, &shard_weights, x);
        samples.push(sample);
        max_rss = max_rss.max(rss_mb().unwrap_or(max_rss));
    }

    let full_layer: Vec<f32> = samples.iter().map(|s| s.full_layer_ms).collect();
    let ffn_only: Vec<f32> = samples.iter().map(|s| s.ffn_only_ms).collect();
    let median_full = median(&full_layer);
    let median_ffn = median(&ffn_only);
    let win_pct = (baseline_full_layer_median - median_full) / baseline_full_layer_median * 100.0;

    ModeResult {
        mode: format!("shard{shard_count}"),
        shard_count,
        compile_s,
        samples,
        median_full_layer_ms: median_full,
        median_ffn_only_ms: median_ffn,
        max_rss_mb: max_rss,
        diff_vs_baseline: Some(diff.clone()),
        correctness_pass: diff.passes(),
        full_layer_win_pct_vs_baseline: Some(win_pct),
        meets_perf_gate: diff.passes() && win_pct >= PERF_WIN_PCT_TOL,
        bucket_note: Some(
            "Shard buckets are recorded as max per-shard phase time; FFN-only wall time is end-to-end outer shard section wall."
                .to_string(),
        ),
    }
}

fn write_summary(suite: &ExperimentSuiteResult) {
    let mut summary = String::new();
    summary.push_str("# FFN Latency Parallel Scale Comparison\n\n");

    if let Some(first) = suite.experiments.first() {
        summary.push_str("## Thresholds\n\n");
        summary.push_str(&format!(
            "- max abs diff <= `{:.3e}`\n- mean abs diff <= `{:.3e}`\n- cosine similarity >= `{:.3}`\n- full-layer win >= `{:.1}%`\n\n",
            first.thresholds.max_abs_diff_tol,
            first.thresholds.mean_abs_diff_tol,
            first.thresholds.cosine_similarity_tol,
            first.thresholds.perf_win_pct_tol
        ));
    }

    summary.push_str("## Best Mode Per Scale\n\n");
    summary.push_str("| scale | baseline(ms) | best mode | best full-layer(ms) | win vs baseline | primary success |\n");
    summary.push_str("| --- | ---: | --- | ---: | ---: | --- |\n");
    for result in &suite.experiments {
        let candidates = [
            &result.shard2,
            &result.shard4,
            &result.shard6,
            &result.shard8,
        ];
        let best = candidates
            .iter()
            .min_by(|a, b| {
                a.median_full_layer_ms
                    .partial_cmp(&b.median_full_layer_ms)
                    .unwrap()
            })
            .unwrap();
        let _ = writeln!(
            summary,
            "| {} | {:.1} | {} | {:.1} | {:+.2}% | {} |",
            result.config_name,
            result.baseline.median_full_layer_ms,
            best.mode,
            best.median_full_layer_ms,
            best.full_layer_win_pct_vs_baseline.unwrap_or(0.0),
            if result.primary_success {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }

    for result in &suite.experiments {
        summary.push_str(&format!(
            "\n## {} — {}d/{}h/{}L/seq{}\n\n",
            result.config_name, result.dim, result.hidden, result.nlayers, result.seq
        ));
        summary.push_str("| mode | compile(s) | median full-layer(ms) | median ffn(ms) | peak RSS(MB) | max abs diff | mean abs diff | cosine | win vs baseline |\n");
        summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for mode in [
            &result.baseline,
            &result.shard2,
            &result.shard4,
            &result.shard6,
            &result.shard8,
        ] {
            let diff = mode.diff_vs_baseline.as_ref().unwrap();
            let win = mode.full_layer_win_pct_vs_baseline.unwrap_or(0.0);
            let _ = writeln!(
                summary,
                "| {} | {:.2} | {:.1} | {:.1} | {:.0} | {:.4} | {:.4} | {:.6} | {:+.2}% |",
                mode.mode,
                mode.compile_s,
                mode.median_full_layer_ms,
                mode.median_ffn_only_ms,
                mode.max_rss_mb,
                diff.max_abs_diff,
                diff.mean_abs_diff,
                diff.cosine_similarity,
                win
            );
        }

        summary.push_str(&format!(
            "\nPrimary success: **{}**\n\n",
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
    }

    summary.push_str("## Notes\n\n");
    summary.push_str("- Baseline full-layer timing comes from `layer::forward_timed(...)`.\n");
    summary.push_str("- Baseline FFN buckets are coarse runtime buckets; sharded buckets are max-per-shard phase times plus merge wall.\n");

    ensure_results_dir();
    fs::write(summary_path(), summary).expect("write summary");
}

fn run_experiment(cfg: &ModelConfig, config_name: &str) -> ExperimentResult {
    let weights = LayerWeights::random(cfg);
    let x = deterministic_input(cfg.dim * cfg.seq);

    let (baseline, baseline_x_next) = run_baseline_mode(cfg, &weights, &x);
    let shard2 = run_sharded_mode(
        cfg,
        &weights,
        &x,
        2,
        &baseline_x_next,
        baseline.median_full_layer_ms,
    );
    let shard4 = run_sharded_mode(
        cfg,
        &weights,
        &x,
        4,
        &baseline_x_next,
        baseline.median_full_layer_ms,
    );
    let shard6 = run_sharded_mode(
        cfg,
        &weights,
        &x,
        6,
        &baseline_x_next,
        baseline.median_full_layer_ms,
    );
    let shard8 = run_sharded_mode(
        cfg,
        &weights,
        &x,
        8,
        &baseline_x_next,
        baseline.median_full_layer_ms,
    );

    let winning_modes = [
        (&shard2, "shard2"),
        (&shard4, "shard4"),
        (&shard6, "shard6"),
        (&shard8, "shard8"),
    ]
    .into_iter()
    .filter_map(|(mode, name)| mode.meets_perf_gate.then_some(name.to_string()))
    .collect::<Vec<_>>();

    ExperimentResult {
        config_name: config_name.to_string(),
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: cfg.heads,
        nlayers: cfg.nlayers,
        seq: cfg.seq,
        primary_success: !winning_modes.is_empty(),
        winning_modes,
        baseline,
        shard2,
        shard4,
        shard6,
        shard8,
        thresholds: Thresholds {
            max_abs_diff_tol: MAX_ABS_DIFF_TOL,
            mean_abs_diff_tol: MEAN_ABS_DIFF_TOL,
            cosine_similarity_tol: COSINE_SIM_TOL,
            perf_win_pct_tol: PERF_WIN_PCT_TOL,
        },
    }
}

#[test]
#[ignore]
fn ffn_shard_smoke_5b_matches_baseline() {
    let _guard = run_lock().lock().unwrap();
    let cfg = cfg_5b();
    let weights = LayerWeights::random(&cfg);
    let x = deterministic_input(cfg.dim * cfg.seq);

    let (_, baseline_x_next) = run_baseline_mode(&cfg, &weights, &x);
    let shard2 = run_sharded_mode(&cfg, &weights, &x, 2, &baseline_x_next, 1.0);
    let shard4 = run_sharded_mode(&cfg, &weights, &x, 4, &baseline_x_next, 1.0);
    let shard8 = run_sharded_mode(&cfg, &weights, &x, 8, &baseline_x_next, 1.0);

    assert!(
        shard2.correctness_pass,
        "5B shard2 correctness failed: {:?}",
        shard2.diff_vs_baseline
    );
    assert!(
        shard4.correctness_pass,
        "5B shard4 correctness failed: {:?}",
        shard4.diff_vs_baseline
    );
    assert!(
        shard8.correctness_pass,
        "5B shard8 correctness failed: {:?}",
        shard8.diff_vs_baseline
    );
}

#[test]
#[ignore]
fn bench_ffn_shard_latency_scales() {
    let _guard = run_lock().lock().unwrap();
    let experiments = scale_configs()
        .into_iter()
        .map(|(cfg, name)| {
            let result = run_experiment(&cfg, name);
            assert!(
                result.shard2.correctness_pass,
                "{} shard2 correctness failed",
                name
            );
            assert!(
                result.shard4.correctness_pass,
                "{} shard4 correctness failed",
                name
            );
            assert!(
                result.shard6.correctness_pass,
                "{} shard6 correctness failed",
                name
            );
            assert!(
                result.shard8.correctness_pass,
                "{} shard8 correctness failed",
                name
            );
            result
        })
        .collect::<Vec<_>>();

    let suite = ExperimentSuiteResult { experiments };
    write_json(&json_path(), &suite);
    write_summary(&suite);
}
