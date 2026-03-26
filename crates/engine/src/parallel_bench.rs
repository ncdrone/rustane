use crate::cpu::{cross_entropy, embedding, rmsnorm, silu, vdsp};
use crate::full_model::{self, ModelForwardWorkspace, ModelWeights, TrainingParallelOptions};
use crate::kernels::{dyn_matmul, ffn_fused, sdpa_fwd};
use crate::layer::{CompiledKernels, LayerWeights, tensor_data_new_logged};
use crate::model::ModelConfig;
use crate::sharding::{ResolvedBackwardMode, ResolvedForwardMode, ShardRequest, resolve_modes};
use ane_bridge::ane::{Error as AneError, Executable, TensorData};
use objc2_foundation::NSQualityOfService;
use std::fmt;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

pub const ATTN_ALLOWED_REQUESTS: &[usize] = &[1, 2, 4, 8, 10];
pub const FFN_ALLOWED_REQUESTS: &[usize] = &[2, 4, 6, 8, 10, 12, 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardPolicy {
    FailFast,
    AutoAdjustNearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelBenchRequest {
    pub attn_request: Option<usize>,
    pub ffn_request: Option<usize>,
    pub policy: ShardPolicy,
}

impl ParallelBenchRequest {
    pub fn from_env(policy: ShardPolicy) -> Result<Self, ParallelBenchError> {
        let request = ShardRequest::from_env().map_err(ParallelBenchError::new)?;
        Ok(Self {
            attn_request: request.attn_fwd_shards,
            ffn_request: request.ffn_fwd_shards,
            policy,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelMode {
    Baseline,
    Attention {
        shards: usize,
    },
    Ffn {
        shards: usize,
    },
    AttentionFfn {
        attn_shards: usize,
        ffn_shards: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedParallelBenchConfig {
    pub requested_attn: Option<usize>,
    pub requested_ffn: Option<usize>,
    pub applied_attn: usize,
    pub applied_ffn: usize,
    pub mode: ParallelMode,
    pub notes: Vec<String>,
}

impl ResolvedParallelBenchConfig {
    pub fn benchmark_suffix(&self) -> String {
        let label = self.mode_label();
        if label == "baseline" {
            String::new()
        } else {
            format!("_{label}")
        }
    }

    pub fn mode_label(&self) -> String {
        match self.mode {
            ParallelMode::Baseline => "baseline".to_string(),
            ParallelMode::Attention { shards } => format!("attn{shards}"),
            ParallelMode::Ffn { shards } => format!("ffn{shards}"),
            ParallelMode::AttentionFfn {
                attn_shards,
                ffn_shards,
            } => format!("attn{attn_shards}_ffn{ffn_shards}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelBenchError {
    message: String,
}

impl ParallelBenchError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ParallelBenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for ParallelBenchError {}

pub struct ParallelBenchRunner {
    cfg: ModelConfig,
    resolved: ResolvedParallelBenchConfig,
    use_lean_workspace: bool,
    kind: ParallelBenchRunnerKind,
}

enum ParallelBenchRunnerKind {
    Baseline {
        kernels: CompiledKernels,
    },
    Mode {
        kernels: CompiledKernels,
        mode: ResolvedForwardMode,
        precompiled_runtime: Option<crate::layer::CombinedForwardRuntime>,
    },
}

/// Retry a closure when the local ane path reports an IOSurface allocation
/// failure during compile-time model setup. Any other error is returned
/// immediately.
fn compile_with_iosurface_retry<T>(
    label: &str,
    max_attempts: usize,
    f: impl Fn() -> Result<T, AneError>,
) -> Result<T, AneError> {
    for attempt in 1..=max_attempts {
        match f() {
            Ok(result) => return Ok(result),
            Err(err) => {
                if is_iosurface_retryable(&err) {
                    if attempt == max_attempts {
                        eprintln!(
                            "      ✗ {label} failed after {max_attempts} attempts, giving up"
                        );
                        return Err(err);
                    }
                    eprintln!(
                        "      ⚠ {label}: IOSurface allocation failed (attempt {attempt}/{max_attempts}), retrying in 2s..."
                    );
                    thread::sleep(Duration::from_secs(2));
                } else {
                    return Err(err);
                }
            }
        }
    }
    unreachable!()
}

fn is_iosurface_retryable(err: &AneError) -> bool {
    matches!(err, AneError::SurfaceWrap) || err.to_string().contains("IOSurface")
}

impl ParallelBenchRunner {
    pub fn compile(
        cfg: &ModelConfig,
        request: ParallelBenchRequest,
        use_lean_workspace: bool,
    ) -> Result<(Self, f32), ParallelBenchError> {
        let resolved = resolve_parallel_bench_config(cfg, request)?;
        let (kind, compile_s) = match resolved.mode {
            ParallelMode::Baseline => {
                let t0 = Instant::now();
                let cfg_clone = cfg.clone();
                let kernels =
                    compile_with_iosurface_retry("ANE kernel compilation", 3, move || {
                        if use_lean_workspace {
                            CompiledKernels::try_compile_forward_only(&cfg_clone)
                        } else {
                            CompiledKernels::try_compile(&cfg_clone)
                        }
                    })
                    .map_err(|err| {
                        ParallelBenchError::new(format!("ANE kernel compilation: {err}"))
                    })?;
                (
                    ParallelBenchRunnerKind::Baseline { kernels },
                    t0.elapsed().as_secs_f32(),
                )
            }
            _ => {
                let t0 = Instant::now();
                let cfg_clone = cfg.clone();
                let resolved_mode = convert_parallel_mode(&resolved.mode);
                let kernels =
                    compile_with_iosurface_retry("ANE kernel compilation", 3, move || {
                        if use_lean_workspace {
                            CompiledKernels::try_compile_forward_only(&cfg_clone)
                        } else {
                            CompiledKernels::try_compile(&cfg_clone)
                        }
                    })
                    .map_err(|err| {
                        ParallelBenchError::new(format!("ANE sharded kernel compilation: {err}"))
                    })?;
                let runtime =
                    compile_with_iosurface_retry("ANE sharded kernel compilation", 3, move || {
                        crate::layer::CombinedForwardRuntime::compile(cfg, resolved_mode)
                    })
                    .map_err(|err| {
                        ParallelBenchError::new(format!("ANE sharded kernel compilation: {err}"))
                    })?;
                (
                    ParallelBenchRunnerKind::Mode {
                        kernels,
                        mode: resolved_mode,
                        precompiled_runtime: Some(runtime),
                    },
                    t0.elapsed().as_secs_f32(),
                )
            }
        };

        Ok((
            Self {
                cfg: cfg.clone(),
                resolved,
                use_lean_workspace,
                kind,
            },
            compile_s,
        ))
    }

    pub fn resolved(&self) -> &ResolvedParallelBenchConfig {
        &self.resolved
    }

    pub fn forward_loss(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        targets: &[u32],
        softcap: f32,
        ws: &mut ModelForwardWorkspace,
    ) -> f32 {
        match &mut self.kind {
            ParallelBenchRunnerKind::Baseline { kernels } => {
                if self.use_lean_workspace {
                    full_model::forward_only_ws(
                        &self.cfg, kernels, weights, tokens, targets, softcap, ws,
                    )
                } else {
                    full_model::forward_ws(
                        &self.cfg, kernels, weights, tokens, targets, softcap, ws,
                    )
                }
            }
            ParallelBenchRunnerKind::Mode {
                kernels,
                mode,
                precompiled_runtime,
            } => {
                if ws.combined_forward.as_ref().map(|runtime| runtime.mode()) != Some(*mode) {
                    ws.combined_forward = precompiled_runtime.take().or_else(|| {
                        Some(
                            crate::layer::CombinedForwardRuntime::compile(&self.cfg, *mode)
                                .expect("compile combined forward runtime"),
                        )
                    });
                }
                if self.use_lean_workspace {
                    full_model::forward_only_ws_with_options(
                        &self.cfg, kernels, weights, tokens, targets, softcap, ws, *mode,
                    )
                } else {
                    let options = TrainingParallelOptions {
                        forward_mode: *mode,
                        backward_mode: ResolvedBackwardMode::Baseline,
                        adjustments: Vec::new(),
                    };
                    full_model::forward_ws_with_options(
                        &self.cfg, kernels, weights, tokens, targets, softcap, ws, &options,
                    )
                }
            }
        }
    }
}

fn parse_env_count(
    var: &str,
    allowed: &[usize],
    allow_baseline_one: bool,
) -> Result<Option<usize>, ParallelBenchError> {
    let Some(raw) = std::env::var(var).ok() else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .map_err(|_| ParallelBenchError::new(format!("{var} must be an integer, got '{raw}'")))?;
    if !allow_baseline_one && value == 1 {
        return Err(ParallelBenchError::new(format!(
            "{var}=1 is invalid; omit {var} to run baseline"
        )));
    }
    if !allowed.contains(&value) {
        return Err(ParallelBenchError::new(format!(
            "{var}={value} is unsupported; allowed values are {}",
            join_usizes(allowed)
        )));
    }
    Ok(Some(value))
}

fn resolve_parallel_bench_config(
    cfg: &ModelConfig,
    request: ParallelBenchRequest,
) -> Result<ResolvedParallelBenchConfig, ParallelBenchError> {
    let policy = match request.policy {
        ShardPolicy::FailFast => crate::sharding::ShardPolicy::FailFast,
        ShardPolicy::AutoAdjustNearest => crate::sharding::ShardPolicy::AutoAdjustNearest,
    };
    let resolution = resolve_modes(
        cfg,
        ShardRequest::from_forward_requests(request.attn_request, request.ffn_request),
        policy,
    )
    .map_err(ParallelBenchError::new)?;
    let applied_attn = resolution.forward.attn_shards();
    let applied_ffn = resolution.forward.ffn_shards();
    let notes = resolution
        .adjustments
        .iter()
        .map(|adjustment| {
            format!(
                "requested {}={}, applied {} ({})",
                adjustment.axis, adjustment.requested, adjustment.applied, adjustment.reason
            )
        })
        .collect::<Vec<_>>();

    let mode = match (applied_attn > 1, applied_ffn > 1) {
        (false, false) => ParallelMode::Baseline,
        (true, false) => ParallelMode::Attention {
            shards: applied_attn,
        },
        (false, true) => ParallelMode::Ffn {
            shards: applied_ffn,
        },
        (true, true) => ParallelMode::AttentionFfn {
            attn_shards: applied_attn,
            ffn_shards: applied_ffn,
        },
    };

    Ok(ResolvedParallelBenchConfig {
        requested_attn: request.attn_request,
        requested_ffn: request.ffn_request,
        applied_attn,
        applied_ffn,
        mode,
        notes,
    })
}

fn convert_parallel_mode(mode: &ParallelMode) -> ResolvedForwardMode {
    match *mode {
        ParallelMode::Baseline => ResolvedForwardMode::Baseline,
        ParallelMode::Attention { shards } => ResolvedForwardMode::AttentionOnly {
            attn_shards: shards,
        },
        ParallelMode::Ffn { shards } => ResolvedForwardMode::FfnOnly { ffn_shards: shards },
        ParallelMode::AttentionFfn {
            attn_shards,
            ffn_shards,
        } => ResolvedForwardMode::AttentionFfn {
            attn_shards,
            ffn_shards,
        },
    }
}

fn resolve_shard_count(
    env_name: &str,
    shape_name: &str,
    shape_value: usize,
    requested: Option<usize>,
    policy: ShardPolicy,
    allow_baseline_one: bool,
    valid_divisors: &[usize],
) -> Result<(usize, Vec<String>), ParallelBenchError> {
    let Some(requested) = requested else {
        return Ok((1, Vec::new()));
    };

    if valid_divisors.contains(&requested) {
        return Ok((requested, Vec::new()));
    }

    match policy {
        ShardPolicy::FailFast => Err(ParallelBenchError::new(format!(
            "{env_name}={requested} is invalid for {shape_name}={shape_value}; valid divisors: {}{}",
            join_usizes(valid_divisors),
            if allow_baseline_one {
                ""
            } else {
                " (omit the flag to run baseline)"
            }
        ))),
        ShardPolicy::AutoAdjustNearest => {
            let applied = nearest_divisor(valid_divisors, requested);
            let mut notes = vec![format!(
                "{env_name}={requested} adjusted to {applied} for {shape_name}={shape_value}",
            )];
            if applied == 1 {
                notes.push(format!(
                    "{env_name} fell back to baseline because no nearby valid divisor above 1 exists for {shape_name}={shape_value}"
                ));
            }
            Ok((applied, notes))
        }
    }
}

fn divisors(value: usize) -> Vec<usize> {
    let mut divisors = Vec::new();
    let mut i = 1usize;
    while i * i <= value {
        if value % i == 0 {
            divisors.push(i);
            if i != value / i {
                divisors.push(value / i);
            }
        }
        i += 1;
    }
    divisors.sort_unstable();
    divisors
}

fn nearest_divisor(divisors: &[usize], requested: usize) -> usize {
    *divisors
        .iter()
        .min_by_key(|&&divisor| (requested.abs_diff(divisor), divisor))
        .expect("at least one divisor")
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
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
    q_head_count: usize,
    kv_head_count: usize,
    q_col_start: usize,
    q_col_count: usize,
    kv_col_start: usize,
    kv_col_count: usize,
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

struct CompiledAttentionShard {
    layout: AttentionShardLayout,
    cfg: ModelConfig,
    sdpa_exe: Executable,
    wo_exe: Executable,
}

struct CompiledFfnShard {
    layout: FfnShardLayout,
    w13_exe: Executable,
    w2_exe: Executable,
}

enum CompiledAttentionArtifacts {
    Baseline {
        sdpa_exe: Executable,
        wo_exe: Executable,
    },
    Sharded {
        workers: Vec<CompiledAttentionShard>,
    },
}

enum CompiledFfnArtifacts {
    Baseline {
        w13_exe: Executable,
        w2_exe: Executable,
        use_dual_w13: bool,
    },
    Sharded {
        workers: Vec<CompiledFfnShard>,
    },
}

fn attention_group_cfg(cfg: &ModelConfig, q_heads: usize, kv_heads: usize) -> ModelConfig {
    ModelConfig {
        dim: cfg.dim,
        hidden: cfg.hidden,
        heads: q_heads,
        kv_heads,
        hd: cfg.hd,
        seq: cfg.seq,
        nlayers: cfg.nlayers,
        vocab: cfg.vocab,
        q_dim: q_heads * cfg.hd,
        kv_dim: kv_heads * cfg.hd,
        gqa_ratio: q_heads / kv_heads,
    }
}

fn attention_shard_layouts(cfg: &ModelConfig, shard_count: usize) -> Vec<AttentionShardLayout> {
    assert!(
        cfg.heads % shard_count == 0 && cfg.kv_heads % shard_count == 0,
        "heads and kv_heads must be divisible by attention shard count"
    );
    let q_head_count = cfg.heads / shard_count;
    let kv_head_count = cfg.kv_heads / shard_count;
    let q_col_count = q_head_count * cfg.hd;
    let kv_col_count = kv_head_count * cfg.hd;
    (0..shard_count)
        .map(|i| AttentionShardLayout {
            q_head_count,
            kv_head_count,
            q_col_start: i * q_col_count,
            q_col_count,
            kv_col_start: i * kv_col_count,
            kv_col_count,
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

fn compile_sdpa_exe(cfg: &ModelConfig) -> Result<Executable, AneError> {
    let qos = NSQualityOfService::UserInteractive;
    sdpa_fwd::build(cfg).compile(qos)
}

fn allocate_sdpa_buffers(exe: Executable, cfg: &ModelConfig) -> SdpaRunner {
    SdpaRunner {
        exe,
        xnorm_in: tensor_data_new_logged("parallel_sdpa_xnorm_in", sdpa_fwd::xnorm_shape(cfg)),
        wq_in: tensor_data_new_logged("parallel_sdpa_wq_in", sdpa_fwd::wq_shape(cfg)),
        wk_in: tensor_data_new_logged("parallel_sdpa_wk_in", sdpa_fwd::wk_shape(cfg)),
        wv_in: tensor_data_new_logged("parallel_sdpa_wv_in", sdpa_fwd::wv_shape(cfg)),
        attn_out: tensor_data_new_logged("parallel_sdpa_attn_out", sdpa_fwd::attn_out_shape(cfg)),
        q_rope_out: tensor_data_new_logged("parallel_sdpa_q_rope_out", sdpa_fwd::q_rope_shape(cfg)),
        k_rope_out: tensor_data_new_logged("parallel_sdpa_k_rope_out", sdpa_fwd::k_rope_shape(cfg)),
        v_out: tensor_data_new_logged("parallel_sdpa_v_out", sdpa_fwd::v_shape(cfg)),
    }
}

fn compile_wo_exe(cfg: &ModelConfig) -> Result<Executable, AneError> {
    let qos = NSQualityOfService::UserInteractive;
    dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq).compile(qos)
}

fn allocate_wo_buffers(exe: Executable, cfg: &ModelConfig) -> WoRunner {
    let sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    WoRunner {
        exe,
        input: tensor_data_new_logged(
            "parallel_wo_input",
            ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.q_dim,
                height: 1,
                width: sp,
            },
        ),
        output: tensor_data_new_logged(
            "parallel_wo_output",
            ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: cfg.seq,
            },
        ),
    }
}

fn compile_baseline_attention_exes(
    cfg: &ModelConfig,
) -> Result<CompiledAttentionArtifacts, AneError> {
    Ok(CompiledAttentionArtifacts::Baseline {
        sdpa_exe: compile_sdpa_exe(cfg)?,
        wo_exe: compile_wo_exe(cfg)?,
    })
}

fn allocate_baseline_attention_runner(
    cfg: &ModelConfig,
    sdpa_exe: Executable,
    wo_exe: Executable,
) -> BaselineAttentionRunner {
    BaselineAttentionRunner {
        cfg: cfg.clone(),
        sdpa: allocate_sdpa_buffers(sdpa_exe, cfg),
        wo: allocate_wo_buffers(wo_exe, cfg),
    }
}

fn compile_sharded_attention_exes(
    cfg: &ModelConfig,
    shard_count: usize,
) -> Result<CompiledAttentionArtifacts, AneError> {
    let workers = attention_shard_layouts(cfg, shard_count)
        .into_iter()
        .map(|layout| {
            let worker_cfg = attention_group_cfg(cfg, layout.q_head_count, layout.kv_head_count);
            Ok(CompiledAttentionShard {
                layout,
                cfg: worker_cfg.clone(),
                sdpa_exe: compile_sdpa_exe(&worker_cfg)?,
                wo_exe: compile_wo_exe(&worker_cfg)?,
            })
        })
        .collect::<Result<Vec<_>, AneError>>()?;
    Ok(CompiledAttentionArtifacts::Sharded { workers })
}

fn allocate_sharded_attention_runner(
    cfg: &ModelConfig,
    workers: Vec<CompiledAttentionShard>,
) -> ShardedAttentionRunner {
    let workers = workers
        .into_iter()
        .map(|worker| AttentionShardWorker {
            layout: worker.layout,
            cfg: worker.cfg.clone(),
            sdpa: allocate_sdpa_buffers(worker.sdpa_exe, &worker.cfg),
            wo: allocate_wo_buffers(worker.wo_exe, &worker.cfg),
        })
        .collect();
    ShardedAttentionRunner {
        cfg: cfg.clone(),
        workers,
    }
}

fn compile_baseline_ffn_exes(cfg: &ModelConfig) -> Result<CompiledFfnArtifacts, AneError> {
    let qos = NSQualityOfService::UserInteractive;
    let use_dual_w13 = ffn_fused::can_use_dual_w13(cfg);
    let w13_exe = if use_dual_w13 {
        dyn_matmul::build_dual_separate(cfg.dim, cfg.hidden, cfg.seq).compile(qos)?
    } else {
        dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq).compile(qos)?
    };
    let w13_out_channels = if use_dual_w13 {
        2 * cfg.hidden
    } else {
        cfg.hidden
    };
    let _ = w13_out_channels;
    Ok(CompiledFfnArtifacts::Baseline {
        w13_exe,
        w2_exe: dyn_matmul::build(cfg.hidden, cfg.dim, cfg.seq).compile(qos)?,
        use_dual_w13,
    })
}

fn allocate_baseline_ffn_buffers(
    cfg: &ModelConfig,
    w13_exe: Executable,
    w2_exe: Executable,
    use_dual_w13: bool,
) -> BaselineFfnRunner {
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
    let w2_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    BaselineFfnRunner {
        cfg: cfg.clone(),
        use_dual_w13,
        w13_exe,
        w13_in: tensor_data_new_logged(
            "parallel_baseline_w13_in",
            ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: w13_sp,
            },
        ),
        w13_out: tensor_data_new_logged(
            "parallel_baseline_w13_out",
            ane_bridge::ane::Shape {
                batch: 1,
                channels: w13_out_channels,
                height: 1,
                width: cfg.seq,
            },
        ),
        w2_exe,
        w2_in: tensor_data_new_logged(
            "parallel_baseline_w2_in",
            ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: w2_sp,
            },
        ),
        w2_out: tensor_data_new_logged(
            "parallel_baseline_w2_out",
            ane_bridge::ane::Shape {
                batch: 1,
                channels: cfg.dim,
                height: 1,
                width: cfg.seq,
            },
        ),
        h1: vec![0.0; cfg.hidden * cfg.seq],
        h3: vec![0.0; cfg.hidden * cfg.seq],
        gate: vec![0.0; cfg.hidden * cfg.seq],
    }
}

fn compile_sharded_ffn_exes(
    cfg: &ModelConfig,
    shard_count: usize,
) -> Result<CompiledFfnArtifacts, AneError> {
    let qos = NSQualityOfService::UserInteractive;
    let workers = ffn_shard_layouts(cfg, shard_count)
        .into_iter()
        .map(|layout| {
            let shard_hidden = layout.col_count;
            Ok(CompiledFfnShard {
                layout,
                w13_exe: dyn_matmul::build_dual_separate(cfg.dim, shard_hidden, cfg.seq)
                    .compile(qos)?,
                w2_exe: dyn_matmul::build(shard_hidden, cfg.dim, cfg.seq).compile(qos)?,
            })
        })
        .collect::<Result<Vec<_>, AneError>>()?;
    Ok(CompiledFfnArtifacts::Sharded { workers })
}

fn allocate_sharded_ffn_runner(
    cfg: &ModelConfig,
    workers: Vec<CompiledFfnShard>,
) -> ShardedFfnRunner {
    let workers = workers
        .into_iter()
        .map(|worker| {
            let shard_hidden = worker.layout.col_count;
            let w13_sp = dyn_matmul::dual_separate_spatial_width(cfg.seq, shard_hidden);
            let w2_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
            FfnShardWorker {
                layout: worker.layout,
                w13_exe: worker.w13_exe,
                w2_exe: worker.w2_exe,
                w13_in: tensor_data_new_logged(
                    "parallel_sharded_w13_in",
                    ane_bridge::ane::Shape {
                        batch: 1,
                        channels: cfg.dim,
                        height: 1,
                        width: w13_sp,
                    },
                ),
                w13_out: tensor_data_new_logged(
                    "parallel_sharded_w13_out",
                    ane_bridge::ane::Shape {
                        batch: 1,
                        channels: 2 * shard_hidden,
                        height: 1,
                        width: cfg.seq,
                    },
                ),
                w2_in: tensor_data_new_logged(
                    "parallel_sharded_w2_in",
                    ane_bridge::ane::Shape {
                        batch: 1,
                        channels: shard_hidden,
                        height: 1,
                        width: w2_sp,
                    },
                ),
                w2_out: tensor_data_new_logged(
                    "parallel_sharded_w2_out",
                    ane_bridge::ane::Shape {
                        batch: 1,
                        channels: cfg.dim,
                        height: 1,
                        width: cfg.seq,
                    },
                ),
                h1: vec![0.0; shard_hidden * cfg.seq],
                h3: vec![0.0; shard_hidden * cfg.seq],
                gate: vec![0.0; shard_hidden * cfg.seq],
            }
        })
        .collect();
    ShardedFfnRunner {
        cfg: cfg.clone(),
        workers,
    }
}

/// Keep this function purely compile-oriented. It is called inside the typed
/// `compile_with_iosurface_retry` wrapper above, so it must preserve `Result`
/// semantics and avoid `TensorData::new()` allocations in the compile phase.
fn compile_mode_runner(
    cfg: &ModelConfig,
    mode: &ParallelMode,
) -> Result<(ModeRunner, f32), AneError> {
    let t0 = Instant::now();
    let compiled_attn = match mode {
        ParallelMode::Attention { shards }
        | ParallelMode::AttentionFfn {
            attn_shards: shards,
            ..
        } => compile_sharded_attention_exes(cfg, *shards)?,
        _ => compile_baseline_attention_exes(cfg)?,
    };
    let compiled_ffn = match mode {
        ParallelMode::Ffn { shards }
        | ParallelMode::AttentionFfn {
            ffn_shards: shards, ..
        } => compile_sharded_ffn_exes(cfg, *shards)?,
        _ => compile_baseline_ffn_exes(cfg)?,
    };

    let attn = match compiled_attn {
        CompiledAttentionArtifacts::Baseline { sdpa_exe, wo_exe } => {
            AttentionRunner::Baseline(allocate_baseline_attention_runner(cfg, sdpa_exe, wo_exe))
        }
        CompiledAttentionArtifacts::Sharded { workers } => {
            AttentionRunner::Sharded(allocate_sharded_attention_runner(cfg, workers))
        }
    };
    let ffn = match compiled_ffn {
        CompiledFfnArtifacts::Baseline {
            w13_exe,
            w2_exe,
            use_dual_w13,
        } => FfnRunner::Baseline(allocate_baseline_ffn_buffers(
            cfg,
            w13_exe,
            w2_exe,
            use_dual_w13,
        )),
        CompiledFfnArtifacts::Sharded { workers } => {
            FfnRunner::Sharded(allocate_sharded_ffn_runner(cfg, workers))
        }
    };
    Ok((
        ModeRunner {
            attn,
            ffn,
            scratch: LayerScratch::new(cfg),
        },
        t0.elapsed().as_secs_f32(),
    ))
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

fn run_baseline_attention_into(
    runner: &mut BaselineAttentionRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    scratch: &mut LayerScratch,
) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);

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
}

fn run_sharded_attention_into(
    runner: &mut ShardedAttentionRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    scratch: &mut LayerScratch,
) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);

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
                        worker.layout.kv_col_start,
                        worker.layout.kv_col_count,
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
                        worker.layout.kv_col_start,
                        worker.layout.kv_col_count,
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
}

fn run_baseline_ffn_into(
    runner: &mut BaselineFfnRunner,
    layer_weights: &LayerWeights,
    scratch: &mut LayerScratch,
    x_next: &mut [f32],
) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);
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
}

fn run_sharded_ffn_into(
    runner: &mut ShardedFfnRunner,
    layer_weights: &LayerWeights,
    scratch: &mut LayerScratch,
    x_next: &mut [f32],
) {
    let cfg = &runner.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq;
    let alpha = layer_alpha(cfg);
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
}

fn run_mode_layer_forward_into(
    runner: &mut ModeRunner,
    layer_weights: &LayerWeights,
    x: &[f32],
    x_next: &mut [f32],
) {
    match &mut runner.attn {
        AttentionRunner::Baseline(attn) => {
            run_baseline_attention_into(attn, layer_weights, x, &mut runner.scratch)
        }
        AttentionRunner::Sharded(attn) => {
            run_sharded_attention_into(attn, layer_weights, x, &mut runner.scratch)
        }
    }
    match &mut runner.ffn {
        FfnRunner::Baseline(ffn) => {
            run_baseline_ffn_into(ffn, layer_weights, &mut runner.scratch, x_next)
        }
        FfnRunner::Sharded(ffn) => {
            run_sharded_ffn_into(ffn, layer_weights, &mut runner.scratch, x_next)
        }
    }
}

fn finalize_logits_and_loss(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ws: &mut ModelForwardWorkspace,
    targets: &[u32],
    softcap: f32,
) -> f32 {
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

    total_loss / seq as f32
}

fn run_mode_full_forward_once(
    cfg: &ModelConfig,
    runner: &mut ModeRunner,
    ws: &mut ModelForwardWorkspace,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> f32 {
    let dim = cfg.dim;
    let seq = cfg.seq;

    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    for layer_weights in &weights.layers {
        let x_buf = ws.x_buf.clone();
        run_mode_layer_forward_into(runner, layer_weights, &x_buf, &mut ws.x_next_buf);
        std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
    }

    finalize_logits_and_loss(cfg, weights, ws, targets, softcap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(heads: usize, kv_heads: usize, hidden: usize) -> ModelConfig {
        ModelConfig {
            dim: 1024,
            hidden,
            heads,
            kv_heads,
            hd: 128,
            seq: 512,
            nlayers: 8,
            vocab: 8192,
            q_dim: heads * 128,
            kv_dim: kv_heads * 128,
            gqa_ratio: heads / kv_heads,
        }
    }

    #[test]
    fn parse_ffn_one_is_invalid() {
        unsafe { std::env::set_var("FFN_SHARDS", "1") };
        let err = ParallelBenchRequest::from_env(ShardPolicy::FailFast).unwrap_err();
        assert!(err.to_string().contains("omit FFN_SHARDS to run baseline"));
        unsafe { std::env::remove_var("FFN_SHARDS") };
    }

    #[test]
    fn parse_rejects_unsupported_attention_request() {
        unsafe { std::env::set_var("ATTN_SHARDS", "3") };
        let err = ParallelBenchRequest::from_env(ShardPolicy::FailFast).unwrap_err();
        assert!(
            err.to_string()
                .contains("allowed values are 1, 2, 4, 8, 10")
        );
        unsafe { std::env::remove_var("ATTN_SHARDS") };
    }

    #[test]
    fn fail_fast_reports_valid_divisors() {
        let err = resolve_parallel_bench_config(
            &cfg(40, 40, 11008),
            ParallelBenchRequest {
                attn_request: None,
                ffn_request: Some(10),
                policy: ShardPolicy::FailFast,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("valid divisors"));
        assert!(err.to_string().contains("11008"));
    }

    #[test]
    fn auto_adjust_uses_nearest_divisor_and_smaller_on_tie() {
        let resolved = resolve_parallel_bench_config(
            &cfg(36, 36, 11008),
            ParallelBenchRequest {
                attn_request: Some(10),
                ffn_request: None,
                policy: ShardPolicy::AutoAdjustNearest,
            },
        )
        .expect("auto adjust");
        assert_eq!(resolved.applied_attn, 9);
        assert!(resolved.notes[0].contains("adjusted to 9"));

        let tied = nearest_divisor(&[1, 4, 8], 6);
        assert_eq!(tied, 4);
    }

    #[test]
    fn resolve_mode_mapping_is_correct() {
        let baseline = resolve_parallel_bench_config(
            &cfg(40, 40, 13824),
            ParallelBenchRequest {
                attn_request: None,
                ffn_request: None,
                policy: ShardPolicy::FailFast,
            },
        )
        .unwrap();
        assert_eq!(baseline.mode, ParallelMode::Baseline);

        let attn = resolve_parallel_bench_config(
            &cfg(40, 40, 13824),
            ParallelBenchRequest {
                attn_request: Some(4),
                ffn_request: None,
                policy: ShardPolicy::FailFast,
            },
        )
        .unwrap();
        assert_eq!(attn.mode, ParallelMode::Attention { shards: 4 });

        let ffn = resolve_parallel_bench_config(
            &cfg(40, 40, 13824),
            ParallelBenchRequest {
                attn_request: None,
                ffn_request: Some(8),
                policy: ShardPolicy::FailFast,
            },
        )
        .unwrap();
        assert_eq!(ffn.mode, ParallelMode::Ffn { shards: 8 });

        let combined = resolve_parallel_bench_config(
            &cfg(40, 40, 13824),
            ParallelBenchRequest {
                attn_request: Some(4),
                ffn_request: Some(8),
                policy: ShardPolicy::FailFast,
            },
        )
        .unwrap();
        assert_eq!(
            combined.mode,
            ParallelMode::AttentionFfn {
                attn_shards: 4,
                ffn_shards: 8
            }
        );
    }
}
