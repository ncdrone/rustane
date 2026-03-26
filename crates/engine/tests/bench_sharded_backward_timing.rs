use engine::layer::{
    self, BackwardWorkspace, CompiledKernels, DispatchTiming, ForwardCache, LayerGrads,
    LayerWeights, ShardTiming, ShardedAttentionBackwardRuntime, ShardedBackwardTimings,
    ShardedFfnBackwardRuntime, TimingMode,
};
use engine::model::ModelConfig;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

const WARMUP_RUNS: usize = 2;
const TIMED_RUNS: usize = 5;

#[derive(Debug, Clone, Serialize)]
struct BaselineSummary {
    total_ms: f32,
    scale_dy_ms: f32,
    rmsnorm2_bwd_ms: f32,
    rmsnorm1_bwd_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
struct CaseResult {
    name: String,
    ffn_shards: Option<usize>,
    attn_shards: Option<usize>,
    wall_only: ShardedBackwardTimings,
    wall_and_hw: ShardedBackwardTimings,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    config_name: String,
    baseline: BaselineSummary,
    cases: Vec<CaseResult>,
}

#[derive(Clone, Copy)]
struct CaseSpec {
    name: &'static str,
    ffn_shards: Option<usize>,
    attn_shards: Option<usize>,
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/sharded_backward_timing")
}

fn ensure_results_dir() {
    fs::create_dir_all(results_dir()).expect("create sharded_backward_timing results dir");
}

fn write_json<T: Serialize>(path: &Path, value: &T) {
    ensure_results_dir();
    let json = serde_json::to_string_pretty(value).expect("serialize json");
    fs::write(path, json).expect("write json");
}

fn deterministic_signal(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 29 + 5) % 113) as f32 - 56.0) * scale)
        .collect()
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn mean_option(values: &[Option<f32>]) -> Option<f32> {
    if values.is_empty() || values.iter().any(|value| value.is_none()) {
        None
    } else {
        Some(values.iter().map(|value| value.unwrap()).sum::<f32>() / values.len() as f32)
    }
}

fn mean_option_u64(values: &[Option<u64>]) -> Option<u64> {
    if values.is_empty() || values.iter().any(|value| value.is_none()) {
        None
    } else {
        Some(
            (values
                .iter()
                .map(|value| value.unwrap() as f64)
                .sum::<f64>()
                / values.len() as f64)
                .round() as u64,
        )
    }
}

fn average_dispatch(samples: &[DispatchTiming]) -> DispatchTiming {
    DispatchTiming {
        kernel_name: samples[0].kernel_name.clone(),
        staging_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.staging_ms)
                .collect::<Vec<_>>(),
        ),
        ane_wall_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ane_wall_ms)
                .collect::<Vec<_>>(),
        ),
        ane_hw_ns: mean_option_u64(
            &samples
                .iter()
                .map(|sample| sample.ane_hw_ns)
                .collect::<Vec<_>>(),
        ),
        readback_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.readback_ms)
                .collect::<Vec<_>>(),
        ),
    }
}

fn average_shard(samples: &[ShardTiming]) -> ShardTiming {
    let dispatch_count = samples[0].dispatches.len();
    let mut dispatches = Vec::with_capacity(dispatch_count);
    for dispatch_idx in 0..dispatch_count {
        let dispatch_samples: Vec<DispatchTiming> = samples
            .iter()
            .map(|sample| sample.dispatches[dispatch_idx].clone())
            .collect();
        dispatches.push(average_dispatch(&dispatch_samples));
    }
    ShardTiming {
        shard_idx: samples[0].shard_idx,
        dispatches,
        cpu_grad_accum_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.cpu_grad_accum_ms)
                .collect::<Vec<_>>(),
        ),
        cpu_silu_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.cpu_silu_ms)
                .collect::<Vec<_>>(),
        ),
        shard_total_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.shard_total_ms)
                .collect::<Vec<_>>(),
        ),
    }
}

fn average_sharded_timings(samples: &[ShardedBackwardTimings]) -> ShardedBackwardTimings {
    let ffn_shard_count = samples[0].ffn_shards.len();
    let mut ffn_shards = Vec::with_capacity(ffn_shard_count);
    for shard_idx in 0..ffn_shard_count {
        let shard_samples: Vec<ShardTiming> = samples
            .iter()
            .map(|sample| sample.ffn_shards[shard_idx].clone())
            .collect();
        ffn_shards.push(average_shard(&shard_samples));
    }

    let attn_shard_count = samples[0].attn_shards.len();
    let mut attn_shards = Vec::with_capacity(attn_shard_count);
    for shard_idx in 0..attn_shard_count {
        let shard_samples: Vec<ShardTiming> = samples
            .iter()
            .map(|sample| sample.attn_shards[shard_idx].clone())
            .collect();
        attn_shards.push(average_shard(&shard_samples));
    }

    ShardedBackwardTimings {
        ffn_scale_dy_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_scale_dy_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_shards,
        ffn_grad_scatter_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_grad_scatter_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_dx_merge_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_dx_merge_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_wall_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_wall_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_sum_ane_wall_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_sum_ane_wall_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_sum_cpu_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_sum_cpu_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_sum_ane_hw_ms: mean_option(
            &samples
                .iter()
                .map(|sample| sample.ffn_sum_ane_hw_ms)
                .collect::<Vec<_>>(),
        ),
        attn_transpose_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.attn_transpose_ms)
                .collect::<Vec<_>>(),
        ),
        attn_shards,
        attn_grad_scatter_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.attn_grad_scatter_ms)
                .collect::<Vec<_>>(),
        ),
        attn_dx_merge_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.attn_dx_merge_ms)
                .collect::<Vec<_>>(),
        ),
        attn_wall_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.attn_wall_ms)
                .collect::<Vec<_>>(),
        ),
        attn_sum_ane_wall_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.attn_sum_ane_wall_ms)
                .collect::<Vec<_>>(),
        ),
        attn_sum_cpu_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.attn_sum_cpu_ms)
                .collect::<Vec<_>>(),
        ),
        attn_sum_ane_hw_ms: mean_option(
            &samples
                .iter()
                .map(|sample| sample.attn_sum_ane_hw_ms)
                .collect::<Vec<_>>(),
        ),
        host_overhead_ms: mean_option(
            &samples
                .iter()
                .map(|sample| sample.host_overhead_ms)
                .collect::<Vec<_>>(),
        ),
    }
}

fn print_case(result: &CaseResult) {
    println!(
        "\n=== {} (FFN_SHARDS={:?}, ATTN_SHARDS={:?}) ===",
        result.name, result.ffn_shards, result.attn_shards
    );
    for (mode_name, timings) in [
        ("WallOnly", &result.wall_only),
        ("WallAndHw", &result.wall_and_hw),
    ] {
        println!("  -- {mode_name} --");
        for shard in &timings.ffn_shards {
            println!("  FFN shard {}:", shard.shard_idx);
            for dispatch in &shard.dispatches {
                let hw = dispatch
                    .ane_hw_ns
                    .map(|value| format!("{value}ns"))
                    .unwrap_or_else(|| "n/a".to_string());
                println!(
                    "    {:<7} stage={:>6.2}ms wall={:>6.2}ms hw={:<12} read={:>6.2}ms",
                    dispatch.kernel_name,
                    dispatch.staging_ms,
                    dispatch.ane_wall_ms,
                    hw,
                    dispatch.readback_ms
                );
            }
            println!(
                "    cpu_silu={:>6.2}ms cpu_grad_accum={:>6.2}ms shard_total={:>6.2}ms",
                shard.cpu_silu_ms, shard.cpu_grad_accum_ms, shard.shard_total_ms
            );
        }
        for shard in &timings.attn_shards {
            println!("  ATTN shard {}:", shard.shard_idx);
            for dispatch in &shard.dispatches {
                let hw = dispatch
                    .ane_hw_ns
                    .map(|value| format!("{value}ns"))
                    .unwrap_or_else(|| "n/a".to_string());
                println!(
                    "    {:<7} stage={:>6.2}ms wall={:>6.2}ms hw={:<12} read={:>6.2}ms",
                    dispatch.kernel_name,
                    dispatch.staging_ms,
                    dispatch.ane_wall_ms,
                    hw,
                    dispatch.readback_ms
                );
            }
            println!(
                "    cpu_grad_accum={:>6.2}ms shard_total={:>6.2}ms",
                shard.cpu_grad_accum_ms, shard.shard_total_ms
            );
        }
        println!(
            "  FFN wall={:>6.2}ms grad_scatter={:>6.2}ms dx_merge={:>6.2}ms cpu_sum={:>6.2}ms ane_wall_sum={:>6.2}ms ane_hw_sum={}",
            timings.ffn_wall_ms,
            timings.ffn_grad_scatter_ms,
            timings.ffn_dx_merge_ms,
            timings.ffn_sum_cpu_ms,
            timings.ffn_sum_ane_wall_ms,
            timings
                .ffn_sum_ane_hw_ms
                .map(|value| format!("{value:.2}ms"))
                .unwrap_or_else(|| "n/a".to_string())
        );
        println!(
            "  ATTN wall={:>6.2}ms grad_scatter={:>6.2}ms dx_merge={:>6.2}ms cpu_sum={:>6.2}ms ane_wall_sum={:>6.2}ms ane_hw_sum={}",
            timings.attn_wall_ms,
            timings.attn_grad_scatter_ms,
            timings.attn_dx_merge_ms,
            timings.attn_sum_cpu_ms,
            timings.attn_sum_ane_wall_ms,
            timings
                .attn_sum_ane_hw_ms
                .map(|value| format!("{value:.2}ms"))
                .unwrap_or_else(|| "n/a".to_string())
        );
    }
}

fn run_case(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    spec: CaseSpec,
) -> CaseResult {
    let mut ffn_runtime = spec.ffn_shards.map(|shards| {
        ShardedFfnBackwardRuntime::compile(cfg, shards)
            .expect("compile sharded FFN backward runtime")
    });
    let mut attn_runtime = spec.attn_shards.map(|shards| {
        ShardedAttentionBackwardRuntime::compile(cfg, shards)
            .expect("compile sharded attention backward runtime")
    });

    let mut wall_samples = Vec::with_capacity(TIMED_RUNS);
    let mut hw_samples = Vec::with_capacity(TIMED_RUNS);
    for iter in 0..(WARMUP_RUNS + TIMED_RUNS) {
        let mut wall_grads = LayerGrads::zeros(cfg);
        let mut wall_ws = BackwardWorkspace::new(cfg);
        let mut wall_dx = vec![0.0f32; cfg.dim * cfg.seq];
        let wall_timings = layer::backward_into_sharded_timed(
            cfg,
            kernels,
            weights,
            cache,
            dy,
            &mut wall_grads,
            &mut wall_ws,
            &mut wall_dx,
            attn_runtime.as_mut(),
            ffn_runtime.as_mut(),
            TimingMode::WallOnly,
        );

        let mut hw_grads = LayerGrads::zeros(cfg);
        let mut hw_ws = BackwardWorkspace::new(cfg);
        let mut hw_dx = vec![0.0f32; cfg.dim * cfg.seq];
        let hw_timings = layer::backward_into_sharded_timed(
            cfg,
            kernels,
            weights,
            cache,
            dy,
            &mut hw_grads,
            &mut hw_ws,
            &mut hw_dx,
            attn_runtime.as_mut(),
            ffn_runtime.as_mut(),
            TimingMode::WallAndHw,
        );

        if iter >= WARMUP_RUNS {
            wall_samples.push(wall_timings);
            hw_samples.push(hw_timings);
        }
    }

    CaseResult {
        name: spec.name.to_string(),
        ffn_shards: spec.ffn_shards,
        attn_shards: spec.attn_shards,
        wall_only: average_sharded_timings(&wall_samples),
        wall_and_hw: average_sharded_timings(&hw_samples),
    }
}

#[test]
#[ignore]
fn bench_sharded_backward_timing() {
    let cfg = ModelConfig::target_600m();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x = deterministic_signal(cfg.dim * cfg.seq, 0.008);
    let dy = deterministic_signal(cfg.dim * cfg.seq, 0.004);
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);

    let mut baseline_samples = Vec::with_capacity(TIMED_RUNS);
    for iter in 0..(WARMUP_RUNS + TIMED_RUNS) {
        let mut grads = LayerGrads::zeros(&cfg);
        let mut ws = BackwardWorkspace::new(&cfg);
        let (_, timings) =
            layer::backward_timed(&cfg, &kernels, &weights, &cache, &dy, &mut grads, &mut ws);
        if iter >= WARMUP_RUNS {
            baseline_samples.push(timings);
        }
    }
    let baseline = BaselineSummary {
        total_ms: mean(
            &baseline_samples
                .iter()
                .map(|sample| sample.total_ms)
                .collect::<Vec<_>>(),
        ),
        scale_dy_ms: mean(
            &baseline_samples
                .iter()
                .map(|sample| sample.scale_dy_ms)
                .collect::<Vec<_>>(),
        ),
        rmsnorm2_bwd_ms: mean(
            &baseline_samples
                .iter()
                .map(|sample| sample.rmsnorm2_bwd_ms)
                .collect::<Vec<_>>(),
        ),
        rmsnorm1_bwd_ms: mean(
            &baseline_samples
                .iter()
                .map(|sample| sample.rmsnorm1_bwd_ms)
                .collect::<Vec<_>>(),
        ),
    };

    let cases = [
        CaseSpec {
            name: "ffn2",
            ffn_shards: Some(2),
            attn_shards: None,
        },
        CaseSpec {
            name: "ffn4",
            ffn_shards: Some(4),
            attn_shards: None,
        },
        CaseSpec {
            name: "attn2",
            ffn_shards: None,
            attn_shards: Some(2),
        },
        CaseSpec {
            name: "both",
            ffn_shards: Some(4),
            attn_shards: Some(2),
        },
    ];

    let case_results: Vec<CaseResult> = cases
        .iter()
        .map(|spec| run_case(&cfg, &kernels, &weights, &cache, &dy, *spec))
        .collect();

    println!("\n=== Baseline 600M ===");
    println!(
        "  total={:.2}ms scale_dy={:.2}ms rmsnorm2={:.2}ms rmsnorm1={:.2}ms",
        baseline.total_ms, baseline.scale_dy_ms, baseline.rmsnorm2_bwd_ms, baseline.rmsnorm1_bwd_ms
    );
    for result in &case_results {
        print_case(result);
    }

    let result = BenchmarkResult {
        config_name: "600M".to_string(),
        baseline,
        cases: case_results,
    };
    write_json(
        &results_dir().join("600m_sharded_backward_timing.json"),
        &result,
    );
}
