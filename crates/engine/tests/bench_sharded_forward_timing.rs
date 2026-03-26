use engine::layer::{
    self, CompiledKernels, ForwardCache, LayerWeights, ShardedFfnForwardRuntime,
    ShardedTrainingForwardTimings, TimingMode,
};
use engine::model::ModelConfig;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

const WARMUP_RUNS: usize = 2;
const TIMED_RUNS: usize = 5;

#[derive(Debug, Clone, Serialize)]
struct CaseResult {
    mode: String,
    wall_only: ShardedTrainingForwardTimings,
    wall_and_hw: ShardedTrainingForwardTimings,
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/sharded_forward_timing")
}

fn ensure_results_dir() {
    fs::create_dir_all(results_dir()).expect("create sharded_forward_timing results dir");
}

fn write_json<T: Serialize>(path: &Path, value: &T) {
    ensure_results_dir();
    let json = serde_json::to_string_pretty(value).expect("serialize json");
    fs::write(path, json).expect("write json");
}

fn deterministic_signal(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 23 + 3) % 101) as f32 - 50.0) * scale)
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

fn average_dispatch(samples: &[engine::layer::DispatchTiming]) -> engine::layer::DispatchTiming {
    engine::layer::DispatchTiming {
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

fn average_shard(
    samples: &[engine::layer::ShardedFfnForwardShardTiming],
) -> engine::layer::ShardedFfnForwardShardTiming {
    engine::layer::ShardedFfnForwardShardTiming {
        shard_idx: samples[0].shard_idx,
        w13: average_dispatch(
            &samples
                .iter()
                .map(|sample| sample.w13.clone())
                .collect::<Vec<_>>(),
        ),
        gate_cpu_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.gate_cpu_ms)
                .collect::<Vec<_>>(),
        ),
        w2: average_dispatch(
            &samples
                .iter()
                .map(|sample| sample.w2.clone())
                .collect::<Vec<_>>(),
        ),
        merge_cpu_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.merge_cpu_ms)
                .collect::<Vec<_>>(),
        ),
        cache_store_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.cache_store_ms)
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

fn average_forward_timings(
    samples: &[ShardedTrainingForwardTimings],
) -> ShardedTrainingForwardTimings {
    let shard_count = samples[0].ffn_shards.len();
    let mut ffn_shards = Vec::with_capacity(shard_count);
    for shard_idx in 0..shard_count {
        ffn_shards.push(average_shard(
            &samples
                .iter()
                .map(|sample| sample.ffn_shards[shard_idx].clone())
                .collect::<Vec<_>>(),
        ));
    }

    ShardedTrainingForwardTimings {
        rmsnorm1_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.rmsnorm1_ms)
                .collect::<Vec<_>>(),
        ),
        stage_sdpa_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.stage_sdpa_ms)
                .collect::<Vec<_>>(),
        ),
        ane_sdpa_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ane_sdpa_ms)
                .collect::<Vec<_>>(),
        ),
        read_sdpa_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.read_sdpa_ms)
                .collect::<Vec<_>>(),
        ),
        stage_wo_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.stage_wo_ms)
                .collect::<Vec<_>>(),
        ),
        ane_wo_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ane_wo_ms)
                .collect::<Vec<_>>(),
        ),
        read_wo_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.read_wo_ms)
                .collect::<Vec<_>>(),
        ),
        residual_rmsnorm2_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.residual_rmsnorm2_ms)
                .collect::<Vec<_>>(),
        ),
        ffn_shards,
        ffn_residual_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.ffn_residual_ms)
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
        host_overhead_ms: mean_option(
            &samples
                .iter()
                .map(|sample| sample.host_overhead_ms)
                .collect::<Vec<_>>(),
        ),
        total_ms: mean(
            &samples
                .iter()
                .map(|sample| sample.total_ms)
                .collect::<Vec<_>>(),
        ),
    }
}

fn print_timings(label: &str, timings: &ShardedTrainingForwardTimings) {
    println!("\n-- {label} --");
    println!(
        "total={:.2}ms rms1={:.2} sdpa_stage={:.2} sdpa_wall={:.2} sdpa_read={:.2} wo_stage={:.2} wo_wall={:.2} wo_read={:.2} residual+rms2={:.2}",
        timings.total_ms,
        timings.rmsnorm1_ms,
        timings.stage_sdpa_ms,
        timings.ane_sdpa_ms,
        timings.read_sdpa_ms,
        timings.stage_wo_ms,
        timings.ane_wo_ms,
        timings.read_wo_ms,
        timings.residual_rmsnorm2_ms
    );
    for shard in &timings.ffn_shards {
        println!(
            "  shard {}: w13(stage={:.2} wall={:.2} hw={}) gate={:.2} w2(stage={:.2} wall={:.2} hw={}) merge={:.2} cache_store={:.2} total={:.2}",
            shard.shard_idx,
            shard.w13.staging_ms,
            shard.w13.ane_wall_ms,
            shard
                .w13
                .ane_hw_ns
                .map(|value| format!("{value}ns"))
                .unwrap_or_else(|| "n/a".to_string()),
            shard.gate_cpu_ms,
            shard.w2.staging_ms,
            shard.w2.ane_wall_ms,
            shard
                .w2
                .ane_hw_ns
                .map(|value| format!("{value}ns"))
                .unwrap_or_else(|| "n/a".to_string()),
            shard.merge_cpu_ms,
            shard.cache_store_ms,
            shard.shard_total_ms
        );
    }
    println!(
        "ffn_wall={:.2} ffn_residual={:.2} ffn_cpu_sum={:.2} ffn_ane_wall_sum={:.2} ffn_ane_hw_sum={} host_overhead={}",
        timings.ffn_wall_ms,
        timings.ffn_residual_ms,
        timings.ffn_sum_cpu_ms,
        timings.ffn_sum_ane_wall_ms,
        timings
            .ffn_sum_ane_hw_ms
            .map(|value| format!("{value:.2}ms"))
            .unwrap_or_else(|| "n/a".to_string()),
        timings
            .host_overhead_ms
            .map(|value| format!("{value:.2}ms"))
            .unwrap_or_else(|| "n/a".to_string())
    );
}

#[test]
#[ignore]
fn bench_sharded_forward_timing() {
    let cfg = ModelConfig::target_600m();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x = deterministic_signal(cfg.dim * cfg.seq, 0.01);

    let mut runtime =
        ShardedFfnForwardRuntime::compile(&cfg, 4).expect("compile sharded FFN forward runtime");
    let mut wall_samples = Vec::with_capacity(TIMED_RUNS);
    let mut hw_samples = Vec::with_capacity(TIMED_RUNS);
    for iter in 0..(WARMUP_RUNS + TIMED_RUNS) {
        let mut wall_cache = ForwardCache::new(&cfg);
        let mut wall_x_next = vec![0.0f32; cfg.dim * cfg.seq];
        let wall = layer::forward_into_with_training_ffn_timed(
            &cfg,
            &kernels,
            &weights,
            &x,
            &mut wall_cache,
            &mut wall_x_next,
            &mut runtime,
            TimingMode::WallOnly,
        );
        let mut hw_cache = ForwardCache::new(&cfg);
        let mut hw_x_next = vec![0.0f32; cfg.dim * cfg.seq];
        let hw = layer::forward_into_with_training_ffn_timed(
            &cfg,
            &kernels,
            &weights,
            &x,
            &mut hw_cache,
            &mut hw_x_next,
            &mut runtime,
            TimingMode::WallAndHw,
        );
        if iter >= WARMUP_RUNS {
            wall_samples.push(wall);
            hw_samples.push(hw);
        }
    }

    let wall_only = average_forward_timings(&wall_samples);
    let wall_and_hw = average_forward_timings(&hw_samples);
    print_timings("WallOnly", &wall_only);
    print_timings("WallAndHw", &wall_and_hw);

    let result = CaseResult {
        mode: "ffn4_training_path".to_string(),
        wall_only,
        wall_and_hw,
    };
    write_json(
        &results_dir().join("600m_sharded_forward_timing.json"),
        &result,
    );
}
