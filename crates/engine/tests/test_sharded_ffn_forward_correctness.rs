use engine::layer::{self, CompiledKernels, ForwardCache, LayerWeights, ShardedFfnForwardRuntime};
use engine::model::ModelConfig;

const MAX_ABS_DIFF_TOL: f32 = 2e-2;
const COSINE_SIM_TOL: f32 = 0.999;

fn deterministic_signal(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 29 + 5) % 103) as f32 - 51.0) * scale)
        .collect()
}

fn assert_exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: {x} vs {y} (not bit-identical)"
        );
    }
}

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((x - y).abs());
        dot += x as f64 * y as f64;
        norm_a += x as f64 * x as f64;
        norm_b += y as f64 * y as f64;
    }
    let cosine = if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    } else {
        1.0
    };
    assert!(
        max_abs <= MAX_ABS_DIFF_TOL,
        "{label}: max_abs_diff={max_abs} exceeds tolerance {MAX_ABS_DIFF_TOL}"
    );
    assert!(
        cosine >= COSINE_SIM_TOL,
        "{label}: cosine={cosine} below tolerance {COSINE_SIM_TOL}"
    );
}

fn assert_cache_close(reference: &ForwardCache, test: &ForwardCache, label: &str) {
    assert_close(&reference.x, &test.x, &format!("{label}.x"));
    assert_close(&reference.xnorm, &test.xnorm, &format!("{label}.xnorm"));
    assert_close(
        &reference.rms_inv1,
        &test.rms_inv1,
        &format!("{label}.rms_inv1"),
    );
    assert_close(&reference.q_rope, &test.q_rope, &format!("{label}.q_rope"));
    assert_close(&reference.k_rope, &test.k_rope, &format!("{label}.k_rope"));
    assert_close(&reference.v, &test.v, &format!("{label}.v"));
    assert_close(
        &reference.attn_out,
        &test.attn_out,
        &format!("{label}.attn_out"),
    );
    assert_close(&reference.o_out, &test.o_out, &format!("{label}.o_out"));
    assert_close(&reference.x2, &test.x2, &format!("{label}.x2"));
    assert_close(&reference.x2norm, &test.x2norm, &format!("{label}.x2norm"));
    assert_close(
        &reference.rms_inv2,
        &test.rms_inv2,
        &format!("{label}.rms_inv2"),
    );
    assert_close(&reference.h1, &test.h1, &format!("{label}.h1"));
    assert_close(&reference.h3, &test.h3, &format!("{label}.h3"));
    assert_close(&reference.gate, &test.gate, &format!("{label}.gate"));
}

fn assert_cache_exact(reference: &ForwardCache, test: &ForwardCache, label: &str) {
    assert_exact(&reference.x, &test.x, &format!("{label}.x"));
    assert_exact(&reference.xnorm, &test.xnorm, &format!("{label}.xnorm"));
    assert_exact(
        &reference.rms_inv1,
        &test.rms_inv1,
        &format!("{label}.rms_inv1"),
    );
    assert_exact(&reference.q_rope, &test.q_rope, &format!("{label}.q_rope"));
    assert_exact(&reference.k_rope, &test.k_rope, &format!("{label}.k_rope"));
    assert_exact(&reference.v, &test.v, &format!("{label}.v"));
    assert_exact(
        &reference.attn_out,
        &test.attn_out,
        &format!("{label}.attn_out"),
    );
    assert_exact(&reference.o_out, &test.o_out, &format!("{label}.o_out"));
    assert_exact(&reference.x2, &test.x2, &format!("{label}.x2"));
    assert_exact(&reference.x2norm, &test.x2norm, &format!("{label}.x2norm"));
    assert_exact(
        &reference.rms_inv2,
        &test.rms_inv2,
        &format!("{label}.rms_inv2"),
    );
    assert_exact(&reference.h1, &test.h1, &format!("{label}.h1"));
    assert_exact(&reference.h3, &test.h3, &format!("{label}.h3"));
    assert_exact(&reference.gate, &test.gate, &format!("{label}.gate"));
    assert_exact(
        &reference.w2t_scratch,
        &test.w2t_scratch,
        &format!("{label}.w2t_scratch"),
    );
    assert_eq!(
        reference.w2t_generation, test.w2t_generation,
        "{label}.w2t_generation: {} vs {}",
        reference.w2t_generation, test.w2t_generation
    );
}

fn setup_case() -> (ModelConfig, CompiledKernels, LayerWeights, Vec<f32>) {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x = deterministic_signal(cfg.dim * cfg.seq, 0.01);
    (cfg, kernels, weights, x)
}

#[test]
fn sharded_ffn_forward_correctness_suite() {
    let (cfg, kernels, weights, x) = setup_case();

    let mut runtime = ShardedFfnForwardRuntime::compile(&cfg, 4).expect("sharded FFN runtime");
    let mut serial_cache = ForwardCache::new(&cfg);
    let mut serial_x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into_with_training_ffn_sharded_serial(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut serial_cache,
        &mut serial_x_next,
        Some(&mut runtime),
    );

    let mut parallel_cache = ForwardCache::new(&cfg);
    let mut parallel_x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut parallel_cache,
        &mut parallel_x_next,
        Some(&mut runtime),
    );

    assert_exact(
        &serial_x_next,
        &parallel_x_next,
        "parallel_vs_serial.x_next",
    );
    assert_exact(
        &serial_cache.h1,
        &parallel_cache.h1,
        "parallel_vs_serial.h1",
    );
    assert_exact(
        &serial_cache.h3,
        &parallel_cache.h3,
        "parallel_vs_serial.h3",
    );
    assert_exact(
        &serial_cache.gate,
        &parallel_cache.gate,
        "parallel_vs_serial.gate",
    );
    assert_eq!(
        parallel_cache.w2t_generation, weights.w2_generation,
        "parallel path did not refresh the cached W2 transpose generation"
    );

    let mut baseline_cache = ForwardCache::new(&cfg);
    let mut baseline_x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut baseline_cache,
        &mut baseline_x_next,
    );

    let mut sharded_cache = ForwardCache::new(&cfg);
    let mut sharded_x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut sharded_cache,
        &mut sharded_x_next,
        Some(&mut runtime),
    );

    assert_close(
        &baseline_x_next,
        &sharded_x_next,
        "sharded_vs_baseline.x_next",
    );
    assert_cache_close(&baseline_cache, &sharded_cache, "sharded_vs_baseline");

    let mut cache1 = ForwardCache::new(&cfg);
    let mut x_next1 = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut cache1,
        &mut x_next1,
        Some(&mut runtime),
    );

    let mut cache2 = ForwardCache::new(&cfg);
    let mut x_next2 = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &x,
        &mut cache2,
        &mut x_next2,
        Some(&mut runtime),
    );

    assert_exact(&x_next1, &x_next2, "idempotent.x_next");
    assert_cache_exact(&cache1, &cache2, "idempotent");
}
