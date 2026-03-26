use engine::layer::{
    self, BackwardWorkspace, CompiledKernels, ForwardCache, LayerGrads, LayerWeights,
    ShardedAttentionBackwardRuntime,
};
use engine::model::ModelConfig;

const MAX_ABS_DIFF_TOL: f32 = 2e-2;
const COSINE_SIM_TOL: f32 = 0.999;

fn deterministic_signal(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + 11) % 89) as f32 - 44.0) * scale)
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

fn assert_all_exact(
    dx_ref: &[f32],
    grads_ref: &LayerGrads,
    dx_test: &[f32],
    grads_test: &LayerGrads,
    label: &str,
) {
    assert_exact(dx_ref, dx_test, &format!("{label}.dx"));
    assert_exact(&grads_ref.dw1, &grads_test.dw1, &format!("{label}.dw1"));
    assert_exact(&grads_ref.dw2, &grads_test.dw2, &format!("{label}.dw2"));
    assert_exact(&grads_ref.dw3, &grads_test.dw3, &format!("{label}.dw3"));
    assert_exact(&grads_ref.dwo, &grads_test.dwo, &format!("{label}.dwo"));
    assert_exact(&grads_ref.dwq, &grads_test.dwq, &format!("{label}.dwq"));
    assert_exact(&grads_ref.dwk, &grads_test.dwk, &format!("{label}.dwk"));
    assert_exact(&grads_ref.dwv, &grads_test.dwv, &format!("{label}.dwv"));
    assert_exact(
        &grads_ref.dgamma1,
        &grads_test.dgamma1,
        &format!("{label}.dgamma1"),
    );
    assert_exact(
        &grads_ref.dgamma2,
        &grads_test.dgamma2,
        &format!("{label}.dgamma2"),
    );
}

fn assert_all_close(
    dx_ref: &[f32],
    grads_ref: &LayerGrads,
    dx_test: &[f32],
    grads_test: &LayerGrads,
    label: &str,
) {
    assert_close(dx_ref, dx_test, &format!("{label}.dx"));
    assert_close(&grads_ref.dw1, &grads_test.dw1, &format!("{label}.dw1"));
    assert_close(&grads_ref.dw2, &grads_test.dw2, &format!("{label}.dw2"));
    assert_close(&grads_ref.dw3, &grads_test.dw3, &format!("{label}.dw3"));
    assert_close(&grads_ref.dwo, &grads_test.dwo, &format!("{label}.dwo"));
    assert_close(&grads_ref.dwq, &grads_test.dwq, &format!("{label}.dwq"));
    assert_close(&grads_ref.dwk, &grads_test.dwk, &format!("{label}.dwk"));
    assert_close(&grads_ref.dwv, &grads_test.dwv, &format!("{label}.dwv"));
    assert_close(
        &grads_ref.dgamma1,
        &grads_test.dgamma1,
        &format!("{label}.dgamma1"),
    );
    assert_close(
        &grads_ref.dgamma2,
        &grads_test.dgamma2,
        &format!("{label}.dgamma2"),
    );
}

fn setup_case() -> (
    ModelConfig,
    CompiledKernels,
    LayerWeights,
    ForwardCache,
    Vec<f32>,
) {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x = deterministic_signal(cfg.dim * cfg.seq, 0.015);
    let dy = deterministic_signal(cfg.dim * cfg.seq, 0.004);
    let mut cache = ForwardCache::new(&cfg);
    let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
    layer::forward_into(&cfg, &kernels, &weights, &x, &mut cache, &mut x_next);
    (cfg, kernels, weights, cache, dy)
}

#[test]
fn sharded_attention_parallel_matches_serial_reference() {
    let (cfg, kernels, weights, cache, dy) = setup_case();

    let mut grads_serial = LayerGrads::zeros(&cfg);
    let mut ws_serial = BackwardWorkspace::new(&cfg);
    let mut dx_serial = vec![0.0f32; cfg.dim * cfg.seq];
    let mut runtime_serial = ShardedAttentionBackwardRuntime::compile(&cfg, 2)
        .expect("serial sharded attention runtime");
    layer::backward_into_sharded_attn_serial(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_serial,
        &mut ws_serial,
        &mut dx_serial,
        Some(&mut runtime_serial),
        None,
    );

    let mut grads_parallel = LayerGrads::zeros(&cfg);
    let mut ws_parallel = BackwardWorkspace::new(&cfg);
    let mut dx_parallel = vec![0.0f32; cfg.dim * cfg.seq];
    let mut runtime_parallel = ShardedAttentionBackwardRuntime::compile(&cfg, 2)
        .expect("parallel sharded attention runtime");
    layer::backward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_parallel,
        &mut ws_parallel,
        &mut dx_parallel,
        Some(&mut runtime_parallel),
        None,
    );

    assert_all_exact(
        &dx_serial,
        &grads_serial,
        &dx_parallel,
        &grads_parallel,
        "parallel_vs_serial",
    );
}

#[test]
fn sharded_attention_matches_non_sharded_within_tolerance() {
    let (cfg, kernels, weights, cache, dy) = setup_case();

    let mut grads_baseline = LayerGrads::zeros(&cfg);
    let mut ws_baseline = BackwardWorkspace::new(&cfg);
    let mut dx_baseline = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_baseline,
        &mut ws_baseline,
        &mut dx_baseline,
    );

    let mut grads_sharded = LayerGrads::zeros(&cfg);
    let mut ws_sharded = BackwardWorkspace::new(&cfg);
    let mut dx_sharded = vec![0.0f32; cfg.dim * cfg.seq];
    let mut runtime =
        ShardedAttentionBackwardRuntime::compile(&cfg, 2).expect("sharded attention runtime");
    layer::backward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads_sharded,
        &mut ws_sharded,
        &mut dx_sharded,
        Some(&mut runtime),
        None,
    );

    assert_all_close(
        &dx_baseline,
        &grads_baseline,
        &dx_sharded,
        &grads_sharded,
        "sharded_vs_baseline",
    );
}

#[test]
fn sharded_attention_parallel_is_idempotent() {
    let (cfg, kernels, weights, cache, dy) = setup_case();
    let mut runtime =
        ShardedAttentionBackwardRuntime::compile(&cfg, 2).expect("sharded attention runtime");

    let mut grads1 = LayerGrads::zeros(&cfg);
    let mut ws1 = BackwardWorkspace::new(&cfg);
    let mut dx1 = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads1,
        &mut ws1,
        &mut dx1,
        Some(&mut runtime),
        None,
    );

    let mut grads2 = LayerGrads::zeros(&cfg);
    let mut ws2 = BackwardWorkspace::new(&cfg);
    let mut dx2 = vec![0.0f32; cfg.dim * cfg.seq];
    layer::backward_into_with_training_ffn(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &dy,
        &mut grads2,
        &mut ws2,
        &mut dx2,
        Some(&mut runtime),
        None,
    );

    assert_all_exact(&dx1, &grads1, &dx2, &grads2, "idempotent");
}
