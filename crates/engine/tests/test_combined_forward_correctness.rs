use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    TrainConfig, TrainingParallelOptions,
};
use engine::layer::{self, CombinedForwardRuntime, CompiledKernels, ForwardCache, LayerWeights};
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use engine::sharding::{
    ResolvedBackwardMode, ResolvedForwardMode, ShardPolicy, ShardRequest, resolve_modes,
};
use std::sync::{Mutex, OnceLock};

const MAX_ABS_DIFF_TOL: f32 = 2e-2;
const COSINE_SIM_TOL: f32 = 0.999;

fn deterministic_signal(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 13 + 19) % 113) as f32 - 56.0) * scale)
        .collect()
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

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
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

fn compile_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("compile lock")
}

#[test]
fn combined_forward_layer_modes_match_baseline_within_tolerance() {
    let _guard = compile_lock();
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x = deterministic_signal(cfg.dim * cfg.seq, 0.01);

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

    for (mode, label) in [
        (
            ResolvedForwardMode::AttentionOnly { attn_shards: 2 },
            "attn2",
        ),
        (ResolvedForwardMode::FfnOnly { ffn_shards: 4 }, "ffn4"),
        (
            ResolvedForwardMode::AttentionFfn {
                attn_shards: 2,
                ffn_shards: 4,
            },
            "attn2_ffn4",
        ),
    ] {
        let mut runtime =
            CombinedForwardRuntime::compile(&cfg, mode).expect("compile combined runtime");
        let mut cache = ForwardCache::new(&cfg);
        let mut x_next = vec![0.0f32; cfg.dim * cfg.seq];
        layer::forward_into_combined(
            &cfg,
            &kernels,
            &weights,
            &x,
            &mut cache,
            &mut x_next,
            &mut runtime,
        );

        assert_close(&baseline_x_next, &x_next, &format!("{label}.x_next"));
        assert_cache_close(&baseline_cache, &cache, label);
    }
}

#[test]
fn forward_only_workspace_reuse_respects_mode_dispatch_and_w2_cache() {
    let _guard = compile_lock();
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile_forward_only(&cfg);
    let weights = ModelWeights::random(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let mut ws = ModelForwardWorkspace::new_lean(&cfg);
    let mut fresh_ws = ModelForwardWorkspace::new_lean(&cfg);

    let baseline_loss_1 = full_model::forward_only_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut ws,
        ResolvedForwardMode::Baseline,
    );
    assert!(
        ws.combined_forward.is_none(),
        "baseline path should not keep combined runtime"
    );

    let ffn_loss = full_model::forward_only_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut ws,
        ResolvedForwardMode::FfnOnly { ffn_shards: 4 },
    );
    assert!(ffn_loss.is_finite());
    assert!(
        ws.combined_forward.is_some(),
        "ffn-only path should allocate combined runtime"
    );
    assert!(
        ws.caches.iter().any(|cache| cache.w2t_generation == 0),
        "sharded FFN run should populate cached W2 transpose generations",
    );

    let baseline_loss_2 = full_model::forward_only_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut ws,
        ResolvedForwardMode::Baseline,
    );
    let fresh_baseline = full_model::forward_only_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut fresh_ws,
        ResolvedForwardMode::Baseline,
    );
    assert_eq!(baseline_loss_2.to_bits(), fresh_baseline.to_bits());
    assert_eq!(baseline_loss_1.to_bits(), fresh_baseline.to_bits());
    assert!(
        ws.combined_forward.is_none(),
        "baseline dispatch should clear combined runtime"
    );
    assert!(
        ws.caches.iter().any(|cache| cache.w2t_generation == 0),
        "mode transitions should not clear cached W2 transposes",
    );

    let backward_only = TrainingParallelOptions {
        forward_mode: ResolvedForwardMode::Baseline,
        backward_mode: ResolvedBackwardMode::FfnOnly { ffn_shards: 4 },
        adjustments: Vec::new(),
    };
    let mut train_ws = ModelForwardWorkspace::new(&cfg);
    let train_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut train_ws,
        &backward_only,
    );
    assert!(train_loss.is_finite());
    assert!(
        train_ws.combined_forward.is_none(),
        "backward-only sharding must keep the forward path on the pipelined baseline"
    );

    let odd_cfg = ModelConfig {
        dim: 256,
        hidden: 768,
        heads: 1,
        kv_heads: 1,
        hd: 128,
        seq: 16,
        nlayers: 2,
        vocab: 256,
        q_dim: 128,
        kv_dim: 128,
        gqa_ratio: 1,
    };
    let resolution = resolve_modes(
        &odd_cfg,
        ShardRequest {
            attn_fwd_shards: Some(8),
            ..ShardRequest::default()
        },
        ShardPolicy::AutoAdjustNearest,
    )
    .expect("resolve modes");
    assert_eq!(resolution.forward, ResolvedForwardMode::Baseline);
}

#[test]
fn training_workspace_reuse_and_mismatched_modes_execute_independently() {
    let _guard = compile_lock();
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let (tokens, targets) = deterministic_tokens(&cfg);
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let metal_adam = MetalAdam::new().expect("Metal GPU required");
    let tc = TrainConfig {
        accum_steps: 1,
        total_steps: 3,
        ..TrainConfig::default()
    };
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let _baseline_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        &TrainingParallelOptions::disabled(),
    );

    let combined_options = TrainingParallelOptions {
        forward_mode: ResolvedForwardMode::AttentionFfn {
            attn_shards: 2,
            ffn_shards: 4,
        },
        backward_mode: ResolvedBackwardMode::Baseline,
        adjustments: Vec::new(),
    };
    let combined_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        &combined_options,
    );
    assert!(combined_loss.is_finite());
    let combined_mode = fwd_ws
        .combined_forward
        .as_ref()
        .map(|runtime| runtime.mode())
        .expect("combined runtime");
    assert_eq!(combined_mode, combined_options.forward_mode);

    let ffn_only_options = TrainingParallelOptions {
        forward_mode: ResolvedForwardMode::FfnOnly { ffn_shards: 4 },
        backward_mode: ResolvedBackwardMode::Baseline,
        adjustments: Vec::new(),
    };
    let ffn_only_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        &ffn_only_options,
    );
    assert!(ffn_only_loss.is_finite());
    let ffn_only_mode = fwd_ws
        .combined_forward
        .as_ref()
        .map(|runtime| runtime.mode())
        .expect("combined runtime");
    assert_eq!(ffn_only_mode, ffn_only_options.forward_mode);

    let ffn_only_loss_repeat = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        &ffn_only_options,
    );
    let ffn_only_mode_repeat = fwd_ws
        .combined_forward
        .as_ref()
        .map(|runtime| runtime.mode())
        .expect("combined runtime");
    assert_eq!(ffn_only_mode_repeat, ffn_only_options.forward_mode);
    assert_eq!(ffn_only_loss.to_bits(), ffn_only_loss_repeat.to_bits());

    let mut fresh_ws = ModelForwardWorkspace::new(&cfg);
    let fresh_ffn_only_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fresh_ws,
        &ffn_only_options,
    );
    assert_eq!(ffn_only_loss.to_bits(), fresh_ffn_only_loss.to_bits());

    let mismatch_options = TrainingParallelOptions {
        forward_mode: ResolvedForwardMode::FfnOnly { ffn_shards: 4 },
        backward_mode: ResolvedBackwardMode::FfnOnly { ffn_shards: 2 },
        adjustments: Vec::new(),
    };
    let initial_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        &mismatch_options,
    );
    for step in 0..3u32 {
        grads.zero_out();
        let loss = full_model::forward_ws_with_options(
            &cfg,
            &kernels,
            &weights,
            &tokens,
            &targets,
            tc.softcap,
            &mut fwd_ws,
            &mismatch_options,
        );
        full_model::backward_ws_with_options(
            &cfg,
            &kernels,
            &weights,
            &fwd_ws,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
            &mismatch_options,
        );
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let combined_scale = if raw_norm * gsc > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights(
            &cfg,
            &mut weights,
            &grads,
            &mut opt,
            step + 1,
            lr,
            &tc,
            &metal_adam,
            combined_scale,
        );
        let _ = loss;
    }
    let final_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        tc.softcap,
        &mut fwd_ws,
        &mismatch_options,
    );
    assert!(
        final_loss < initial_loss,
        "mismatched forward/backward sharding should still train: initial={initial_loss} final={final_loss}"
    );
    assert!(
        bwd_ws.training_ffn_sharded.is_some(),
        "backward runtime should be present"
    );
    assert!(
        fwd_ws.combined_forward.is_some(),
        "forward runtime should be present"
    );
}
