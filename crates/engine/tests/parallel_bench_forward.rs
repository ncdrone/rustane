use engine::full_model::{ModelForwardWorkspace, ModelWeights, TrainConfig};
use engine::model::ModelConfig;
use engine::parallel_bench::{ParallelBenchRequest, ParallelBenchRunner, ShardPolicy};

const MAX_ABS_DIFF_TOL: f32 = 2e-2;
const MEAN_ABS_DIFF_TOL: f32 = 5e-3;
const COSINE_SIM_TOL: f32 = 0.999;
const LOSS_ABS_DIFF_TOL: f32 = 1e-4;

fn compare_logits_and_loss(
    actual_logits: &[f32],
    actual_loss: f32,
    expected_logits: &[f32],
    expected_loss: f32,
) {
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
    let mean_abs_diff = sum_abs / actual_logits.len() as f32;
    let loss_abs_diff = (actual_loss - expected_loss).abs();

    assert!(
        max_abs <= MAX_ABS_DIFF_TOL,
        "max abs diff too large: {max_abs}"
    );
    assert!(
        mean_abs_diff <= MEAN_ABS_DIFF_TOL,
        "mean abs diff too large: {mean_abs_diff}"
    );
    assert!(
        cosine_similarity >= COSINE_SIM_TOL,
        "cosine similarity too small: {cosine_similarity}"
    );
    assert!(
        loss_abs_diff <= LOSS_ABS_DIFF_TOL,
        "loss abs diff too large: {loss_abs_diff}"
    );
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

fn run_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    request: ParallelBenchRequest,
) -> (Vec<f32>, f32) {
    let (mut runner, _) =
        ParallelBenchRunner::compile(cfg, request, true).expect("compile parallel bench runner");
    let mut ws = ModelForwardWorkspace::new_lean(cfg);
    let (tokens, targets) = deterministic_tokens(cfg);
    let softcap = TrainConfig::default().softcap;
    let loss = runner.forward_loss(weights, &tokens, &targets, softcap, &mut ws);
    (ws.logits.clone(), loss)
}

#[test]
#[ignore]
fn parallel_bench_modes_match_baseline() {
    let cfg = ModelConfig::gpt_karpathy();
    let weights = ModelWeights::random(&cfg);

    let (baseline_logits, baseline_loss) = run_mode(
        &cfg,
        &weights,
        ParallelBenchRequest {
            attn_request: None,
            ffn_request: None,
            policy: ShardPolicy::FailFast,
        },
    );

    let modes = [
        ParallelBenchRequest {
            attn_request: Some(2),
            ffn_request: None,
            policy: ShardPolicy::FailFast,
        },
        ParallelBenchRequest {
            attn_request: None,
            ffn_request: Some(4),
            policy: ShardPolicy::FailFast,
        },
        ParallelBenchRequest {
            attn_request: Some(2),
            ffn_request: Some(4),
            policy: ShardPolicy::FailFast,
        },
    ];

    for request in modes {
        let (logits, loss) = run_mode(&cfg, &weights, request);
        compare_logits_and_loss(&logits, loss, &baseline_logits, baseline_loss);
    }
}
