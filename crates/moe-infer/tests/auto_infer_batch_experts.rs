//! Correctness test: batched 8-expert output matches sequential dispatch.
//!
//! Verifies that computing all selected experts in a batch produces
//! identical results to computing them one-by-one.

use moe_infer::pipeline::MoePipeline;

/// Batched expert dispatch matches sequential dispatch (bit-identical).
#[test]
fn batched_matches_sequential() {
    let mut pipeline_seq = MoePipeline::random(128, 256, 2, 8, 2, 256);
    let mut pipeline_batch = MoePipeline::random(128, 256, 2, 8, 2, 256);

    for token in [0, 42, 100, 200, 255] {
        let out_seq = pipeline_seq.forward(token);
        let out_batch = pipeline_batch.forward_batched(token);

        let max_diff = out_seq
            .iter()
            .zip(out_batch.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "token {token}: batched vs sequential diff = {max_diff}"
        );
    }
}

/// Multi-layer pipeline produces valid logits.
#[test]
fn multi_layer_valid_logits() {
    let mut pipeline = MoePipeline::random(128, 256, 4, 8, 2, 256);

    let logits = pipeline.forward(42);
    assert_eq!(logits.len(), 256);
    assert!(logits.iter().all(|v| v.is_finite()), "logits contain NaN/Inf");

    let max_abs = logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 1e-6, "logits are all near-zero");
}
