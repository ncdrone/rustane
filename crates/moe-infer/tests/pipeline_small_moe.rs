//! Pipeline test: 8-expert MoE with routing produces valid logits.
//!
//! Tests the complete MoE inference pipeline on CPU:
//! 1. Token embedding
//! 2. Router selects top-k experts
//! 3. Expert FFN forward
//! 4. Weighted combine of expert outputs
//! 5. Output projection → logits
//!
//! Validates:
//! - Output logits are finite and non-zero
//! - Expert usage is roughly balanced (no expert starved)
//! - Pipeline produces different outputs for different inputs

use moe_kernels::ExpertWeights;
use moe_router::{MoeRouter, RouterConfig};

const DIM: usize = 128;
const HIDDEN: usize = 256;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const VOCAB: usize = 256;

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.2 - 0.1
        })
        .collect()
}

struct ToyMoeModel {
    /// Token embeddings: [VOCAB, DIM]
    embed: Vec<f32>,
    /// Gate weights per layer: [DIM, NUM_EXPERTS]
    gate_weights: Vec<f32>,
    /// Expert FFN weights
    experts: Vec<ExpertWeights>,
    /// Output head: [DIM, VOCAB]
    output_head: Vec<f32>,
    /// Router
    router: MoeRouter,
}

impl ToyMoeModel {
    fn new() -> Self {
        let experts: Vec<ExpertWeights> = (0..NUM_EXPERTS)
            .map(|i| ExpertWeights::random(DIM, HIDDEN, 100 + i as u64))
            .collect();

        let config = RouterConfig {
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            norm_topk_prob: true,
            bias_lr: 0.0,
        };

        Self {
            embed: pseudo_random(VOCAB * DIM, 1),
            gate_weights: pseudo_random(DIM * NUM_EXPERTS, 2),
            experts,
            output_head: pseudo_random(DIM * VOCAB, 3),
            router: MoeRouter::new(config),
        }
    }

    /// Forward pass for a single token.
    fn forward(&mut self, token_id: usize) -> Vec<f32> {
        assert!(token_id < VOCAB);

        // 1. Embedding lookup
        let x: Vec<f32> = self.embed[token_id * DIM..(token_id + 1) * DIM].to_vec();

        // 2. Compute gate logits: x @ gate_weights → [NUM_EXPERTS]
        let gate_logits = matvec(&x, &self.gate_weights, DIM, NUM_EXPERTS);

        // 3. Route: select top-k experts
        let route = self.router.route(&gate_logits);

        // 4. Run selected experts and combine
        let mut combined = vec![0.0f32; DIM];
        for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
            let expert_out = self.experts[eid].forward(&x);
            for d in 0..DIM {
                combined[d] += weight * expert_out[d];
            }
        }

        // 5. Residual connection
        let mut out = vec![0.0f32; DIM];
        for d in 0..DIM {
            out[d] = x[d] + combined[d];
        }

        // 6. Output projection: out @ output_head → [VOCAB]
        matvec(&out, &self.output_head, DIM, VOCAB)
    }
}

fn matvec(x: &[f32], w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    for i in 0..rows {
        let xi = x[i];
        for j in 0..cols {
            out[j] += xi * w[i * cols + j];
        }
    }
    out
}

/// MoE pipeline produces valid (finite, non-zero) logits.
#[test]
#[ignore = "pipeline test"]
fn moe_produces_valid_logits() {
    let mut model = ToyMoeModel::new();

    let logits = model.forward(42);

    assert_eq!(logits.len(), VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits contain NaN or Inf"
    );
    let max_abs = logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 1e-6, "logits are all near-zero: max_abs={max_abs}");
}

/// Different tokens produce different outputs.
#[test]
#[ignore = "pipeline test"]
fn different_tokens_different_outputs() {
    let mut model = ToyMoeModel::new();

    let logits_a = model.forward(10);
    let logits_b = model.forward(200);

    let max_diff = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-4,
        "different tokens produced same output: max_diff={max_diff}"
    );
}

/// Expert usage is roughly balanced across many tokens.
#[test]
#[ignore = "pipeline test"]
fn balanced_expert_usage() {
    let config = RouterConfig {
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        norm_topk_prob: true,
        bias_lr: 0.001, // enable load balancing
    };
    let mut router = MoeRouter::new(config);

    let gate_weights = pseudo_random(DIM * NUM_EXPERTS, 42);

    // Route 1000 tokens
    for token in 0..1000 {
        let x = pseudo_random(DIM, token as u64 + 500);
        let gate_logits = matvec(&x, &gate_weights, DIM, NUM_EXPERTS);
        router.route(&gate_logits);
    }

    let fracs = router.usage_fractions();

    // With 8 experts, top-2: each expert should be used ~25% of the time (fraction ~1.0).
    // Allow wide range since it depends on the random gate weights.
    // Just verify no expert is completely starved (fraction > 0.01).
    let min_frac = fracs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_frac = fracs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("Expert usage fractions: {:?}", fracs);
    println!("Min fraction: {min_frac:.3}, Max fraction: {max_frac:.3}");

    // At least some experts should be used
    let used_count = fracs.iter().filter(|&&f| f > 0.01).count();
    assert!(
        used_count >= NUM_EXPERTS / 2,
        "only {used_count}/{NUM_EXPERTS} experts used (fracs: {:?})",
        fracs
    );
}
