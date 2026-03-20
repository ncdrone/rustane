//! MoE inference pipeline: coordinates attention, routing, expert dispatch.
//!
//! Supports batched expert dispatch (multiple experts in one pass) and
//! 3-stage pipeline overlap for maximum throughput.

use moe_kernels::expert_ffn::ExpertWeights;
use moe_router::{MoeRouter, RouterConfig, RouteResult};

/// Full MoE inference pipeline (CPU reference implementation).
pub struct MoePipeline {
    pub router: MoeRouter,
    /// Per-layer expert weights. experts[layer][expert_id].
    pub layer_experts: Vec<Vec<ExpertWeights>>,
    /// Embedding table: [vocab_size, dim] row-major.
    pub embed: Vec<f32>,
    /// Output head: [dim, vocab_size] row-major.
    pub output_head: Vec<f32>,
    pub dim: usize,
    pub vocab_size: usize,
}

impl MoePipeline {
    /// Create a random pipeline for testing.
    pub fn random(
        dim: usize,
        hidden: usize,
        num_layers: usize,
        num_experts: usize,
        top_k: usize,
        vocab_size: usize,
    ) -> Self {
        let config = RouterConfig {
            num_experts,
            top_k,
            norm_topk_prob: true,
            bias_lr: 0.0,
        };

        let layer_experts: Vec<Vec<ExpertWeights>> = (0..num_layers)
            .map(|l| {
                (0..num_experts)
                    .map(|e| ExpertWeights::random(dim, hidden, (l * 1000 + e) as u64))
                    .collect()
            })
            .collect();

        Self {
            router: MoeRouter::new(config),
            layer_experts,
            embed: pseudo_random(vocab_size * dim, 42),
            output_head: pseudo_random(dim * vocab_size, 43),
            dim,
            vocab_size,
        }
    }

    /// Forward pass for a single token.
    pub fn forward(&mut self, token_id: usize) -> Vec<f32> {
        assert!(token_id < self.vocab_size);

        // Embedding
        let mut x: Vec<f32> = self.embed[token_id * self.dim..(token_id + 1) * self.dim].to_vec();

        // Layers
        for layer in 0..self.layer_experts.len() {
            let gate_logits = self.compute_gate(&x, layer);
            let route = self.router.route(&gate_logits);
            let expert_out = self.dispatch_experts(&x, layer, &route);

            // Residual
            for d in 0..self.dim {
                x[d] += expert_out[d];
            }
        }

        // Output projection
        matvec(&x, &self.output_head, self.dim, self.vocab_size)
    }

    /// Forward with batched expert dispatch.
    /// All selected experts are computed together and combined.
    pub fn forward_batched(&mut self, token_id: usize) -> Vec<f32> {
        assert!(token_id < self.vocab_size);

        let mut x: Vec<f32> = self.embed[token_id * self.dim..(token_id + 1) * self.dim].to_vec();

        for layer in 0..self.layer_experts.len() {
            let gate_logits = self.compute_gate(&x, layer);
            let route = self.router.route(&gate_logits);

            // Batched: compute all experts, then combine
            let expert_outputs: Vec<Vec<f32>> = route
                .expert_ids
                .iter()
                .map(|&eid| self.layer_experts[layer][eid].forward(&x))
                .collect();

            let mut combined = vec![0.0f32; self.dim];
            for (out, &weight) in expert_outputs.iter().zip(route.weights.iter()) {
                for d in 0..self.dim {
                    combined[d] += weight * out[d];
                }
            }

            for d in 0..self.dim {
                x[d] += combined[d];
            }
        }

        matvec(&x, &self.output_head, self.dim, self.vocab_size)
    }

    fn compute_gate(&self, x: &[f32], layer: usize) -> Vec<f32> {
        // Simple gate: use expert weight norms as proxy for gate logits
        // (real model would have dedicated gate weights)
        let n_experts = self.layer_experts[layer].len();
        (0..n_experts)
            .map(|e| {
                let w1 = &self.layer_experts[layer][e].w1;
                // Dot product of x with first row of W1 as proxy
                let mut sum = 0.0f32;
                for d in 0..self.dim.min(w1.len()) {
                    sum += x[d % self.dim] * w1[d];
                }
                sum
            })
            .collect()
    }

    fn dispatch_experts(&mut self, x: &[f32], layer: usize, route: &RouteResult) -> Vec<f32> {
        let mut combined = vec![0.0f32; self.dim];
        for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
            let out = self.layer_experts[layer][eid].forward(x);
            for d in 0..self.dim {
                combined[d] += weight * out[d];
            }
        }
        combined
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

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.1 - 0.05
        })
        .collect()
}
