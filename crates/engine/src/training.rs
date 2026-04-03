//! Training step: forward → loss → backward → weight update.
//!
//! Single-layer training loop for Phase 3 validation.
//! Exit criteria: loss decreases over 10 steps.

use crate::cpu::adam::{self, AdamConfig};
use crate::layer::{self, CompiledKernels, LayerGrads, LayerWeights};
use crate::model::ModelConfig;

/// Adam optimizer state for one layer's weights.
pub struct LayerOptState {
    pub m_wq: Vec<f32>,
    pub v_wq: Vec<f32>,
    pub m_wk: Vec<f32>,
    pub v_wk: Vec<f32>,
    pub m_wv: Vec<f32>,
    pub v_wv: Vec<f32>,
    pub m_wo: Vec<f32>,
    pub v_wo: Vec<f32>,
    pub m_w1: Vec<f32>,
    pub v_w1: Vec<f32>,
    pub m_w3: Vec<f32>,
    pub v_w3: Vec<f32>,
    pub m_w2: Vec<f32>,
    pub v_w2: Vec<f32>,
    pub m_gamma1: Vec<f32>,
    pub v_gamma1: Vec<f32>,
    pub m_gamma2: Vec<f32>,
    pub v_gamma2: Vec<f32>,
}

impl LayerOptState {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            m_wq: vec![0.0; cfg.dim * cfg.q_dim],
            v_wq: vec![0.0; cfg.dim * cfg.q_dim],
            m_wk: vec![0.0; cfg.dim * cfg.kv_dim],
            v_wk: vec![0.0; cfg.dim * cfg.kv_dim],
            m_wv: vec![0.0; cfg.dim * cfg.kv_dim],
            v_wv: vec![0.0; cfg.dim * cfg.kv_dim],
            m_wo: vec![0.0; cfg.q_dim * cfg.dim],
            v_wo: vec![0.0; cfg.q_dim * cfg.dim],
            m_w1: vec![0.0; cfg.dim * cfg.hidden],
            v_w1: vec![0.0; cfg.dim * cfg.hidden],
            m_w3: vec![0.0; cfg.dim * cfg.hidden],
            v_w3: vec![0.0; cfg.dim * cfg.hidden],
            m_w2: vec![0.0; cfg.dim * cfg.hidden],
            v_w2: vec![0.0; cfg.dim * cfg.hidden],
            m_gamma1: vec![0.0; cfg.dim],
            v_gamma1: vec![0.0; cfg.dim],
            m_gamma2: vec![0.0; cfg.dim],
            v_gamma2: vec![0.0; cfg.dim],
        }
    }
}

/// Apply Adam update to all weights in a layer.
pub fn update_weights(
    weights: &mut LayerWeights,
    grads: &LayerGrads,
    opt: &mut LayerOptState,
    t: u32,
    adam_cfg: &AdamConfig,
) {
    adam::step(
        &mut weights.wq,
        &grads.dwq,
        &mut opt.m_wq,
        &mut opt.v_wq,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.wk,
        &grads.dwk,
        &mut opt.m_wk,
        &mut opt.v_wk,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.wv,
        &grads.dwv,
        &mut opt.m_wv,
        &mut opt.v_wv,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.wo,
        &grads.dwo,
        &mut opt.m_wo,
        &mut opt.v_wo,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.w1,
        &grads.dw1,
        &mut opt.m_w1,
        &mut opt.v_w1,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.w3,
        &grads.dw3,
        &mut opt.m_w3,
        &mut opt.v_w3,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.w2,
        &grads.dw2,
        &mut opt.m_w2,
        &mut opt.v_w2,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.gamma1,
        &grads.dgamma1,
        &mut opt.m_gamma1,
        &mut opt.v_gamma1,
        t,
        adam_cfg,
    );
    adam::step(
        &mut weights.gamma2,
        &grads.dgamma2,
        &mut opt.m_gamma2,
        &mut opt.v_gamma2,
        t,
        adam_cfg,
    );
}

/// Run a single training step: forward → loss → backward → update.
/// Uses MSE loss against a zero target: loss = mean(x_next²).
/// Returns the loss value.
pub fn train_step(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &mut LayerWeights,
    grads: &mut LayerGrads,
    opt: &mut LayerOptState,
    x: &[f32],
    t: u32,
    adam_cfg: &AdamConfig,
) -> f32 {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let n = dim * seq;

    // Forward
    let (x_next, cache) = layer::forward(cfg, kernels, weights, x);

    // MSE loss = mean(x_next²)
    let loss: f32 = x_next.iter().map(|v| v * v).sum::<f32>() / n as f32;

    // d(loss)/d(x_next) = 2 * x_next / n
    let mut dy = vec![0.0f32; n];
    let scale = 2.0 / n as f32;
    for i in 0..n {
        dy[i] = x_next[i] * scale;
    }

    // Backward
    grads.zero_out();
    let mut bwd_ws = layer::BackwardWorkspace::new(cfg);
    let _dx = layer::backward(cfg, kernels, weights, &cache, &dy, grads, &mut bwd_ws);

    // Update weights
    update_weights(weights, grads, opt, t, adam_cfg);

    loss
}
