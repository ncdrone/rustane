//! Metal-backed KV-cache decode for checkpoint-backed inference.

use crate::cpu::{rmsnorm, vdsp};
use crate::full_model::ModelWeights;
use crate::layer::ForwardCache;
use crate::model::ModelConfig;
use metal_decode::{
    Config as MetalConfig, FinalHeadInit as MetalFinalHeadInit, LayerInit as MetalLayerInit,
    MAX_GPU_TOPK as METAL_MAX_GPU_TOPK, Model as MetalDecodeModel,
};

pub const MAX_GPU_TOPK: usize = METAL_MAX_GPU_TOPK;

pub struct DecodeContext {
    metal: MetalDecodeModel,
    ws: DecodeWorkspace,
}

struct DecodeWorkspace {
    layer_out: Vec<f32>,
    x_final: Vec<f32>,
    rms_inv_final: Vec<f32>,
    logits: Vec<f32>,
    block_maxes: Vec<f32>,
    block_sums: Vec<f32>,
    block_logits: Vec<f32>,
    topk_vals: Vec<f32>,
    topk_idxs: Vec<u32>,
    merged_vals: Vec<f32>,
    merged_idxs: Vec<u32>,
}

impl DecodeContext {
    pub fn new(cfg: &ModelConfig, weights: &ModelWeights) -> Self {
        let metal_cfg = MetalConfig {
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            hd: cfg.hd,
            seq: cfg.seq,
            nlayers: cfg.nlayers,
            vocab: cfg.vocab,
            q_dim: cfg.q_dim,
            kv_dim: cfg.kv_dim,
            gqa_ratio: cfg.gqa_ratio,
        };
        let metal = MetalDecodeModel::new(
            &metal_cfg,
            weights.layers.iter().map(|layer| MetalLayerInit {
                wq: &layer.wq,
                wk: &layer.wk,
                wv: &layer.wv,
                wo: &layer.wo,
                w1: &layer.w1,
                w3: &layer.w3,
                w2: &layer.w2,
                gamma1: &layer.gamma1,
                gamma2: &layer.gamma2,
            }),
            MetalFinalHeadInit {
                gamma_final: &weights.gamma_final,
                embed: &weights.embed,
            },
        )
        .expect("failed to initialize metal decode");
        Self {
            metal,
            ws: DecodeWorkspace::new(cfg),
        }
    }

    pub fn seed_from_forward_caches(
        &mut self,
        cfg: &ModelConfig,
        caches: &[ForwardCache],
        live_len: usize,
    ) {
        let live_len = live_len.min(cfg.seq);
        self.metal.set_len(live_len);
        for (layer_idx, cache) in caches.iter().enumerate() {
            self.metal.seed_layer_cache_from_channel_first(
                layer_idx,
                live_len,
                &cache.k_rope,
                &cache.v,
            );
        }
    }

    pub fn decode_next_logits<'a>(
        &'a mut self,
        cfg: &ModelConfig,
        weights: &ModelWeights,
        token: u32,
        softcap: f32,
    ) -> &'a [f32] {
        self.metal
            .decode_token_from_id(token, &mut self.ws.layer_out);

        rmsnorm::forward_channel_first(
            &self.ws.layer_out,
            &weights.gamma_final,
            &mut self.ws.x_final,
            &mut self.ws.rms_inv_final,
            cfg.dim,
            1,
        );

        self.ws.logits.fill(0.0);
        vdsp::sgemm_at(
            &self.ws.x_final,
            1,
            cfg.dim,
            &weights.embed,
            cfg.vocab,
            &mut self.ws.logits,
        );

        if softcap > 0.0 {
            vdsp::sscal(&mut self.ws.logits, 1.0 / softcap);
            vdsp::tanhf_inplace(&mut self.ws.logits);
            vdsp::sscal(&mut self.ws.logits, softcap);
        }

        &self.ws.logits
    }

    pub fn decode_next_greedy_token(&mut self, token: u32) -> u32 {
        self.metal.decode_token_argmax_from_id(token)
    }

    pub fn decode_next_topk_candidates<'a>(
        &'a mut self,
        cfg: &ModelConfig,
        token: u32,
        top_k: usize,
        softcap: f32,
    ) -> (&'a [u32], &'a [f32]) {
        let top_k = top_k.min(MAX_GPU_TOPK).min(cfg.vocab);
        let count = self.metal.decode_token_topk_from_id(
            token,
            top_k,
            &mut self.ws.topk_vals,
            &mut self.ws.topk_idxs,
        );
        let merged_len = merge_topk_candidates(
            &self.ws.topk_vals[..count],
            &self.ws.topk_idxs[..count],
            top_k,
            &mut self.ws.merged_vals,
            &mut self.ws.merged_idxs,
        );

        if softcap > 0.0 {
            for logit in &mut self.ws.merged_vals[..merged_len] {
                *logit = softcap * (*logit / softcap).tanh();
            }
        }

        (
            &self.ws.merged_idxs[..merged_len],
            &self.ws.merged_vals[..merged_len],
        )
    }

    pub fn decode_next_fullvocab_block_stats<'a>(
        &'a mut self,
        _cfg: &ModelConfig,
        token: u32,
        temperature: f32,
        softcap: f32,
    ) -> (&'a [f32], &'a [f32]) {
        let inv_temperature = 1.0 / temperature.max(f32::MIN_POSITIVE);
        let count = self.metal.decode_token_block_stats_from_id(
            token,
            inv_temperature,
            softcap,
            &mut self.ws.block_maxes,
            &mut self.ws.block_sums,
        );
        (&self.ws.block_maxes[..count], &self.ws.block_sums[..count])
    }

    pub fn materialize_current_logits_block<'a>(
        &'a mut self,
        _cfg: &ModelConfig,
        block_idx: usize,
        softcap: f32,
    ) -> &'a [f32] {
        let valid = self.metal.materialize_current_logits_block(
            block_idx,
            softcap,
            &mut self.ws.block_logits,
        );
        &self.ws.block_logits[..valid]
    }
}

impl DecodeWorkspace {
    fn new(cfg: &ModelConfig) -> Self {
        Self {
            layer_out: vec![0.0; cfg.dim],
            x_final: vec![0.0; cfg.dim],
            rms_inv_final: vec![0.0; 1],
            logits: vec![0.0; cfg.vocab],
            block_maxes: vec![0.0; cfg.vocab.div_ceil(256)],
            block_sums: vec![0.0; cfg.vocab.div_ceil(256)],
            block_logits: vec![0.0; 256],
            topk_vals: vec![0.0; cfg.vocab],
            topk_idxs: vec![u32::MAX; cfg.vocab],
            merged_vals: vec![f32::NEG_INFINITY; MAX_GPU_TOPK],
            merged_idxs: vec![u32::MAX; MAX_GPU_TOPK],
        }
    }
}

fn merge_topk_candidates(
    vals: &[f32],
    idxs: &[u32],
    top_k: usize,
    out_vals: &mut [f32],
    out_idxs: &mut [u32],
) -> usize {
    out_vals[..top_k].fill(f32::NEG_INFINITY);
    out_idxs[..top_k].fill(u32::MAX);

    for (&val, &idx) in vals.iter().zip(idxs.iter()) {
        if idx == u32::MAX || !val.is_finite() {
            continue;
        }
        let mut insert_at = None;
        for slot in 0..top_k {
            if val > out_vals[slot] || (val == out_vals[slot] && idx < out_idxs[slot]) {
                insert_at = Some(slot);
                break;
            }
        }
        if let Some(slot) = insert_at {
            for i in (slot + 1..top_k).rev() {
                out_vals[i] = out_vals[i - 1];
                out_idxs[i] = out_idxs[i - 1];
            }
            out_vals[slot] = val;
            out_idxs[slot] = idx;
        }
    }

    out_idxs[..top_k]
        .iter()
        .position(|&idx| idx == u32::MAX)
        .unwrap_or(top_k)
}
