//! Metal-backed single-token decode for cached transformer inference.

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::ffi::c_void;
use std::ptr::NonNull;

const RMS_EPS: f32 = 1e-5;
pub const MAX_GPU_TOPK: usize = 32;

const SHADER_PREAMBLE: &str = r#"
#include <metal_stdlib>
using namespace metal;
"#;

#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,
    pub seq: usize,
    pub nlayers: usize,
    pub vocab: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub gqa_ratio: usize,
}

pub struct LayerInit<'a> {
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub w1: &'a [f32],
    pub w3: &'a [f32],
    pub w2: &'a [f32],
    pub gamma1: &'a [f32],
    pub gamma2: &'a [f32],
}

pub struct FinalHeadInit<'a> {
    pub gamma_final: &'a [f32],
    pub embed: &'a [f32],
}

struct LayerState {
    wq: Retained<ProtocolObject<dyn MTLBuffer>>,
    wk: Retained<ProtocolObject<dyn MTLBuffer>>,
    wv: Retained<ProtocolObject<dyn MTLBuffer>>,
    wo: Retained<ProtocolObject<dyn MTLBuffer>>,
    w1: Retained<ProtocolObject<dyn MTLBuffer>>,
    w3: Retained<ProtocolObject<dyn MTLBuffer>>,
    w2: Retained<ProtocolObject<dyn MTLBuffer>>,
    gamma1: Retained<ProtocolObject<dyn MTLBuffer>>,
    gamma2: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_cache: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_cache: Retained<ProtocolObject<dyn MTLBuffer>>,
}

struct Workspace {
    x_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    x_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    h1: Retained<ProtocolObject<dyn MTLBuffer>>,
    h3: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
}

struct Pipelines {
    embed_lookup: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rmsnorm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rope: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    append_kv: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    sdpa: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residual_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu_gate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logits_block_stats: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logits_block_materialize: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logits_topk_blocks: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logits_argmax_blocks: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logits_argmax_finalize: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

struct Matmuls {
    q_proj: Retained<MPSMatrixMultiplication>,
    kv_proj: Retained<MPSMatrixMultiplication>,
    wo_proj: Retained<MPSMatrixMultiplication>,
    w1_proj: Retained<MPSMatrixMultiplication>,
    w3_proj: Retained<MPSMatrixMultiplication>,
    w2_proj: Retained<MPSMatrixMultiplication>,
}

struct Descriptors {
    dim_vec: Retained<MPSMatrixDescriptor>,
    q_vec: Retained<MPSMatrixDescriptor>,
    kv_vec: Retained<MPSMatrixDescriptor>,
    hidden_vec: Retained<MPSMatrixDescriptor>,
    wq: Retained<MPSMatrixDescriptor>,
    wkv: Retained<MPSMatrixDescriptor>,
    wo: Retained<MPSMatrixDescriptor>,
    wh: Retained<MPSMatrixDescriptor>,
}

struct FinalHeadState {
    gamma_final: Retained<ProtocolObject<dyn MTLBuffer>>,
    embed: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_maxes: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_sums: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_logits: Retained<ProtocolObject<dyn MTLBuffer>>,
    topk_vals: Retained<ProtocolObject<dyn MTLBuffer>>,
    topk_idxs: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_vals: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_idxs: Retained<ProtocolObject<dyn MTLBuffer>>,
    out_idx: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub struct Model {
    cfg: Config,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Pipelines,
    mm: Matmuls,
    desc: Descriptors,
    layers: Vec<LayerState>,
    final_head: FinalHeadState,
    ws: Workspace,
    len: usize,
    alpha: f32,
    rms_threads: usize,
    attn_threads: usize,
    argmax_threads: usize,
    argmax_blocks: usize,
}

impl Model {
    pub fn new<'a>(
        cfg: &Config,
        layers: impl IntoIterator<Item = LayerInit<'a>>,
        final_head: FinalHeadInit<'a>,
    ) -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        let rms_threads = 256usize.min(max_pow2_threads(cfg.dim));
        let attn_threads = max_pow2_threads(cfg.hd).max(32);
        let argmax_threads = 256usize;
        let argmax_blocks = cfg.vocab.div_ceil(argmax_threads);
        let library = compile_library(
            &device,
            cfg,
            rms_threads,
            attn_threads,
            argmax_threads,
            argmax_blocks,
        )?;
        let pipes = Pipelines {
            embed_lookup: new_pipeline(&device, &library, "embed_lookup")?,
            rmsnorm: new_pipeline(&device, &library, "rmsnorm_single")?,
            rope: new_pipeline(&device, &library, "rope_inplace")?,
            append_kv: new_pipeline(&device, &library, "append_kv")?,
            sdpa: new_pipeline(&device, &library, "sdpa_causal")?,
            residual_add: new_pipeline(&device, &library, "residual_add")?,
            silu_gate: new_pipeline(&device, &library, "silu_gate")?,
            logits_block_stats: new_pipeline(&device, &library, "logits_block_stats")?,
            logits_block_materialize: new_pipeline(&device, &library, "logits_block_materialize")?,
            logits_topk_blocks: new_pipeline(&device, &library, "logits_topk_blocks")?,
            logits_argmax_blocks: new_pipeline(&device, &library, "logits_argmax_blocks")?,
            logits_argmax_finalize: new_pipeline(&device, &library, "logits_argmax_finalize")?,
        };
        let desc = Descriptors::new(cfg);
        let mm = Matmuls::new(cfg, &device);
        let layers: Vec<_> = layers
            .into_iter()
            .map(|layer| LayerState::new(cfg, &device, layer))
            .collect();
        if layers.len() != cfg.nlayers {
            return None;
        }
        let final_head = FinalHeadState::new(cfg, &device, argmax_blocks, final_head);
        let ws = Workspace::new(cfg, &device);
        Some(Self {
            cfg: cfg.clone(),
            queue,
            pipes,
            mm,
            desc,
            layers,
            final_head,
            ws,
            len: 0,
            alpha: 1.0 / (2.0 * cfg.nlayers as f32).sqrt(),
            rms_threads,
            attn_threads,
            argmax_threads,
            argmax_blocks,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn set_len(&mut self, len: usize) {
        self.len = len.min(self.cfg.seq);
    }

    pub fn seed_layer_cache_from_channel_first(
        &mut self,
        layer_idx: usize,
        live_len: usize,
        k_rope_cf: &[f32],
        v_cf: &[f32],
    ) {
        let layer = &self.layers[layer_idx];
        let live_len = live_len.min(self.cfg.seq);
        let mut k_row_major = vec![0.0f32; self.cfg.seq * self.cfg.kv_dim];
        let mut v_row_major = vec![0.0f32; self.cfg.seq * self.cfg.kv_dim];
        for pos in 0..live_len {
            let dst = pos * self.cfg.kv_dim;
            for d in 0..self.cfg.kv_dim {
                k_row_major[dst + d] = k_rope_cf[d * self.cfg.seq + pos];
                v_row_major[dst + d] = v_cf[d * self.cfg.seq + pos];
            }
        }
        upload_f32(&layer.k_cache, &k_row_major);
        upload_f32(&layer.v_cache, &v_row_major);
    }

    pub fn decode_token(&mut self, x_in: &[f32], x_out: &mut [f32]) {
        assert_eq!(x_in.len(), self.cfg.dim);
        assert_eq!(x_out.len(), self.cfg.dim);
        assert!(
            self.len < self.cfg.seq,
            "metal decode only supports lengths within cfg.seq"
        );

        upload_f32(&self.ws.x_a, x_in);
        let mut cur_is_a = true;
        let slot = self.len as u32;
        let attn_len = (self.len + 1) as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
        for layer in &self.layers {
            let (cur, next) = if cur_is_a {
                (&self.ws.x_a, &self.ws.x_b)
            } else {
                (&self.ws.x_b, &self.ws.x_a)
            };
            self.encode_layer(&cmd, layer, cur, next, slot, attn_len);
            cur_is_a = !cur_is_a;
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        let out_buf = if cur_is_a { &self.ws.x_a } else { &self.ws.x_b };
        download_f32(out_buf, x_out);
        self.len += 1;
    }

    pub fn decode_token_from_id(&mut self, token: u32, x_out: &mut [f32]) {
        assert_eq!(x_out.len(), self.cfg.dim);
        assert!(
            self.len < self.cfg.seq,
            "metal decode only supports lengths within cfg.seq"
        );
        assert!(
            (token as usize) < self.cfg.vocab,
            "token id must fit vocab for metal decode"
        );

        let mut cur_is_a = true;
        let slot = self.len as u32;
        let attn_len = (self.len + 1) as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
        self.encode_embed_lookup(&cmd, token, &self.ws.x_a);
        for layer in &self.layers {
            let (cur, next) = if cur_is_a {
                (&self.ws.x_a, &self.ws.x_b)
            } else {
                (&self.ws.x_b, &self.ws.x_a)
            };
            self.encode_layer(&cmd, layer, cur, next, slot, attn_len);
            cur_is_a = !cur_is_a;
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        let out_buf = if cur_is_a { &self.ws.x_a } else { &self.ws.x_b };
        download_f32(out_buf, x_out);
        self.len += 1;
    }

    pub fn decode_token_argmax(&mut self, x_in: &[f32]) -> u32 {
        assert_eq!(x_in.len(), self.cfg.dim);
        assert!(
            self.len < self.cfg.seq,
            "metal decode only supports lengths within cfg.seq"
        );

        upload_f32(&self.ws.x_a, x_in);
        let mut cur_is_a = true;
        let slot = self.len as u32;
        let attn_len = (self.len + 1) as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
        for layer in &self.layers {
            let (cur, next) = if cur_is_a {
                (&self.ws.x_a, &self.ws.x_b)
            } else {
                (&self.ws.x_b, &self.ws.x_a)
            };
            self.encode_layer(&cmd, layer, cur, next, slot, attn_len);
            cur_is_a = !cur_is_a;
        }

        let out_buf = if cur_is_a { &self.ws.x_a } else { &self.ws.x_b };
        self.encode_greedy_argmax(&cmd, out_buf);

        cmd.commit();
        cmd.waitUntilCompleted();

        let token = unsafe { *(self.final_head.out_idx.contents().as_ptr() as *const u32) };
        self.len += 1;
        token
    }

    pub fn decode_token_argmax_from_id(&mut self, token: u32) -> u32 {
        assert!(
            self.len < self.cfg.seq,
            "metal decode only supports lengths within cfg.seq"
        );
        assert!(
            (token as usize) < self.cfg.vocab,
            "token id must fit vocab for metal decode"
        );

        let mut cur_is_a = true;
        let slot = self.len as u32;
        let attn_len = (self.len + 1) as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
        self.encode_embed_lookup(&cmd, token, &self.ws.x_a);
        for layer in &self.layers {
            let (cur, next) = if cur_is_a {
                (&self.ws.x_a, &self.ws.x_b)
            } else {
                (&self.ws.x_b, &self.ws.x_a)
            };
            self.encode_layer(&cmd, layer, cur, next, slot, attn_len);
            cur_is_a = !cur_is_a;
        }

        let out_buf = if cur_is_a { &self.ws.x_a } else { &self.ws.x_b };
        self.encode_greedy_argmax(&cmd, out_buf);

        cmd.commit();
        cmd.waitUntilCompleted();

        let next = unsafe { *(self.final_head.out_idx.contents().as_ptr() as *const u32) };
        self.len += 1;
        next
    }

    pub fn decode_token_topk_from_id(
        &mut self,
        token: u32,
        top_k: usize,
        out_vals: &mut [f32],
        out_idxs: &mut [u32],
    ) -> usize {
        assert!(
            self.len < self.cfg.seq,
            "metal decode only supports lengths within cfg.seq"
        );
        assert!(
            (token as usize) < self.cfg.vocab,
            "token id must fit vocab for metal decode"
        );
        assert!(top_k > 0, "top_k must be positive");
        assert!(
            top_k <= MAX_GPU_TOPK,
            "top_k exceeds metal decode top-k capacity"
        );

        let count = self.argmax_blocks * top_k;
        assert!(
            out_vals.len() >= count,
            "out_vals too small for top-k candidates"
        );
        assert!(
            out_idxs.len() >= count,
            "out_idxs too small for top-k candidates"
        );

        let mut cur_is_a = true;
        let slot = self.len as u32;
        let attn_len = (self.len + 1) as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
        self.encode_embed_lookup(&cmd, token, &self.ws.x_a);
        for layer in &self.layers {
            let (cur, next) = if cur_is_a {
                (&self.ws.x_a, &self.ws.x_b)
            } else {
                (&self.ws.x_b, &self.ws.x_a)
            };
            self.encode_layer(&cmd, layer, cur, next, slot, attn_len);
            cur_is_a = !cur_is_a;
        }

        let out_buf = if cur_is_a { &self.ws.x_a } else { &self.ws.x_b };
        self.encode_topk_candidates(&cmd, out_buf, top_k as u32);

        cmd.commit();
        cmd.waitUntilCompleted();

        download_f32(&self.final_head.topk_vals, &mut out_vals[..count]);
        download_u32(&self.final_head.topk_idxs, &mut out_idxs[..count]);
        self.len += 1;
        count
    }

    pub fn decode_token_block_stats_from_id(
        &mut self,
        token: u32,
        inv_temperature: f32,
        softcap: f32,
        out_maxes: &mut [f32],
        out_sums: &mut [f32],
    ) -> usize {
        assert!(
            self.len < self.cfg.seq,
            "metal decode only supports lengths within cfg.seq"
        );
        assert!(
            (token as usize) < self.cfg.vocab,
            "token id must fit vocab for metal decode"
        );
        assert!(out_maxes.len() >= self.argmax_blocks, "out_maxes too small");
        assert!(out_sums.len() >= self.argmax_blocks, "out_sums too small");

        let mut cur_is_a = true;
        let slot = self.len as u32;
        let attn_len = (self.len + 1) as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
        self.encode_embed_lookup(&cmd, token, &self.ws.x_a);
        for layer in &self.layers {
            let (cur, next) = if cur_is_a {
                (&self.ws.x_a, &self.ws.x_b)
            } else {
                (&self.ws.x_b, &self.ws.x_a)
            };
            self.encode_layer(&cmd, layer, cur, next, slot, attn_len);
            cur_is_a = !cur_is_a;
        }

        let out_buf = if cur_is_a { &self.ws.x_a } else { &self.ws.x_b };
        self.encode_block_stats(&cmd, out_buf, inv_temperature, softcap);

        cmd.commit();
        cmd.waitUntilCompleted();

        download_f32(
            &self.final_head.block_maxes,
            &mut out_maxes[..self.argmax_blocks],
        );
        download_f32(
            &self.final_head.block_sums,
            &mut out_sums[..self.argmax_blocks],
        );
        self.len += 1;
        self.argmax_blocks
    }

    pub fn materialize_current_logits_block(
        &mut self,
        block_idx: usize,
        softcap: f32,
        out_logits: &mut [f32],
    ) -> usize {
        assert!(block_idx < self.argmax_blocks, "block_idx out of range");
        let valid = self
            .cfg
            .vocab
            .saturating_sub(block_idx * self.argmax_threads)
            .min(self.argmax_threads);
        assert!(out_logits.len() >= valid, "out_logits too small");

        let block_idx_u32 = block_idx as u32;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        self.encode_materialize_block(&cmd, block_idx_u32, softcap);
        cmd.commit();
        cmd.waitUntilCompleted();

        download_f32(&self.final_head.block_logits, &mut out_logits[..valid]);
        valid
    }

    fn encode_layer(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        layer: &LayerState,
        x_cur: &ProtocolObject<dyn MTLBuffer>,
        x_next: &ProtocolObject<dyn MTLBuffer>,
        slot: u32,
        attn_len: u32,
    ) {
        self.encode_rmsnorm(cmd, x_cur, &layer.gamma1, &self.ws.norm);

        let norm_matrix = self.make_matrix(&self.ws.norm, &self.desc.dim_vec);
        let q_matrix = self.make_matrix(&self.ws.q, &self.desc.q_vec);
        let k_matrix = self.make_matrix(&self.ws.k, &self.desc.kv_vec);
        let v_matrix = self.make_matrix(&self.ws.v, &self.desc.kv_vec);
        let wq_matrix = self.make_matrix(&layer.wq, &self.desc.wq);
        let wk_matrix = self.make_matrix(&layer.wk, &self.desc.wkv);
        let wv_matrix = self.make_matrix(&layer.wv, &self.desc.wkv);

        unsafe {
            self.mm
                .q_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &wq_matrix,
                    &norm_matrix,
                    &q_matrix,
                );
            self.mm
                .kv_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &wk_matrix,
                    &norm_matrix,
                    &k_matrix,
                );
            self.mm
                .kv_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &wv_matrix,
                    &norm_matrix,
                    &v_matrix,
                );
        }

        self.encode_rope(cmd, &self.ws.q, self.cfg.q_dim / 2, slot);
        self.encode_rope(cmd, &self.ws.k, self.cfg.kv_dim / 2, slot);
        self.encode_append_kv(
            cmd,
            &self.ws.k,
            &self.ws.v,
            &layer.k_cache,
            &layer.v_cache,
            slot,
        );
        self.encode_sdpa(
            cmd,
            &self.ws.q,
            &layer.k_cache,
            &layer.v_cache,
            &self.ws.attn,
            attn_len,
        );

        let attn_matrix = self.make_matrix(&self.ws.attn, &self.desc.q_vec);
        let wo_matrix = self.make_matrix(&layer.wo, &self.desc.wo);
        let o_matrix = self.make_matrix(&self.ws.norm, &self.desc.dim_vec);
        unsafe {
            self.mm
                .wo_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &wo_matrix,
                    &attn_matrix,
                    &o_matrix,
                );
        }

        self.encode_residual_add(cmd, x_cur, &self.ws.norm, x_next);
        self.encode_rmsnorm(cmd, x_next, &layer.gamma2, &self.ws.norm);

        let x2norm_matrix = self.make_matrix(&self.ws.norm, &self.desc.dim_vec);
        let h1_matrix = self.make_matrix(&self.ws.h1, &self.desc.hidden_vec);
        let h3_matrix = self.make_matrix(&self.ws.h3, &self.desc.hidden_vec);
        let w1_matrix = self.make_matrix(&layer.w1, &self.desc.wh);
        let w3_matrix = self.make_matrix(&layer.w3, &self.desc.wh);
        unsafe {
            self.mm
                .w1_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &w1_matrix,
                    &x2norm_matrix,
                    &h1_matrix,
                );
            self.mm
                .w3_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &w3_matrix,
                    &x2norm_matrix,
                    &h3_matrix,
                );
        }

        self.encode_silu_gate(cmd, &self.ws.h1, &self.ws.h3, &self.ws.gate);

        let w2_matrix = self.make_matrix(&layer.w2, &self.desc.wh);
        let gate_matrix = self.make_matrix(&self.ws.gate, &self.desc.hidden_vec);
        let x2_matrix = self.make_matrix(x_next, &self.desc.dim_vec);
        unsafe {
            self.mm
                .w2_proj
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &w2_matrix,
                    &gate_matrix,
                    &x2_matrix,
                );
        }
    }

    fn encode_embed_lookup(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        token: u32,
        out: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.embed_lookup);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.embed), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out), 0, 1);
            set_bytes(&enc, &token, 2);
        }
        let tg = self
            .pipes
            .embed_lookup
            .maxTotalThreadsPerThreadgroup()
            .min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.dim,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_rmsnorm(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
        gamma: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.rmsnorm);
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(gamma), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.rms_threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_rope(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
        pair_count: usize,
        pos: u32,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.rope);
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            set_bytes(&enc, &pos, 1);
        }
        let tg = self.pipes.rope.maxTotalThreadsPerThreadgroup().min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: pair_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_append_kv(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        k: &ProtocolObject<dyn MTLBuffer>,
        v: &ProtocolObject<dyn MTLBuffer>,
        k_cache: &ProtocolObject<dyn MTLBuffer>,
        v_cache: &ProtocolObject<dyn MTLBuffer>,
        slot: u32,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.append_kv);
            enc.setBuffer_offset_atIndex(Some(k), 0, 0);
            enc.setBuffer_offset_atIndex(Some(v), 0, 1);
            enc.setBuffer_offset_atIndex(Some(k_cache), 0, 2);
            enc.setBuffer_offset_atIndex(Some(v_cache), 0, 3);
            set_bytes(&enc, &slot, 4);
        }
        let tg = self
            .pipes
            .append_kv
            .maxTotalThreadsPerThreadgroup()
            .min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.kv_dim,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_sdpa(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        q: &ProtocolObject<dyn MTLBuffer>,
        k_cache: &ProtocolObject<dyn MTLBuffer>,
        v_cache: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
        attn_len: u32,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.sdpa);
            enc.setBuffer_offset_atIndex(Some(q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(k_cache), 0, 1);
            enc.setBuffer_offset_atIndex(Some(v_cache), 0, 2);
            enc.setBuffer_offset_atIndex(Some(out), 0, 3);
            set_bytes(&enc, &attn_len, 4);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.attn_threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_residual_add(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
        y: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.residual_add);
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(y), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out), 0, 2);
            set_bytes(&enc, &self.alpha, 3);
        }
        let tg = self
            .pipes
            .residual_add
            .maxTotalThreadsPerThreadgroup()
            .min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.dim,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_silu_gate(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        h1: &ProtocolObject<dyn MTLBuffer>,
        h3: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.silu_gate);
            enc.setBuffer_offset_atIndex(Some(h1), 0, 0);
            enc.setBuffer_offset_atIndex(Some(h3), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        }
        let tg = self
            .pipes
            .silu_gate
            .maxTotalThreadsPerThreadgroup()
            .min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.hidden,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_greedy_argmax(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
    ) {
        self.encode_rmsnorm(cmd, x, &self.final_head.gamma_final, &self.ws.norm);

        {
            let enc = cmd.computeCommandEncoder().expect("compute encoder");
            unsafe {
                enc.setComputePipelineState(&self.pipes.logits_argmax_blocks);
                enc.setBuffer_offset_atIndex(Some(&self.ws.norm), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.final_head.embed), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.final_head.block_vals), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.final_head.block_idxs), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.argmax_blocks,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.argmax_threads,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }

        {
            let enc = cmd.computeCommandEncoder().expect("compute encoder");
            unsafe {
                enc.setComputePipelineState(&self.pipes.logits_argmax_finalize);
                enc.setBuffer_offset_atIndex(Some(&self.final_head.block_vals), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.final_head.block_idxs), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.final_head.out_idx), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.argmax_threads,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
    }

    fn encode_topk_candidates(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
        top_k: u32,
    ) {
        self.encode_rmsnorm(cmd, x, &self.final_head.gamma_final, &self.ws.norm);

        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.logits_topk_blocks);
            enc.setBuffer_offset_atIndex(Some(&self.ws.norm), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.embed), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.topk_vals), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.topk_idxs), 0, 3);
            set_bytes(&enc, &top_k, 4);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.argmax_blocks,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.argmax_threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_block_stats(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &ProtocolObject<dyn MTLBuffer>,
        inv_temperature: f32,
        softcap: f32,
    ) {
        self.encode_rmsnorm(cmd, x, &self.final_head.gamma_final, &self.ws.norm);

        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.logits_block_stats);
            enc.setBuffer_offset_atIndex(Some(&self.ws.norm), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.embed), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.block_maxes), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.block_sums), 0, 3);
            set_bytes(&enc, &inv_temperature, 4);
            set_bytes(&enc, &softcap, 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.argmax_blocks,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.argmax_threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn encode_materialize_block(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        block_idx: u32,
        softcap: f32,
    ) {
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        unsafe {
            enc.setComputePipelineState(&self.pipes.logits_block_materialize);
            enc.setBuffer_offset_atIndex(Some(&self.ws.norm), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.embed), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.final_head.block_logits), 0, 2);
            set_bytes(&enc, &block_idx, 3);
            set_bytes(&enc, &softcap, 4);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.argmax_threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn make_matrix(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        descriptor: &MPSMatrixDescriptor,
    ) -> Retained<MPSMatrix> {
        unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buffer, descriptor) }
    }
}

impl LayerState {
    fn new<'a>(cfg: &Config, device: &ProtocolObject<dyn MTLDevice>, layer: LayerInit<'a>) -> Self {
        assert_eq!(layer.wq.len(), cfg.dim * cfg.q_dim);
        assert_eq!(layer.wk.len(), cfg.dim * cfg.kv_dim);
        assert_eq!(layer.wv.len(), cfg.dim * cfg.kv_dim);
        assert_eq!(layer.wo.len(), cfg.q_dim * cfg.dim);
        assert_eq!(layer.w1.len(), cfg.dim * cfg.hidden);
        assert_eq!(layer.w3.len(), cfg.dim * cfg.hidden);
        assert_eq!(layer.w2.len(), cfg.dim * cfg.hidden);
        assert_eq!(layer.gamma1.len(), cfg.dim);
        assert_eq!(layer.gamma2.len(), cfg.dim);

        Self {
            wq: buffer_from_slice(device, layer.wq),
            wk: buffer_from_slice(device, layer.wk),
            wv: buffer_from_slice(device, layer.wv),
            wo: buffer_from_slice(device, layer.wo),
            w1: buffer_from_slice(device, layer.w1),
            w3: buffer_from_slice(device, layer.w3),
            w2: buffer_from_slice(device, layer.w2),
            gamma1: buffer_from_slice(device, layer.gamma1),
            gamma2: buffer_from_slice(device, layer.gamma2),
            k_cache: zero_buffer(device, cfg.seq * cfg.kv_dim * std::mem::size_of::<f32>()),
            v_cache: zero_buffer(device, cfg.seq * cfg.kv_dim * std::mem::size_of::<f32>()),
        }
    }
}

impl Workspace {
    fn new(cfg: &Config, device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let dim_bytes = cfg.dim * std::mem::size_of::<f32>();
        let q_bytes = cfg.q_dim * std::mem::size_of::<f32>();
        let kv_bytes = cfg.kv_dim * std::mem::size_of::<f32>();
        let hidden_bytes = cfg.hidden * std::mem::size_of::<f32>();
        Self {
            x_a: zero_buffer(device, dim_bytes),
            x_b: zero_buffer(device, dim_bytes),
            norm: zero_buffer(device, dim_bytes),
            q: zero_buffer(device, q_bytes),
            k: zero_buffer(device, kv_bytes),
            v: zero_buffer(device, kv_bytes),
            attn: zero_buffer(device, q_bytes),
            h1: zero_buffer(device, hidden_bytes),
            h3: zero_buffer(device, hidden_bytes),
            gate: zero_buffer(device, hidden_bytes),
        }
    }
}

impl Descriptors {
    fn new(cfg: &Config) -> Self {
        Self {
            dim_vec: matrix_desc(cfg.dim, 1),
            q_vec: matrix_desc(cfg.q_dim, 1),
            kv_vec: matrix_desc(cfg.kv_dim, 1),
            hidden_vec: matrix_desc(cfg.hidden, 1),
            wq: matrix_desc(cfg.dim, cfg.q_dim),
            wkv: matrix_desc(cfg.dim, cfg.kv_dim),
            wo: matrix_desc(cfg.q_dim, cfg.dim),
            wh: matrix_desc(cfg.dim, cfg.hidden),
        }
    }
}

impl FinalHeadState {
    fn new<'a>(
        cfg: &Config,
        device: &ProtocolObject<dyn MTLDevice>,
        argmax_blocks: usize,
        head: FinalHeadInit<'a>,
    ) -> Self {
        assert_eq!(head.gamma_final.len(), cfg.dim);
        assert_eq!(head.embed.len(), cfg.vocab * cfg.dim);
        Self {
            gamma_final: buffer_from_slice(device, head.gamma_final),
            embed: buffer_from_slice(device, head.embed),
            block_maxes: zero_buffer(device, argmax_blocks * std::mem::size_of::<f32>()),
            block_sums: zero_buffer(device, argmax_blocks * std::mem::size_of::<f32>()),
            block_logits: zero_buffer(device, MAX_GPU_TOPK.max(256) * std::mem::size_of::<f32>()),
            topk_vals: zero_buffer(
                device,
                argmax_blocks * MAX_GPU_TOPK * std::mem::size_of::<f32>(),
            ),
            topk_idxs: zero_buffer(
                device,
                argmax_blocks * MAX_GPU_TOPK * std::mem::size_of::<u32>(),
            ),
            block_vals: zero_buffer(device, argmax_blocks * std::mem::size_of::<f32>()),
            block_idxs: zero_buffer(device, argmax_blocks * std::mem::size_of::<u32>()),
            out_idx: zero_buffer(device, std::mem::size_of::<u32>()),
        }
    }
}

impl Matmuls {
    fn new(cfg: &Config, device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            q_proj: unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    true,
                    false,
                    cfg.q_dim,
                    1,
                    cfg.dim,
                    1.0,
                    0.0,
                )
            },
            kv_proj: unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    true,
                    false,
                    cfg.kv_dim,
                    1,
                    cfg.dim,
                    1.0,
                    0.0,
                )
            },
            wo_proj: unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    true,
                    false,
                    cfg.dim,
                    1,
                    cfg.q_dim,
                    1.0,
                    0.0,
                )
            },
            w1_proj: unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    true,
                    false,
                    cfg.hidden,
                    1,
                    cfg.dim,
                    1.0,
                    0.0,
                )
            },
            w3_proj: unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    true,
                    false,
                    cfg.hidden,
                    1,
                    cfg.dim,
                    1.0,
                    0.0,
                )
            },
            w2_proj: unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    false,
                    false,
                    cfg.dim,
                    1,
                    cfg.hidden,
                    (1.0 / (2.0 * cfg.nlayers as f32).sqrt()) as f64,
                    1.0,
                )
            },
        }
    }
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    cfg: &Config,
    rms_threads: usize,
    attn_threads: usize,
    argmax_threads: usize,
    argmax_blocks: usize,
) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let source = NSString::from_str(&shader_source(
        cfg,
        rms_threads,
        attn_threads,
        argmax_threads,
        argmax_blocks,
    ));
    device
        .newLibraryWithSource_options_error(&source, None)
        .ok()
}

fn new_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let name = NSString::from_str(name);
    let function = library.newFunctionWithName(&name)?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .ok()
}

fn matrix_desc(rows: usize, cols: usize) -> Retained<MPSMatrixDescriptor> {
    unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            rows,
            cols,
            cols * std::mem::size_of::<f32>(),
            MPSDataType::Float32,
        )
    }
}

fn buffer_from_slice(
    device: &ProtocolObject<dyn MTLDevice>,
    xs: &[f32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(xs.as_ptr() as *mut c_void).expect("non-null"),
                std::mem::size_of_val(xs),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("buffer from slice")
    }
}

fn zero_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    byte_len: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let buf = device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
        .expect("zero buffer");
    unsafe {
        std::ptr::write_bytes(buf.contents().as_ptr(), 0, byte_len);
    }
    buf
}

fn upload_f32(buf: &ProtocolObject<dyn MTLBuffer>, xs: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), buf.contents().as_ptr() as *mut f32, xs.len());
    }
}

fn download_f32(buf: &ProtocolObject<dyn MTLBuffer>, xs: &mut [f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr() as *const f32,
            xs.as_mut_ptr(),
            xs.len(),
        );
    }
}

fn download_u32(buf: &ProtocolObject<dyn MTLBuffer>, xs: &mut [u32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr() as *const u32,
            xs.as_mut_ptr(),
            xs.len(),
        );
    }
}

unsafe fn set_bytes<T>(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new(value as *const T as *mut c_void).expect("non-null"),
            std::mem::size_of::<T>(),
            index,
        );
    }
}

fn max_pow2_threads(n: usize) -> usize {
    let target = n.next_power_of_two();
    if target <= 32 {
        32
    } else if target <= 64 {
        64
    } else if target <= 128 {
        128
    } else {
        256
    }
}

fn shader_source(
    cfg: &Config,
    rms_threads: usize,
    attn_threads: usize,
    argmax_threads: usize,
    argmax_blocks: usize,
) -> String {
    format!(
        r#"{SHADER_PREAMBLE}
constant uint DIM = {dim};
constant uint HEADS = {heads};
constant uint HD = {hd};
constant uint KV_DIM = {kv_dim};
constant uint VOCAB = {vocab};
constant uint GQA_RATIO = {gqa_ratio};
constant uint MAX_SEQ = {max_seq};
constant uint RMS_THREADS = {rms_threads};
constant uint ATTN_THREADS = {attn_threads};
constant uint ARGMAX_THREADS = {argmax_threads};
constant uint ARGMAX_BLOCKS = {argmax_blocks};
constant uint MAX_GPU_TOPK = {max_gpu_topk};
constant float RMS_EPSILON = {rms_eps}f;
constant float ATTN_SCALE = {attn_scale}f;

kernel void rmsnorm_single(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    threadgroup float partials[RMS_THREADS];
    float sum_sq = 0.0f;
    for (uint i = tid; i < DIM; i += RMS_THREADS) {{
        float v = x[i];
        sum_sq += v * v;
    }}
    partials[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RMS_THREADS >> 1; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            partials[tid] += partials[tid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float inv = rsqrt(partials[0] / float(DIM) + RMS_EPSILON);
    for (uint i = tid; i < DIM; i += RMS_THREADS) {{
        out[i] = x[i] * inv * gamma[i];
    }}
}}

kernel void embed_lookup(
    device const float* embed [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= DIM) {{
        return;
    }}
    out[gid] = embed[token * DIM + gid];
}}

kernel void rope_inplace(
    device float* x [[buffer(0)]],
    constant uint& pos [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint pair = gid;
    const uint pair_in_head = pair % (HD / 2);
    const uint even = pair * 2;
    const uint odd = even + 1;
    float theta = float(pos) / pow(10000.0f, 2.0f * float(pair_in_head) / float(HD));
    float c = cos(theta);
    float s = sin(theta);
    float xe = x[even];
    float xo = x[odd];
    x[even] = xe * c - xo * s;
    x[odd] = xo * c + xe * s;
}}

kernel void append_kv(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* k_cache [[buffer(2)]],
    device float* v_cache [[buffer(3)]],
    constant uint& slot [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= KV_DIM) {{
        return;
    }}
    uint idx = slot * KV_DIM + gid;
    k_cache[idx] = k[gid];
    v_cache[idx] = v[gid];
}}

kernel void sdpa_causal(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& attn_len [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint qh [[threadgroup_position_in_grid]]
) {{
    if (qh >= HEADS) {{
        return;
    }}
    uint kvh = qh / GQA_RATIO;
    uint q_base = qh * HD;
    uint kv_base = kvh * HD;
    threadgroup float partials[ATTN_THREADS];
    threadgroup float scores[MAX_SEQ];
    threadgroup float probs[MAX_SEQ];

    for (uint t = 0; t < attn_len; ++t) {{
        float partial = 0.0f;
        if (tid < HD) {{
            partial = q[q_base + tid] * k_cache[t * KV_DIM + kv_base + tid];
        }}
        partials[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_THREADS >> 1; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                partials[tid] += partials[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            scores[t] = partials[0] * ATTN_SCALE;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid == 0) {{
        float max_score = -3.402823466e+38f;
        for (uint t = 0; t < attn_len; ++t) {{
            max_score = max(max_score, scores[t]);
        }}
        float sum = 0.0f;
        for (uint t = 0; t < attn_len; ++t) {{
            float p = exp(scores[t] - max_score);
            probs[t] = p;
            sum += p;
        }}
        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint t = 0; t < attn_len; ++t) {{
            probs[t] *= inv_sum;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HD) {{
        float acc = 0.0f;
        for (uint t = 0; t < attn_len; ++t) {{
            acc += probs[t] * v_cache[t * KV_DIM + kv_base + tid];
        }}
        out[q_base + tid] = acc;
    }}
}}

kernel void residual_add(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= DIM) {{
        return;
    }}
    out[gid] = x[gid] + alpha * y[gid];
}}

kernel void silu_gate(
    device const float* h1 [[buffer(0)]],
    device const float* h3 [[buffer(1)]],
    device float* gate [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    float x = h1[gid];
    float sig = 1.0f / (1.0f + exp(-x));
    gate[gid] = x * sig * h3[gid];
}}

kernel void logits_argmax_blocks(
    device const float* x [[buffer(0)]],
    device const float* embed [[buffer(1)]],
    device float* block_vals [[buffer(2)]],
    device uint* block_idxs [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {{
    threadgroup float shared_x[DIM];
    threadgroup float vals[ARGMAX_THREADS];
    threadgroup uint idxs[ARGMAX_THREADS];

    for (uint i = tid; i < DIM; i += ARGMAX_THREADS) {{
        shared_x[i] = x[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = group * ARGMAX_THREADS + tid;
    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    if (row < VOCAB) {{
        float dot = 0.0f;
        const device float* row_ptr = embed + row * DIM;
        for (uint i = 0; i < DIM; ++i) {{
            dot += row_ptr[i] * shared_x[i];
        }}
        best = dot;
        best_idx = row;
    }}

    vals[tid] = best;
    idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ARGMAX_THREADS >> 1; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float other_val = vals[tid + stride];
            uint other_idx = idxs[tid + stride];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < idxs[tid])) {{
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid == 0) {{
        block_vals[group] = vals[0];
        block_idxs[group] = idxs[0];
    }}
}}

kernel void logits_topk_blocks(
    device const float* x [[buffer(0)]],
    device const float* embed [[buffer(1)]],
    device float* out_vals [[buffer(2)]],
    device uint* out_idxs [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {{
    threadgroup float shared_x[DIM];
    threadgroup float candidates[ARGMAX_THREADS];
    threadgroup uint candidate_idxs[ARGMAX_THREADS];
    threadgroup float reduce_vals[ARGMAX_THREADS];
    threadgroup uint reduce_idxs[ARGMAX_THREADS];

    for (uint i = tid; i < DIM; i += ARGMAX_THREADS) {{
        shared_x[i] = x[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = group * ARGMAX_THREADS + tid;
    float dot = -INFINITY;
    uint idx = 0xffffffffu;
    if (row < VOCAB) {{
        dot = 0.0f;
        const device float* row_ptr = embed + row * DIM;
        for (uint i = 0; i < DIM; ++i) {{
            dot += row_ptr[i] * shared_x[i];
        }}
        idx = row;
    }}
    candidates[tid] = dot;
    candidate_idxs[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_base = group * top_k;
    for (uint rank = 0; rank < top_k; ++rank) {{
        reduce_vals[tid] = candidates[tid];
        reduce_idxs[tid] = candidate_idxs[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ARGMAX_THREADS >> 1; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                float other_val = reduce_vals[tid + stride];
                uint other_idx = reduce_idxs[tid + stride];
                if (other_val > reduce_vals[tid] ||
                    (other_val == reduce_vals[tid] && other_idx < reduce_idxs[tid])) {{
                    reduce_vals[tid] = other_val;
                    reduce_idxs[tid] = other_idx;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float chosen_val = reduce_vals[0];
        uint chosen_idx = reduce_idxs[0];
        if (tid == 0) {{
            out_vals[block_base + rank] = chosen_val;
            out_idxs[block_base + rank] = chosen_idx;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (candidate_idxs[tid] == chosen_idx) {{
            candidates[tid] = -INFINITY;
            candidate_idxs[tid] = 0xffffffffu;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}

kernel void logits_block_stats(
    device const float* x [[buffer(0)]],
    device const float* embed [[buffer(1)]],
    device float* block_maxes [[buffer(2)]],
    device float* block_sums [[buffer(3)]],
    constant float& inv_temperature [[buffer(4)]],
    constant float& softcap [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {{
    threadgroup float shared_x[DIM];
    threadgroup float vals[ARGMAX_THREADS];

    for (uint i = tid; i < DIM; i += ARGMAX_THREADS) {{
        shared_x[i] = x[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = group * ARGMAX_THREADS + tid;
    float val = -INFINITY;
    if (row < VOCAB) {{
        float dot = 0.0f;
        const device float* row_ptr = embed + row * DIM;
        for (uint i = 0; i < DIM; ++i) {{
            dot += row_ptr[i] * shared_x[i];
        }}
        if (softcap > 0.0f) {{
            dot = softcap * tanh(dot / softcap);
        }}
        val = dot * inv_temperature;
    }}

    vals[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ARGMAX_THREADS >> 1; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            vals[tid] = max(vals[tid], vals[tid + stride]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float block_max = vals[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    if (row < VOCAB) {{
        sum = exp(val - block_max);
    }}
    vals[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ARGMAX_THREADS >> 1; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            vals[tid] += vals[tid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid == 0) {{
        block_maxes[group] = block_max;
        block_sums[group] = vals[0];
    }}
}}

kernel void logits_block_materialize(
    device const float* x [[buffer(0)]],
    device const float* embed [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& block_idx [[buffer(3)]],
    constant float& softcap [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    uint row = block_idx * ARGMAX_THREADS + tid;
    if (row >= VOCAB) {{
        return;
    }}
    float dot = 0.0f;
    const device float* row_ptr = embed + row * DIM;
    for (uint i = 0; i < DIM; ++i) {{
        dot += row_ptr[i] * x[i];
    }}
    if (softcap > 0.0f) {{
        dot = softcap * tanh(dot / softcap);
    }}
    out[tid] = dot;
}}

kernel void logits_argmax_finalize(
    device const float* block_vals [[buffer(0)]],
    device const uint* block_idxs [[buffer(1)]],
    device uint* out_idx [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    threadgroup float vals[ARGMAX_THREADS];
    threadgroup uint idxs[ARGMAX_THREADS];

    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    if (tid < ARGMAX_BLOCKS) {{
        best = block_vals[tid];
        best_idx = block_idxs[tid];
    }}

    vals[tid] = best;
    idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ARGMAX_THREADS >> 1; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float other_val = vals[tid + stride];
            uint other_idx = idxs[tid + stride];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < idxs[tid])) {{
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid == 0) {{
        out_idx[0] = idxs[0];
    }}
}}
"#,
        dim = cfg.dim,
        heads = cfg.heads,
        hd = cfg.hd,
        kv_dim = cfg.kv_dim,
        vocab = cfg.vocab,
        gqa_ratio = cfg.gqa_ratio.max(1),
        max_seq = cfg.seq,
        rms_threads = rms_threads,
        attn_threads = attn_threads,
        argmax_threads = argmax_threads,
        argmax_blocks = argmax_blocks,
        max_gpu_topk = MAX_GPU_TOPK,
        rms_eps = RMS_EPS,
        attn_scale = 1.0f32 / (cfg.hd as f32).sqrt(),
    )
}
