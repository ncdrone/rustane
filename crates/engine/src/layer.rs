//! Single transformer layer: compile kernels, run forward/backward on ANE + CPU.
//!
//! Forward: RMSNorm1(CPU) → sdpaFwd(ANE) → woFwd(ANE) → residual+RMSNorm2(CPU) → ffnFused(ANE)
//! Backward: Scale dy → ffnBwdW2t(ANE) → SiLU'(CPU) → ffnBwdW13t(ANE) → RMSNorm2 bwd(CPU)
//!           → wotBwd(ANE) → sdpaBwd1(ANE) → sdpaBwd2(ANE) → RoPE bwd(CPU)
//!           → qBwd(ANE) → kvBwd(ANE) → RMSNorm1 bwd(CPU)

use crate::cpu::{rmsnorm, vdsp};
use crate::kernels::{dyn_matmul, ffn_fused, sdpa_bwd, sdpa_fwd};
use crate::metal_ffn::MetalFFN;
use crate::model::{FfnActivation, ModelConfig};
use ane_bridge::ane::{Executable, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::sync::OnceLock;
use std::time::Instant;

/// Per-layer weights (f32, CPU-side).
#[derive(Clone)]
pub struct LayerWeights {
    pub wq: Vec<f32>,     // [DIM * Q_DIM]
    pub wk: Vec<f32>,     // [DIM * KV_DIM]
    pub wv: Vec<f32>,     // [DIM * KV_DIM]
    pub wo: Vec<f32>,     // [Q_DIM * DIM]
    pub w1: Vec<f32>,     // [DIM * HIDDEN]
    pub w3: Vec<f32>,     // [DIM * HIDDEN]
    pub w2: Vec<f32>,     // [DIM * HIDDEN]
    pub wqt: Vec<f32>,    // [Q_DIM * DIM]
    pub wkt: Vec<f32>,    // [KV_DIM * DIM]
    pub wvt: Vec<f32>,    // [KV_DIM * DIM]
    pub wot: Vec<f32>,    // [DIM * Q_DIM]
    pub w1t: Vec<f32>,    // [HIDDEN * DIM]
    pub w3t: Vec<f32>,    // [HIDDEN * DIM]
    pub gamma1: Vec<f32>, // [DIM]
    pub gamma2: Vec<f32>, // [DIM]
    pub generation: u64,  // bumps after each optimizer update
}

/// Weight gradients (same layout as weights).
pub struct LayerGrads {
    pub dwq: Vec<f32>,
    pub dwk: Vec<f32>,
    pub dwv: Vec<f32>,
    pub dwo: Vec<f32>,
    pub dw1: Vec<f32>,
    pub dw3: Vec<f32>,
    pub dw2: Vec<f32>,
    pub dgamma1: Vec<f32>,
    pub dgamma2: Vec<f32>,
}

/// Cached activations from forward pass, needed for backward.
pub struct ForwardCache {
    pub x: Vec<f32>,        // layer input [DIM * SEQ]
    pub xnorm: Vec<f32>,    // after RMSNorm1 [DIM * SEQ]
    pub rms_inv1: Vec<f32>, // per-position rms_inv [SEQ]
    pub q_rope: Vec<f32>,   // [Q_DIM * SEQ]
    pub k_rope: Vec<f32>,   // [Q_DIM * SEQ] (GQA tiles KV heads to query-head layout)
    pub v: Vec<f32>,        // [Q_DIM * SEQ] (GQA tiles KV heads to query-head layout)
    pub attn_out: Vec<f32>, // [Q_DIM * SEQ]
    pub o_out: Vec<f32>,    // woFwd output [DIM * SEQ]
    pub x2: Vec<f32>,       // post-attn residual [DIM * SEQ]
    pub x2norm: Vec<f32>,   // after RMSNorm2 [DIM * SEQ]
    pub rms_inv2: Vec<f32>, // per-position rms_inv [SEQ]
    pub h1: Vec<f32>,       // gate projection [HIDDEN * SEQ]
    pub h3: Vec<f32>,       // up projection [HIDDEN * SEQ]
    pub gate: Vec<f32>,     // silu(h1) * h3 [HIDDEN * SEQ]
}

impl ForwardCache {
    /// Pre-allocate all cache buffers for the given model config.
    /// Buffers are fully overwritten by forward_into — no zeroing needed at reuse.
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let hidden = cfg.hidden;
        Self {
            x: vec![0.0; dim * seq],
            xnorm: vec![0.0; dim * seq],
            rms_inv1: vec![0.0; seq],
            q_rope: vec![0.0; q_dim * seq],
            k_rope: vec![0.0; q_dim * seq],
            v: vec![0.0; q_dim * seq],
            attn_out: vec![0.0; q_dim * seq],
            o_out: vec![0.0; dim * seq],
            x2: vec![0.0; dim * seq],
            x2norm: vec![0.0; dim * seq],
            rms_inv2: vec![0.0; seq],
            h1: vec![0.0; hidden * seq],
            h3: vec![0.0; hidden * seq],
            gate: vec![0.0; hidden * seq],
        }
    }
}

/// Pre-allocated IOSurface buffers for all compiled kernels (input + output each).
/// Eliminates ~100 IOSurface alloc/dealloc cycles per training step.
/// All writes use `TensorData::copy_from_f32(&self, ..)` which takes `&self`,
/// so no interior mutability wrapper is needed.
pub struct KernelBuffers {
    // Forward: sdpa_fwd, wo_fwd, ffn_fused
    sdpa_fwd_xnorm: TensorData,
    sdpa_fwd_wq: TensorData,
    sdpa_fwd_wk: TensorData,
    sdpa_fwd_wv: TensorData,
    sdpa_fwd_out: TensorData,
    wo_fwd_attn: TensorData,
    wo_fwd_wo: TensorData,
    wo_fwd_out: TensorData,
    ffn_fused_x2norm: TensorData,
    ffn_fused_x2: TensorData,
    ffn_fused_w1: TensorData,
    ffn_fused_w3: TensorData,
    ffn_fused_w2: TensorData,
    ffn_fused_out: TensorData,
    // Backward: ffn_bwd_w2t, ffn_bwd_w13t, wot_bwd, sdpa_bwd1, sdpa_bwd2, q_bwd, kv_bwd
    ffn_bwd_w2t_dffn: TensorData,
    ffn_bwd_w2t_w2: TensorData,
    ffn_bwd_w2t_out: TensorData,
    ffn_bwd_w13t_dh1: TensorData,
    ffn_bwd_w13t_dh3: TensorData,
    ffn_bwd_w13t_w1t: TensorData,
    ffn_bwd_w13t_w3t: TensorData,
    ffn_bwd_w13t_out: TensorData,
    wot_bwd_dx2: TensorData,
    wot_bwd_wot: TensorData,
    wot_bwd_out: TensorData,
    sdpa_bwd1_q: TensorData,
    sdpa_bwd1_k: TensorData,
    sdpa_bwd1_v: TensorData,
    sdpa_bwd1_da: TensorData,
    sdpa_bwd1_out: TensorData,
    sdpa_bwd2_q: TensorData,
    sdpa_bwd2_k: TensorData,
    sdpa_bwd2_out: TensorData,
    q_bwd_dq: TensorData,
    q_bwd_wqt: TensorData,
    q_bwd_out: TensorData,
    kv_bwd_in: TensorData,
    kv_bwd_out: TensorData,
}

impl KernelBuffers {
    /// Pre-allocate all IOSurface buffers for the given model config.
    fn allocate(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;

        // Forward: sdpa_fwd
        let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
        let sdpa_fwd_xnorm = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let sdpa_fwd_wq = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: q_dim,
        });
        let sdpa_fwd_wk = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: kv_dim,
        });
        let sdpa_fwd_wv = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: kv_dim,
        });
        let sdpa_fwd_out = TensorData::new(Shape {
            batch: 1,
            channels: sdpa_out_ch,
            height: 1,
            width: seq,
        });

        // Forward: wo_fwd
        let wo_fwd_attn = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let wo_fwd_wo = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: dim,
        });
        let wo_fwd_out = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });

        // Forward: ffn_fused
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_fused_x2norm = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let ffn_fused_x2 = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let ffn_fused_w1 = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let ffn_fused_w3 = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let ffn_fused_w2 = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let ffn_fused_out = TensorData::new(Shape {
            batch: 1,
            channels: ffn_out_ch,
            height: 1,
            width: seq,
        });

        // Backward: ffn_bwd_w2t
        let ffn_bwd_w2t_dffn = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let ffn_bwd_w2t_w2 = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let ffn_bwd_w2t_out = TensorData::new(Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: seq,
        });

        // Backward: ffn_bwd_w13t
        let ffn_bwd_w13t_dh1 = TensorData::new(Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: seq,
        });
        let ffn_bwd_w13t_dh3 = TensorData::new(Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: seq,
        });
        let ffn_bwd_w13t_w1t = TensorData::new(Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: dim,
        });
        let ffn_bwd_w13t_w3t = TensorData::new(Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: dim,
        });
        let ffn_bwd_w13t_out = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });

        // Backward: wot_bwd
        let wot_bwd_dx2 = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let wot_bwd_wot = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: q_dim,
        });
        let wot_bwd_out = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });

        // Backward: sdpa_bwd1
        let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
        let sdpa_bwd1_q = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let sdpa_bwd1_k = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let sdpa_bwd1_v = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let sdpa_bwd1_da = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let sdpa_bwd1_out = TensorData::new(Shape {
            batch: 1,
            channels: bwd1_out_ch,
            height: 1,
            width: seq,
        });

        // Backward: sdpa_bwd2
        let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
        let sdpa_bwd2_q = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let sdpa_bwd2_k = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let sdpa_bwd2_out = TensorData::new(Shape {
            batch: 1,
            channels: bwd2_out_ch,
            height: 1,
            width: seq,
        });

        // Backward: q_bwd
        let q_bwd_dq = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        });
        let q_bwd_wqt = TensorData::new(Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: dim,
        });
        let q_bwd_out = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });

        // Backward: kv_bwd
        let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let kv_bwd_in = TensorData::new(Shape {
            batch: 1,
            channels: kv_dim,
            height: 1,
            width: kv_bwd_sp,
        });
        let kv_bwd_out = TensorData::new(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });

        Self {
            sdpa_fwd_xnorm,
            sdpa_fwd_wq,
            sdpa_fwd_wk,
            sdpa_fwd_wv,
            sdpa_fwd_out,
            wo_fwd_attn,
            wo_fwd_wo,
            wo_fwd_out,
            ffn_fused_x2norm,
            ffn_fused_x2,
            ffn_fused_w1,
            ffn_fused_w3,
            ffn_fused_w2,
            ffn_fused_out,
            ffn_bwd_w2t_dffn,
            ffn_bwd_w2t_w2,
            ffn_bwd_w2t_out,
            ffn_bwd_w13t_dh1,
            ffn_bwd_w13t_dh3,
            ffn_bwd_w13t_w1t,
            ffn_bwd_w13t_w3t,
            ffn_bwd_w13t_out,
            wot_bwd_dx2,
            wot_bwd_wot,
            wot_bwd_out,
            sdpa_bwd1_q,
            sdpa_bwd1_k,
            sdpa_bwd1_v,
            sdpa_bwd1_da,
            sdpa_bwd1_out,
            sdpa_bwd2_q,
            sdpa_bwd2_k,
            sdpa_bwd2_out,
            q_bwd_dq,
            q_bwd_wqt,
            q_bwd_out,
            kv_bwd_in,
            kv_bwd_out,
        }
    }
}

/// Pre-computed RoPE cos/sin tables (deterministic, depends only on hd and seq).
/// Eliminates 12× per-step recomputation of powf+cos+sin over 16K elements.
pub struct RopeTable {
    pub cos: Vec<f32>, // [pairs * seq] where pairs = hd/2
    pub sin: Vec<f32>, // [pairs * seq]
}

impl RopeTable {
    fn compute(hd: usize, seq: usize) -> Self {
        let pairs = hd / 2;
        let mut cos = vec![0.0f32; pairs * seq];
        let mut sin = vec![0.0f32; pairs * seq];
        for i in 0..pairs {
            let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / hd as f32);
            for p in 0..seq {
                let theta = p as f32 * freq;
                cos[i * seq + p] = theta.cos();
                sin[i * seq + p] = theta.sin();
            }
        }
        Self { cos, sin }
    }
}

/// Compiled kernels for one layer (shared across layers since same dims).
pub struct CompiledKernels {
    pub sdpa_fwd: Executable,
    pub wo_fwd: Executable,
    pub ffn_fused: Executable,
    pub ffn_bwd_w2t: Executable,
    pub ffn_bwd_w13t: Executable,
    pub ffn_bwd_w13t_split: Executable,
    pub wot_bwd: Executable,
    pub wot_bwd_split: Executable,
    pub sdpa_bwd1: Executable,
    pub sdpa_bwd2: Executable,
    pub q_bwd: Executable,
    pub q_bwd_split: Executable,
    pub kv_bwd: Executable,
    /// Pre-allocated IOSurface buffers for all kernels (avoids alloc/dealloc per call).
    bufs: KernelBuffers,
    /// Pre-computed RoPE tables (avoids 12× per-step recomputation).
    pub rope: RopeTable,
}

impl CompiledKernels {
    /// Compile all kernels for the given model config.
    pub fn compile(cfg: &ModelConfig) -> Self {
        let qos = NSQualityOfService::UserInteractive;

        // Forward kernels
        let sdpa_fwd = sdpa_fwd::build_split(cfg)
            .compile(qos)
            .expect("sdpaFwd compile");
        let wo_fwd = dyn_matmul::build_conv_split(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("woFwd compile");
        let ffn_fused = ffn_fused::build_split(cfg)
            .compile(qos)
            .expect("ffnFused compile");

        // Backward kernels
        let ffn_bwd_w2t = dyn_matmul::build_conv_split(cfg.dim, cfg.hidden, cfg.seq)
            .compile(qos)
            .expect("ffnBwdW2t compile");
        let ffn_bwd_w13t = dyn_matmul::build_dual(cfg.hidden, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("ffnBwdW13t compile");
        let ffn_bwd_w13t_split = dyn_matmul::build_dual_conv_split(cfg.hidden, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("ffnBwdW13t split compile");
        let wot_bwd = dyn_matmul::build(cfg.dim, cfg.q_dim, cfg.seq)
            .compile(qos)
            .expect("wotBwd compile");
        let wot_bwd_split = dyn_matmul::build_conv_split(cfg.dim, cfg.q_dim, cfg.seq)
            .compile(qos)
            .expect("wotBwd split compile");
        let sdpa_bwd1 = sdpa_bwd::build_bwd1_split(cfg)
            .compile(qos)
            .expect("sdpaBwd1 compile");
        let sdpa_bwd2 = sdpa_bwd::build_bwd2_split_from_bwd1(cfg)
            .compile(qos)
            .expect("sdpaBwd2 compile");
        let q_bwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("qBwd compile");
        let q_bwd_split = dyn_matmul::build_conv_split(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("qBwd split compile");
        let kv_bwd = dyn_matmul::build_dual(cfg.kv_dim, cfg.dim, cfg.seq)
            .compile(qos)
            .expect("kvBwd compile");

        // Pre-allocate IOSurface buffers for all kernels
        let bufs = KernelBuffers::allocate(cfg);

        // Pre-compute RoPE tables (deterministic, reused 12× per step)
        let rope = RopeTable::compute(cfg.hd, cfg.seq);

        Self {
            sdpa_fwd,
            wo_fwd,
            ffn_fused,
            ffn_bwd_w2t,
            ffn_bwd_w13t,
            ffn_bwd_w13t_split,
            wot_bwd,
            wot_bwd_split,
            sdpa_bwd1,
            sdpa_bwd2,
            q_bwd,
            q_bwd_split,
            kv_bwd,
            bufs,
            rope,
        }
    }
}

/// Persistent per-layer split-input surfaces for ffnBwdW13t.
/// Weight surfaces are only refreshed when `LayerWeights.generation` changes.
#[derive(Debug, Clone, Copy, Default)]
pub struct FfnBwdW13tCacheStats {
    pub weight_refreshes: u64,
    pub weight_reuse_hits: u64,
    pub stage_weights_ms: f64,
    pub stage_activations_ms: f64,
}

pub struct FfnBwdW13tCache {
    dh1: TensorData,
    dh3: TensorData,
    w1t: TensorData,
    w3t: TensorData,
    staged_generation: u64,
    stats: FfnBwdW13tCacheStats,
}

impl FfnBwdW13tCache {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            dh1: TensorData::new(Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: cfg.seq,
            }),
            dh3: TensorData::new(Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: cfg.seq,
            }),
            w1t: TensorData::new(Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: cfg.dim,
            }),
            w3t: TensorData::new(Shape {
                batch: 1,
                channels: cfg.hidden,
                height: 1,
                width: cfg.dim,
            }),
            staged_generation: u64::MAX,
            stats: FfnBwdW13tCacheStats::default(),
        }
    }

    pub fn reset_stats(&mut self) {
        self.stats = FfnBwdW13tCacheStats::default();
    }

    pub fn stats(&self) -> FfnBwdW13tCacheStats {
        self.stats
    }
}

/// Pre-allocated scratch buffers for backward pass.
/// Eliminates ~32 vec allocations per layer × 6 layers = 192 malloc+memset+free cycles.
/// All buffers are fully overwritten before use — no zeroing needed.
pub struct BackwardWorkspace {
    // Activation buffers [dim*seq] or [q_dim*seq]
    pub dffn: Vec<f32>,
    pub dx_ffn: Vec<f32>,
    pub dx2: Vec<f32>,
    pub dx2_tmp: Vec<f32>,
    pub dx2_scaled: Vec<f32>,
    pub da: Vec<f32>,
    pub dv_full: Vec<f32>,
    pub dq: Vec<f32>,
    pub dk: Vec<f32>,
    pub dv_kv: Vec<f32>,
    pub dk_kv: Vec<f32>,
    pub dx_attn: Vec<f32>,
    pub dx_kv: Vec<f32>,
    pub dx_merged: Vec<f32>,
    pub dx_rms1: Vec<f32>,
    // Hidden-sized buffers [hidden*seq]
    pub dsilu_raw: Vec<f32>,
    pub dh1: Vec<f32>,
    pub dh3: Vec<f32>,
    pub neg_h1: Vec<f32>,
    pub exp_neg: Vec<f32>,
    // Score buffers [heads*seq*seq]
    pub probs_flat: Vec<f32>,
    pub dp_flat: Vec<f32>,
    // Channel-first RMSNorm scratch [seq]
    pub rms_dot_buf: Vec<f32>,
}

impl BackwardWorkspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;
        let heads = cfg.heads;
        Self {
            dffn: vec![0.0; dim * seq],
            dx_ffn: vec![0.0; dim * seq],
            dx2: vec![0.0; dim * seq],
            dx2_tmp: vec![0.0; dim * seq],
            dx2_scaled: vec![0.0; dim * seq],
            da: vec![0.0; q_dim * seq],
            dv_full: vec![0.0; q_dim * seq],
            dq: vec![0.0; q_dim * seq],
            dk: vec![0.0; q_dim * seq],
            dv_kv: vec![0.0; kv_dim * seq],
            dk_kv: vec![0.0; kv_dim * seq],
            dx_attn: vec![0.0; dim * seq],
            dx_kv: vec![0.0; dim * seq],
            dx_merged: vec![0.0; dim * seq],
            dx_rms1: vec![0.0; dim * seq],
            dsilu_raw: vec![0.0; hidden * seq],
            dh1: vec![0.0; hidden * seq],
            dh3: vec![0.0; hidden * seq],
            neg_h1: vec![0.0; hidden * seq],
            exp_neg: vec![0.0; hidden * seq],
            probs_flat: vec![0.0; heads * seq * seq],
            dp_flat: vec![0.0; heads * seq * seq],
            rms_dot_buf: vec![0.0; seq],
        }
    }
}

impl LayerWeights {
    /// Initialize to match Obj-C reference (train.m):
    /// Wq/Wk/Wv: 1/√DIM, Wo/W2: zero-init (DeepNet), W1/W3: 1/√HIDDEN.
    pub fn random(cfg: &ModelConfig) -> Self {
        let scale_qkv = 1.0 / (cfg.dim as f32).sqrt();
        let scale_ffn = 1.0 / (cfg.hidden as f32).sqrt();
        let mut weights = Self {
            wq: random_vec(cfg.dim * cfg.q_dim, scale_qkv),
            wk: random_vec(cfg.dim * cfg.kv_dim, scale_qkv),
            wv: random_vec(cfg.dim * cfg.kv_dim, scale_qkv),
            wo: vec![0.0; cfg.q_dim * cfg.dim], // zero-init (DeepNet)
            w1: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w3: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w2: vec![0.0; cfg.dim * cfg.hidden], // zero-init (DeepNet)
            wqt: vec![0.0; cfg.q_dim * cfg.dim],
            wkt: vec![0.0; cfg.kv_dim * cfg.dim],
            wvt: vec![0.0; cfg.kv_dim * cfg.dim],
            wot: vec![0.0; cfg.dim * cfg.q_dim],
            w1t: vec![0.0; cfg.hidden * cfg.dim],
            w3t: vec![0.0; cfg.hidden * cfg.dim],
            gamma1: vec![1.0; cfg.dim],
            gamma2: vec![1.0; cfg.dim],
            generation: 0,
        };
        weights.refresh_transposes(cfg);
        weights
    }

    /// Refresh cached transpose views used by backward ANE kernels.
    pub fn refresh_transposes(&mut self, cfg: &ModelConfig) {
        vdsp::mtrans(
            &self.wq,
            cfg.q_dim,
            &mut self.wqt,
            cfg.dim,
            cfg.dim,
            cfg.q_dim,
        );
        vdsp::mtrans(
            &self.wk,
            cfg.kv_dim,
            &mut self.wkt,
            cfg.dim,
            cfg.dim,
            cfg.kv_dim,
        );
        vdsp::mtrans(
            &self.wv,
            cfg.kv_dim,
            &mut self.wvt,
            cfg.dim,
            cfg.dim,
            cfg.kv_dim,
        );
        vdsp::mtrans(
            &self.wo,
            cfg.dim,
            &mut self.wot,
            cfg.q_dim,
            cfg.q_dim,
            cfg.dim,
        );
        vdsp::mtrans(
            &self.w1,
            cfg.hidden,
            &mut self.w1t,
            cfg.dim,
            cfg.dim,
            cfg.hidden,
        );
        vdsp::mtrans(
            &self.w3,
            cfg.hidden,
            &mut self.w3t,
            cfg.dim,
            cfg.dim,
            cfg.hidden,
        );
    }
}

impl LayerGrads {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            dwq: vec![0.0; cfg.dim * cfg.q_dim],
            dwk: vec![0.0; cfg.dim * cfg.kv_dim],
            dwv: vec![0.0; cfg.dim * cfg.kv_dim],
            dwo: vec![0.0; cfg.q_dim * cfg.dim],
            dw1: vec![0.0; cfg.dim * cfg.hidden],
            dw3: vec![0.0; cfg.dim * cfg.hidden],
            dw2: vec![0.0; cfg.dim * cfg.hidden],
            dgamma1: vec![0.0; cfg.dim],
            dgamma2: vec![0.0; cfg.dim],
        }
    }

    pub fn zero_out(&mut self) {
        self.dwq.fill(0.0);
        self.dwk.fill(0.0);
        self.dwv.fill(0.0);
        self.dwo.fill(0.0);
        self.dw1.fill(0.0);
        self.dw3.fill(0.0);
        self.dw2.fill(0.0);
        self.dgamma1.fill(0.0);
        self.dgamma2.fill(0.0);
    }
}

/// Simple LCG pseudo-random for reproducible init (no external dep).
fn random_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let mut seed: u64 = 42 + n as u64;
    for x in v.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *x = r * scale;
    }
    v
}

#[inline]
fn ffn_gate_activation(cfg: &ModelConfig, x: f32) -> f32 {
    match cfg.ffn_activation {
        FfnActivation::SwiGlu => {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        }
        FfnActivation::LeakyReluSq => {
            let l = x.max(0.5 * x);
            l * l
        }
    }
}

#[inline]
fn ffn_gate_activation_derivative(cfg: &ModelConfig, x: f32, sig_cache: Option<f32>) -> f32 {
    match cfg.ffn_activation {
        FfnActivation::SwiGlu => {
            let sig = sig_cache.unwrap_or_else(|| 1.0 / (1.0 + (-x).exp()));
            sig * (1.0 + x * (1.0 - sig))
        }
        FfnActivation::LeakyReluSq => {
            let l = x.max(0.5 * x);
            let slope = if x > 0.0 { 1.0 } else { 0.5 };
            2.0 * l * slope
        }
    }
}

/// Pack two row-major weight matrices `[src_rows, channels]` directly into the
/// dual DynMatmul spatial layout as their transposes, avoiding temporary
/// transposed buffers.
/// Read a slice of channels from ANE output buffer into a pre-allocated destination.
/// No-alloc version of the former `read_channels`.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
/// Read contiguous channels from an IOSurface output buffer (stride = seq, no spatial padding).
/// Single memcpy instead of per-channel loop.
fn read_channels_into(
    src: &[f32],
    _total_ch: usize,
    seq: usize,
    ch_start: usize,
    ch_count: usize,
    dst: &mut [f32],
) {
    let start = ch_start * seq;
    dst.copy_from_slice(&src[start..start + ch_count * seq]);
}

// ── Forward pass ──

/// Run forward pass for one transformer layer.
/// Returns (x_next, cache) where x_next is [DIM * SEQ].
pub fn forward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
) -> (Vec<f32>, ForwardCache) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. RMSNorm1 (CPU): channel-first, no transpose needed
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(x, &weights.gamma1, &mut xnorm, &mut rms_inv1, dim, seq);

    // 2. Stage sdpaFwd split inputs
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    kernels.bufs.sdpa_fwd_xnorm.copy_from_f32(&xnorm);
    kernels.bufs.sdpa_fwd_wq.copy_from_f32(&weights.wq);
    kernels.bufs.sdpa_fwd_wk.copy_from_f32(&weights.wk);
    kernels.bufs.sdpa_fwd_wv.copy_from_f32(&weights.wv);

    // 3. Run sdpaFwd (ANE)
    kernels
        .sdpa_fwd
        .run_cached_direct(
            &[
                &kernels.bufs.sdpa_fwd_xnorm,
                &kernels.bufs.sdpa_fwd_wq,
                &kernels.bufs.sdpa_fwd_wk,
                &kernels.bufs.sdpa_fwd_wv,
            ],
            &[&kernels.bufs.sdpa_fwd_out],
        )
        .expect("ANE eval failed");

    // Extract: attn_out[Q_DIM,SEQ], Q_rope[Q_DIM,SEQ], K_rope[KV_DIM,SEQ], V[KV_DIM,SEQ]
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; q_dim * seq];
    let mut v = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut q_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim, q_dim, &mut k_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 3 * q_dim, q_dim, &mut v);
    }

    // 4. Stage woFwd directly into IOSurface
    kernels.bufs.wo_fwd_attn.copy_from_f32(&attn_out);
    kernels.bufs.wo_fwd_wo.copy_from_f32(&weights.wo);

    // 5. Run woFwd (ANE)
    kernels
        .wo_fwd
        .run_cached_direct(
            &[&kernels.bufs.wo_fwd_attn, &kernels.bufs.wo_fwd_wo],
            &[&kernels.bufs.wo_fwd_out],
        )
        .expect("ANE eval failed");

    // Read o_out directly from output IOSurface
    let mut o_out = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2 (CPU)
    // x2 = x + alpha * o_out  (vDSP: vsma = o_out * alpha + x)
    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(&x2, &weights.gamma2, &mut x2norm, &mut rms_inv2, dim, seq);

    // 7. Stage ffnFused split inputs
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    kernels.bufs.ffn_fused_x2norm.copy_from_f32(&x2norm);
    kernels.bufs.ffn_fused_x2.copy_from_f32(&x2);
    kernels.bufs.ffn_fused_w1.copy_from_f32(&weights.w1);
    kernels.bufs.ffn_fused_w3.copy_from_f32(&weights.w3);
    kernels.bufs.ffn_fused_w2.copy_from_f32(&weights.w2);

    // 8. Run ffnFused (ANE)
    kernels
        .ffn_fused
        .run_cached_direct(
            &[
                &kernels.bufs.ffn_fused_x2norm,
                &kernels.bufs.ffn_fused_x2,
                &kernels.bufs.ffn_fused_w1,
                &kernels.bufs.ffn_fused_w3,
                &kernels.bufs.ffn_fused_w2,
            ],
            &[&kernels.bufs.ffn_fused_out],
        )
        .expect("ANE eval failed");

    // Extract: x_next[DIM,SEQ], h1[HIDDEN,SEQ], h3[HIDDEN,SEQ], gate[HIDDEN,SEQ]
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
        read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
        read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
        read_channels_into(
            &locked,
            ffn_out_ch,
            seq,
            dim + 2 * hidden,
            hidden,
            &mut gate,
        );
    }

    let cache = ForwardCache {
        x: x.to_vec(),
        xnorm,
        rms_inv1,
        q_rope,
        k_rope,
        v,
        attn_out,
        o_out,
        x2,
        x2norm,
        rms_inv2,
        h1,
        h3,
        gate,
    };

    (x_next, cache)
}

/// Forward pass writing into pre-allocated cache (zero allocations).
/// `x_next` is written with the layer output [DIM * SEQ].
pub fn forward_into(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,
    x_next: &mut [f32],
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // Save layer input
    cache.x.copy_from_slice(x);

    // 1. RMSNorm1 (CPU)
    rmsnorm::forward_channel_first(
        x,
        &weights.gamma1,
        &mut cache.xnorm,
        &mut cache.rms_inv1,
        dim,
        seq,
    );

    // 2. Stage sdpaFwd split inputs
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    kernels.bufs.sdpa_fwd_xnorm.copy_from_f32(&cache.xnorm);
    kernels.bufs.sdpa_fwd_wq.copy_from_f32(&weights.wq);
    kernels.bufs.sdpa_fwd_wk.copy_from_f32(&weights.wk);
    kernels.bufs.sdpa_fwd_wv.copy_from_f32(&weights.wv);

    // 3. Run sdpaFwd (ANE) || pre-stage woFwd weights + ffnFused weights
    // sdpaFwd ANE takes ~2ms, giving plenty of CPU headroom to stage both
    // woFwd weights (~0.3ms) and ffnFused weights (~1.2ms) = ~1.5ms total CPU < 2ms ANE.
    // This eliminates the CPU bottleneck that previously slowed step 5 (woFwd overlap).
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_fwd
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_fwd_xnorm,
                        &kernels.bufs.sdpa_fwd_wq,
                        &kernels.bufs.sdpa_fwd_wk,
                        &kernels.bufs.sdpa_fwd_wv,
                    ],
                    &[&kernels.bufs.sdpa_fwd_out],
                )
                .expect("ANE eval failed");
        });
        // Stage woFwd weights
        kernels.bufs.wo_fwd_wo.copy_from_f32(&weights.wo);
        // Stage ffnFused weights (moved from step 5 — hidden behind sdpaFwd ANE time)
        kernels.bufs.ffn_fused_w1.copy_from_f32(&weights.w1);
        kernels.bufs.ffn_fused_w3.copy_from_f32(&weights.w3);
        kernels.bufs.ffn_fused_w2.copy_from_f32(&weights.w2);
        ane_handle.join().expect("ANE thread panicked");
    });

    // Extract sdpaFwd output
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut cache.attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut cache.q_rope);
        read_channels_into(
            &locked,
            sdpa_out_ch,
            seq,
            2 * q_dim,
            q_dim,
            &mut cache.k_rope,
        );
        read_channels_into(&locked, sdpa_out_ch, seq, 3 * q_dim, q_dim, &mut cache.v);
    }

    // 4. Stage woFwd activations only (weights already staged during sdpaFwd)
    kernels.bufs.wo_fwd_attn.copy_from_f32(&cache.attn_out);

    // 5. Run woFwd (ANE) — ffnFused weights already staged in step 3 during sdpaFwd
    kernels
        .wo_fwd
        .run_cached_direct(
            &[&kernels.bufs.wo_fwd_attn, &kernels.bufs.wo_fwd_wo],
            &[&kernels.bufs.wo_fwd_out],
        )
        .expect("ANE eval failed");

    // Read o_out
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        cache.o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2
    vdsp::vsma(&cache.o_out, alpha, x, &mut cache.x2);
    rmsnorm::forward_channel_first(
        &cache.x2,
        &weights.gamma2,
        &mut cache.x2norm,
        &mut cache.rms_inv2,
        dim,
        seq,
    );

    // 7. Stage ffnFused activations only (weights already staged in step 3 during sdpaFwd)
    kernels.bufs.ffn_fused_x2norm.copy_from_f32(&cache.x2norm);
    kernels.bufs.ffn_fused_x2.copy_from_f32(&cache.x2);

    // 8. Run ffnFused (ANE)
    kernels
        .ffn_fused
        .run_cached_direct(
            &[
                &kernels.bufs.ffn_fused_x2norm,
                &kernels.bufs.ffn_fused_x2,
                &kernels.bufs.ffn_fused_w1,
                &kernels.bufs.ffn_fused_w3,
                &kernels.bufs.ffn_fused_w2,
            ],
            &[&kernels.bufs.ffn_fused_out],
        )
        .expect("ANE eval failed");

    // Extract: x_next + cache intermediates
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, x_next);
        read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut cache.h1);
        read_channels_into(
            &locked,
            ffn_out_ch,
            seq,
            dim + hidden,
            hidden,
            &mut cache.h3,
        );
        read_channels_into(
            &locked,
            ffn_out_ch,
            seq,
            dim + 2 * hidden,
            hidden,
            &mut cache.gate,
        );
    }
}

/// Forward pass writing into pre-allocated buffers, with the FFN run on Metal.
pub fn forward_into_gpu_ffn(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    metal_ffn: &MetalFFN,
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,
    x_next: &mut [f32],
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    cache.x.copy_from_slice(x);

    // 1. RMSNorm1 (CPU)
    rmsnorm::forward_channel_first(
        x,
        &weights.gamma1,
        &mut cache.xnorm,
        &mut cache.rms_inv1,
        dim,
        seq,
    );

    // 2. Stage sdpaFwd split inputs
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    kernels.bufs.sdpa_fwd_xnorm.copy_from_f32(&cache.xnorm);
    kernels.bufs.sdpa_fwd_wq.copy_from_f32(&weights.wq);
    kernels.bufs.sdpa_fwd_wk.copy_from_f32(&weights.wk);
    kernels.bufs.sdpa_fwd_wv.copy_from_f32(&weights.wv);

    // 3. Run sdpaFwd (ANE) while staging wo weights.
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_fwd
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_fwd_xnorm,
                        &kernels.bufs.sdpa_fwd_wq,
                        &kernels.bufs.sdpa_fwd_wk,
                        &kernels.bufs.sdpa_fwd_wv,
                    ],
                    &[&kernels.bufs.sdpa_fwd_out],
                )
                .expect("ANE eval failed");
        });
        kernels.bufs.wo_fwd_wo.copy_from_f32(&weights.wo);
        ane_handle.join().expect("ANE thread panicked");
    });

    // Extract sdpaFwd output
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut cache.attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut cache.q_rope);
        read_channels_into(
            &locked,
            sdpa_out_ch,
            seq,
            2 * q_dim,
            q_dim,
            &mut cache.k_rope,
        );
        read_channels_into(&locked, sdpa_out_ch, seq, 3 * q_dim, q_dim, &mut cache.v);
    }

    // 4. Stage woFwd activations only
    kernels.bufs.wo_fwd_attn.copy_from_f32(&cache.attn_out);

    // 5. Run woFwd (ANE)
    kernels
        .wo_fwd
        .run_cached_direct(
            &[&kernels.bufs.wo_fwd_attn, &kernels.bufs.wo_fwd_wo],
            &[&kernels.bufs.wo_fwd_out],
        )
        .expect("ANE eval failed");

    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        cache.o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2
    vdsp::vsma(&cache.o_out, alpha, x, &mut cache.x2);
    rmsnorm::forward_channel_first(
        &cache.x2,
        &weights.gamma2,
        &mut cache.x2norm,
        &mut cache.rms_inv2,
        dim,
        seq,
    );

    // 7. FFN on Metal GPU.
    metal_ffn.forward_into(
        cfg,
        &cache.x2norm,
        &weights.w1,
        &weights.w3,
        &weights.w2,
        &cache.x2,
        &mut cache.h1,
        &mut cache.h3,
        &mut cache.gate,
        x_next,
    );
}

/// Pipelined forward: defers own h1/h3/gate readback, optionally reads previous
/// layer's deferred h1/h3/gate during sdpaFwd ANE overlap (step 3).
///
/// The ffnFused output IOSurface retains the previous layer's data until this
/// layer's ffnFused runs (~5ms later), giving ample time to read during step 3.
/// sdpaFwd ANE takes ~3.2ms, CPU staging takes ~1.5ms → ~1.7ms spare for readback.
/// The 12MB readback takes ~0.8ms, fitting within the spare window.
pub fn forward_into_pipelined(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
    cache: &mut ForwardCache,
    x_next: &mut [f32],
    prev_cache: Option<&mut ForwardCache>,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    cache.x.copy_from_slice(x);

    // 1. RMSNorm1 (CPU)
    rmsnorm::forward_channel_first(
        x,
        &weights.gamma1,
        &mut cache.xnorm,
        &mut cache.rms_inv1,
        dim,
        seq,
    );

    // 2. Stage sdpaFwd split inputs
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    kernels.bufs.sdpa_fwd_xnorm.copy_from_f32(&cache.xnorm);
    kernels.bufs.sdpa_fwd_wq.copy_from_f32(&weights.wq);
    kernels.bufs.sdpa_fwd_wk.copy_from_f32(&weights.wk);
    kernels.bufs.sdpa_fwd_wv.copy_from_f32(&weights.wv);

    // 3. Run sdpaFwd (ANE) || pre-stage weights + deferred prev-layer cache readback
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_fwd
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_fwd_xnorm,
                        &kernels.bufs.sdpa_fwd_wq,
                        &kernels.bufs.sdpa_fwd_wk,
                        &kernels.bufs.sdpa_fwd_wv,
                    ],
                    &[&kernels.bufs.sdpa_fwd_out],
                )
                .expect("ANE eval failed");
        });
        // Stage woFwd weights
        kernels.bufs.wo_fwd_wo.copy_from_f32(&weights.wo);
        // Stage ffnFused weights
        kernels.bufs.ffn_fused_w1.copy_from_f32(&weights.w1);
        kernels.bufs.ffn_fused_w3.copy_from_f32(&weights.w3);
        kernels.bufs.ffn_fused_w2.copy_from_f32(&weights.w2);
        // Deferred readback: read PREVIOUS layer's h1/h3/gate from ffn_fused_out.
        // This IOSurface still holds the previous layer's output (not yet overwritten).
        // Safe: ffn_fused_out is not touched by sdpaFwd (which uses sdpa_fwd_* inputs/out).
        if let Some(prev) = prev_cache {
            let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
            read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut prev.h1);
            read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut prev.h3);
            read_channels_into(
                &locked,
                ffn_out_ch,
                seq,
                dim + 2 * hidden,
                hidden,
                &mut prev.gate,
            );
        }
        ane_handle.join().expect("ANE thread panicked");
    });

    // Extract sdpaFwd output
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut cache.attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut cache.q_rope);
        read_channels_into(
            &locked,
            sdpa_out_ch,
            seq,
            2 * q_dim,
            q_dim,
            &mut cache.k_rope,
        );
        read_channels_into(&locked, sdpa_out_ch, seq, 3 * q_dim, q_dim, &mut cache.v);
    }

    // 4. Stage woFwd activations only
    kernels.bufs.wo_fwd_attn.copy_from_f32(&cache.attn_out);

    // 5. Run woFwd (ANE)
    kernels
        .wo_fwd
        .run_cached_direct(
            &[&kernels.bufs.wo_fwd_attn, &kernels.bufs.wo_fwd_wo],
            &[&kernels.bufs.wo_fwd_out],
        )
        .expect("ANE eval failed");

    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        cache.o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2
    vdsp::vsma(&cache.o_out, alpha, x, &mut cache.x2);
    rmsnorm::forward_channel_first(
        &cache.x2,
        &weights.gamma2,
        &mut cache.x2norm,
        &mut cache.rms_inv2,
        dim,
        seq,
    );

    // 7. Stage ffnFused activations only
    kernels.bufs.ffn_fused_x2norm.copy_from_f32(&cache.x2norm);
    kernels.bufs.ffn_fused_x2.copy_from_f32(&cache.x2);

    // 8. Run ffnFused (ANE)
    kernels
        .ffn_fused
        .run_cached_direct(
            &[
                &kernels.bufs.ffn_fused_x2norm,
                &kernels.bufs.ffn_fused_x2,
                &kernels.bufs.ffn_fused_w1,
                &kernels.bufs.ffn_fused_w3,
                &kernels.bufs.ffn_fused_w2,
            ],
            &[&kernels.bufs.ffn_fused_out],
        )
        .expect("ANE eval failed");

    // Extract x_next ONLY — h1/h3/gate deferred to next layer's step 3
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, x_next);
    }
}

/// Read deferred h1/h3/gate from ffnFused IOSurface into cache.
/// Used for the last layer (no next layer to overlap with).
pub fn read_ffn_cache(cfg: &ModelConfig, kernels: &CompiledKernels, cache: &mut ForwardCache) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let hidden = cfg.hidden;
    let ffn_out_ch = ffn_fused::output_channels(cfg);

    let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
    read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut cache.h1);
    read_channels_into(
        &locked,
        ffn_out_ch,
        seq,
        dim + hidden,
        hidden,
        &mut cache.h3,
    );
    read_channels_into(
        &locked,
        ffn_out_ch,
        seq,
        dim + 2 * hidden,
        hidden,
        &mut cache.gate,
    );
}

/// Timing breakdown for forward pass.
#[derive(Debug, Clone)]
pub struct ForwardTimings {
    pub rmsnorm1_ms: f32,
    pub stage_sdpa_ms: f32,
    pub ane_sdpa_ms: f32,
    pub read_sdpa_ms: f32,
    pub stage_wo_ms: f32,
    pub ane_wo_ms: f32,
    pub read_wo_ms: f32,
    pub residual_rmsnorm2_ms: f32,
    pub stage_ffn_ms: f32,
    pub ane_ffn_ms: f32,
    pub read_ffn_ms: f32,
    pub total_ms: f32,
}

impl ForwardTimings {
    pub fn print(&self) {
        println!("  {:<30} {:>6.2}ms", "RMSNorm1 (CPU)", self.rmsnorm1_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "stage sdpaFwd IOSurf", self.stage_sdpa_ms
        );
        println!("  {:<30} {:>6.2}ms", "ANE sdpaFwd", self.ane_sdpa_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "read sdpaFwd output", self.read_sdpa_ms
        );
        println!("  {:<30} {:>6.2}ms", "stage woFwd IOSurf", self.stage_wo_ms);
        println!("  {:<30} {:>6.2}ms", "ANE woFwd", self.ane_wo_ms);
        println!("  {:<30} {:>6.2}ms", "read woFwd output", self.read_wo_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "residual + RMSNorm2 (CPU)", self.residual_rmsnorm2_ms
        );
        println!(
            "  {:<30} {:>6.2}ms",
            "stage ffnFused IOSurf", self.stage_ffn_ms
        );
        println!("  {:<30} {:>6.2}ms", "ANE ffnFused", self.ane_ffn_ms);
        println!(
            "  {:<30} {:>6.2}ms",
            "read ffnFused output", self.read_ffn_ms
        );
        println!("  {:<30} {:>6.2}ms", "TOTAL", self.total_ms);
    }
}

/// Forward pass with per-operation timing (same output as `forward`).
pub fn forward_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
) -> (Vec<f32>, ForwardCache, ForwardTimings) {
    let t_total = Instant::now();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. RMSNorm1 (channel-first, no transpose)
    let t = Instant::now();
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(x, &weights.gamma1, &mut xnorm, &mut rms_inv1, dim, seq);
    let rmsnorm1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 2. Stage sdpaFwd
    let t = Instant::now();
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    kernels.bufs.sdpa_fwd_xnorm.copy_from_f32(&xnorm);
    kernels.bufs.sdpa_fwd_wq.copy_from_f32(&weights.wq);
    kernels.bufs.sdpa_fwd_wk.copy_from_f32(&weights.wk);
    kernels.bufs.sdpa_fwd_wv.copy_from_f32(&weights.wv);
    let stage_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 3. ANE sdpaFwd || pre-stage woFwd weights + ffnFused weights
    // sdpaFwd ANE takes ~2ms, hiding ~1.5ms of CPU staging work.
    let t = Instant::now();
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_fwd
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_fwd_xnorm,
                        &kernels.bufs.sdpa_fwd_wq,
                        &kernels.bufs.sdpa_fwd_wk,
                        &kernels.bufs.sdpa_fwd_wv,
                    ],
                    &[&kernels.bufs.sdpa_fwd_out],
                )
                .expect("ANE eval failed");
        });
        // Stage woFwd weights
        kernels.bufs.wo_fwd_wo.copy_from_f32(&weights.wo);
        // Stage ffnFused weights (moved from woFwd overlap — hidden behind sdpaFwd ANE)
        kernels.bufs.ffn_fused_w1.copy_from_f32(&weights.w1);
        kernels.bufs.ffn_fused_w3.copy_from_f32(&weights.w3);
        kernels.bufs.ffn_fused_w2.copy_from_f32(&weights.w2);
        ane_handle.join().expect("ANE thread panicked");
    });
    let ane_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 4. Read output
    let t = Instant::now();
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; q_dim * seq];
    let mut v = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut q_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim, q_dim, &mut k_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 3 * q_dim, q_dim, &mut v);
    }
    let read_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 5. Stage woFwd activations only (weights already staged during sdpaFwd)
    let t = Instant::now();
    kernels.bufs.wo_fwd_attn.copy_from_f32(&attn_out);
    let stage_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 6. ANE woFwd — ffnFused weights already staged in step 3
    let t = Instant::now();
    kernels
        .wo_fwd
        .run_cached_direct(
            &[&kernels.bufs.wo_fwd_attn, &kernels.bufs.wo_fwd_wo],
            &[&kernels.bufs.wo_fwd_out],
        )
        .expect("ANE eval failed");
    let ane_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 7. Read woFwd output
    let t = Instant::now();
    let mut o_out = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        o_out.copy_from_slice(&locked[..dim * seq]);
    }
    let read_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 8. Residual + RMSNorm2 (bulk transpose)
    let t = Instant::now();
    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(&x2, &weights.gamma2, &mut x2norm, &mut rms_inv2, dim, seq);
    let residual_rmsnorm2_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 9. Stage ffnFused activations only (weights already staged in step 3 during sdpaFwd)
    let t = Instant::now();
    kernels.bufs.ffn_fused_x2norm.copy_from_f32(&x2norm);
    kernels.bufs.ffn_fused_x2.copy_from_f32(&x2);
    let stage_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 10. ANE ffnFused
    let t = Instant::now();
    kernels
        .ffn_fused
        .run_cached_direct(
            &[
                &kernels.bufs.ffn_fused_x2norm,
                &kernels.bufs.ffn_fused_x2,
                &kernels.bufs.ffn_fused_w1,
                &kernels.bufs.ffn_fused_w3,
                &kernels.bufs.ffn_fused_w2,
            ],
            &[&kernels.bufs.ffn_fused_out],
        )
        .expect("ANE eval failed");
    let ane_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 11. Read ffnFused output
    let t = Instant::now();
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
        read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
        read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
        read_channels_into(
            &locked,
            ffn_out_ch,
            seq,
            dim + 2 * hidden,
            hidden,
            &mut gate,
        );
    }
    let read_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let cache = ForwardCache {
        x: x.to_vec(),
        xnorm,
        rms_inv1,
        q_rope,
        k_rope,
        v,
        attn_out,
        o_out,
        x2,
        x2norm,
        rms_inv2,
        h1,
        h3,
        gate,
    };

    let timings = ForwardTimings {
        rmsnorm1_ms,
        stage_sdpa_ms,
        ane_sdpa_ms,
        read_sdpa_ms,
        stage_wo_ms,
        ane_wo_ms,
        read_wo_ms,
        residual_rmsnorm2_ms,
        stage_ffn_ms,
        ane_ffn_ms,
        read_ffn_ms,
        total_ms,
    };

    (x_next, cache, timings)
}

/// Timing breakdown for backward pass.
#[derive(Debug, Clone)]
pub struct BackwardTimings {
    pub scale_dy_ms: f32,
    pub stage_run_ffn_bwd_w2t_ms: f32,
    pub silu_deriv_ms: f32,
    pub stage_ffn_bwd_w13t_ms: f32,
    pub async_ffn_bwd_w13t_plus_dw_ms: f32,
    pub rmsnorm2_bwd_ms: f32,
    pub stage_run_wot_bwd_ms: f32,
    pub stage_sdpa_bwd1_ms: f32,
    pub async_sdpa_bwd1_plus_dwo_ms: f32,
    pub read_sdpa_bwd1_ms: f32,
    pub stage_run_sdpa_bwd2_ms: f32,
    pub rope_bwd_ms: f32,
    pub stage_q_bwd_ms: f32,
    pub async_q_bwd_plus_dw_ms: f32,
    pub stage_run_kv_bwd_ms: f32,
    pub rmsnorm1_bwd_ms: f32,
    pub merge_dx_ms: f32,
    pub total_ms: f32,
}

impl BackwardTimings {
    pub fn print(&self) {
        println!("  {:<35} {:>6.2}ms", "scale dy (vDSP)", self.scale_dy_ms);
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run ffnBwdW2t (ANE)", self.stage_run_ffn_bwd_w2t_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "SiLU derivative (CPU)", self.silu_deriv_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage ffnBwdW13t", self.stage_ffn_bwd_w13t_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "async ffnBwdW13t + dW2+dW1+dW3", self.async_ffn_bwd_w13t_plus_dw_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "RMSNorm2 backward (CPU)", self.rmsnorm2_bwd_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run wotBwd (ANE)", self.stage_run_wot_bwd_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage sdpaBwd1", self.stage_sdpa_bwd1_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "async sdpaBwd1 + dWo", self.async_sdpa_bwd1_plus_dwo_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "read sdpaBwd1 output", self.read_sdpa_bwd1_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run sdpaBwd2 (ANE)", self.stage_run_sdpa_bwd2_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "RoPE backward (CPU)", self.rope_bwd_ms
        );
        println!("  {:<35} {:>6.2}ms", "stage qBwd", self.stage_q_bwd_ms);
        println!(
            "  {:<35} {:>6.2}ms",
            "async qBwd + dWq+dWk+dWv", self.async_q_bwd_plus_dw_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "stage+run kvBwd (ANE)", self.stage_run_kv_bwd_ms
        );
        println!(
            "  {:<35} {:>6.2}ms",
            "RMSNorm1 backward (CPU)", self.rmsnorm1_bwd_ms
        );
        println!("  {:<35} {:>6.2}ms", "merge dx (vDSP)", self.merge_dx_ms);
        println!("  {:<35} {:>6.2}ms", "TOTAL", self.total_ms);
    }
}

/// Backward pass with per-operation timing (same output as `backward`).
pub fn backward_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
) -> (Vec<f32>, BackwardTimings) {
    let t_total = Instant::now();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. Scale dy
    let t = Instant::now();
    vdsp::vsmul(dy, alpha, &mut ws.dffn);
    let scale_dy_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 2. ffnBwdW2t
    let t = Instant::now();
    kernels.bufs.ffn_bwd_w2t_dffn.copy_from_f32(&ws.dffn);
    kernels.bufs.ffn_bwd_w2t_w2.copy_from_f32(&weights.w2);
    kernels
        .ffn_bwd_w2t
        .run_cached_direct(
            &[&kernels.bufs.ffn_bwd_w2t_dffn, &kernels.bufs.ffn_bwd_w2t_w2],
            &[&kernels.bufs.ffn_bwd_w2t_out],
        )
        .expect("ANE eval failed");
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        ws.dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }
    let stage_run_ffn_bwd_w2t_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 3. Gate activation derivative.
    let t = Instant::now();
    let n = hidden * seq;
    if cfg.ffn_activation == FfnActivation::SwiGlu {
        vdsp::vsmul(&cache.h1, -1.0, &mut ws.neg_h1);
        vdsp::expf(&ws.neg_h1, &mut ws.exp_neg);
        vdsp::vsadd(&ws.exp_neg, 1.0, &mut ws.neg_h1); // neg_h1 = 1 + exp(-h1)
        vdsp::recf_inplace(&mut ws.neg_h1); // neg_h1 = sig = 1/(1+exp(-h1))
        for i in 0..n {
            let sig = ws.neg_h1[i];
            let act = ffn_gate_activation(&cfg, cache.h1[i]);
            let act_deriv = ffn_gate_activation_derivative(&cfg, cache.h1[i], Some(sig));
            ws.dh3[i] = ws.dsilu_raw[i] * act;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * act_deriv;
        }
    } else {
        for i in 0..n {
            let act = ffn_gate_activation(&cfg, cache.h1[i]);
            let act_deriv = ffn_gate_activation_derivative(&cfg, cache.h1[i], None);
            ws.dh3[i] = ws.dsilu_raw[i] * act;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * act_deriv;
        }
    }
    let silu_deriv_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 4. Stage ffnBwdW13t split inputs (flat IOSurface copies)
    let t = Instant::now();
    {
        kernels.bufs.ffn_bwd_w13t_dh1.copy_from_f32(&ws.dh1);
        kernels.bufs.ffn_bwd_w13t_dh3.copy_from_f32(&ws.dh3);
        kernels.bufs.ffn_bwd_w13t_w1t.copy_from_f32(&weights.w1t);
        kernels.bufs.ffn_bwd_w13t_w3t.copy_from_f32(&weights.w3t);
    }
    let stage_ffn_bwd_w13t_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 5. ASYNC: ANE ffnBwdW13t || CPU dW
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .ffn_bwd_w13t_split
                .run_cached_direct(
                    &[
                        &kernels.bufs.ffn_bwd_w13t_dh1,
                        &kernels.bufs.ffn_bwd_w13t_dh3,
                        &kernels.bufs.ffn_bwd_w13t_w1t,
                        &kernels.bufs.ffn_bwd_w13t_w3t,
                    ],
                    &[&kernels.bufs.ffn_bwd_w13t_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(&ws.dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        accumulate_dw(&cache.x2norm, dim, &ws.dh1, hidden, seq, &mut grads.dw1);
        accumulate_dw(&cache.x2norm, dim, &ws.dh3, hidden, seq, &mut grads.dw3);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        ws.dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }
    let async_ffn_bwd_w13t_plus_dw_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 6. RMSNorm2 backward (channel-first, no transpose)
    let t = Instant::now();
    rmsnorm::backward_channel_first(
        &ws.dx_ffn,
        &cache.x2,
        &weights.gamma2,
        &cache.rms_inv2,
        &mut ws.dx2,
        &mut grads.dgamma2,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    vdsp::vadd(&ws.dx2, dy, &mut ws.dx2_tmp);
    ws.dx2.copy_from_slice(&ws.dx2_tmp);
    let rmsnorm2_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 7. wotBwd (flat split-input copies) + async pre-stage sdpaBwd1
    let t = Instant::now();
    vdsp::vsmul(&ws.dx2, alpha, &mut ws.dx2_scaled);
    {
        kernels.bufs.wot_bwd_dx2.copy_from_f32(&ws.dx2_scaled);
        kernels.bufs.wot_bwd_wot.copy_from_f32(&weights.wot);
    }
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    // ASYNC: ANE wotBwd || pre-stage 3 of 4 sdpaBwd1 inputs (from forward cache)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .wot_bwd_split
                .run_cached_direct(
                    &[&kernels.bufs.wot_bwd_dx2, &kernels.bufs.wot_bwd_wot],
                    &[&kernels.bufs.wot_bwd_out],
                )
                .expect("ANE eval failed");
        });
        kernels.bufs.sdpa_bwd1_q.copy_from_f32(&cache.q_rope);
        kernels.bufs.sdpa_bwd1_k.copy_from_f32(&cache.k_rope);
        kernels.bufs.sdpa_bwd1_v.copy_from_f32(&cache.v);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        ws.da.copy_from_slice(&locked[..q_dim * seq]);
    }
    let stage_run_wot_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 8. Stage remaining sdpaBwd1 input (da depends on wotBwd output)
    let t = Instant::now();
    kernels.bufs.sdpa_bwd1_da.copy_from_f32(&ws.da);
    let stage_sdpa_bwd1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 9. ASYNC: ANE sdpaBwd1 || CPU dWo
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd1
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_bwd1_q,
                        &kernels.bufs.sdpa_bwd1_k,
                        &kernels.bufs.sdpa_bwd1_v,
                        &kernels.bufs.sdpa_bwd1_da,
                    ],
                    &[&kernels.bufs.sdpa_bwd1_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(
            &cache.attn_out,
            q_dim,
            &ws.dx2_scaled,
            dim,
            seq,
            &mut grads.dwo,
        );
        ane_handle.join().expect("ANE thread panicked");
    });
    let async_sdpa_bwd1_plus_dwo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 10. Read sdpaBwd1
    let t = Instant::now();
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut ws.dv_full);
    }
    let read_sdpa_bwd1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 11. sdpaBwd2
    let t = Instant::now();
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    kernels.bufs.sdpa_bwd2_q.copy_from_f32(&cache.q_rope);
    kernels.bufs.sdpa_bwd2_k.copy_from_f32(&cache.k_rope);
    kernels
        .sdpa_bwd2
        .run_cached_direct(
            &[
                &kernels.bufs.sdpa_bwd1_out,
                &kernels.bufs.sdpa_bwd2_q,
                &kernels.bufs.sdpa_bwd2_k,
            ],
            &[&kernels.bufs.sdpa_bwd2_out],
        )
        .expect("ANE eval failed");
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut ws.dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut ws.dk);
    }
    let stage_run_sdpa_bwd2_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 12. RoPE backward
    let t = Instant::now();
    rope_backward_inplace(&mut ws.dq, heads, hd, seq, &kernels.rope);
    rope_backward_inplace(&mut ws.dk, heads, hd, seq, &kernels.rope);
    reduce_gqa_grads(cfg, &ws.dk, &ws.dv_full, &mut ws.dk_kv, &mut ws.dv_kv);
    let rope_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 12.5. Stage kvBwd early (dk post-RoPE)
    let t = Instant::now();
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..kv_dim {
            let row = c * kv_bwd_sp;
            buf[row..row + seq].copy_from_slice(&ws.dk_kv[c * seq..c * seq + seq]);
            buf[row + seq..row + 2 * seq].copy_from_slice(&ws.dv_kv[c * seq..c * seq + seq]);
            buf[row + 2 * seq..row + 2 * seq + dim]
                .copy_from_slice(&weights.wkt[c * dim..c * dim + dim]);
            buf[row + 2 * seq + dim..row + 2 * seq + 2 * dim]
                .copy_from_slice(&weights.wvt[c * dim..c * dim + dim]);
        }
    }
    let stage_kv_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 13. Stage qBwd
    let t = Instant::now();
    kernels.bufs.q_bwd_dq.copy_from_f32(&ws.dq);
    kernels.bufs.q_bwd_wqt.copy_from_f32(&weights.wqt);
    let stage_q_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 14. ASYNC: ANE qBwd+kvBwd || CPU dWq+dWk+dWv
    // kvBwd is already staged (step 12.5), so both kernels run back-to-back on ANE
    // while the main thread computes weight gradients.
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .q_bwd_split
                .run_cached_direct(
                    &[&kernels.bufs.q_bwd_dq, &kernels.bufs.q_bwd_wqt],
                    &[&kernels.bufs.q_bwd_out],
                )
                .expect("ANE eval failed");
            kernels
                .kv_bwd
                .run_cached_direct(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out])
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &ws.dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &ws.dk_kv, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &ws.dv_kv, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        ws.dx_attn.copy_from_slice(&locked[..dim * seq]);
    }
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        ws.dx_kv.copy_from_slice(&locked[..dim * seq]);
    }
    let async_q_bwd_plus_dw_ms = t.elapsed().as_secs_f32() * 1000.0;
    let stage_run_kv_bwd_ms = stage_kv_bwd_ms;

    // 15. Merge + RMSNorm1 backward
    let t = Instant::now();
    vdsp::vadd(&ws.dx_attn, &ws.dx_kv, &mut ws.dx_merged);
    let merge_dx_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = Instant::now();
    rmsnorm::backward_channel_first(
        &ws.dx_merged,
        &cache.x,
        &weights.gamma1,
        &cache.rms_inv1,
        &mut ws.dx_rms1,
        &mut grads.dgamma1,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    let rmsnorm1_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 16. Final dx
    let mut dx = vec![0.0f32; dim * seq]; // only allocation — return value
    vdsp::vadd(&ws.dx_rms1, &ws.dx2, &mut dx);

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let timings = BackwardTimings {
        scale_dy_ms,
        stage_run_ffn_bwd_w2t_ms,
        silu_deriv_ms,
        stage_ffn_bwd_w13t_ms,
        async_ffn_bwd_w13t_plus_dw_ms,
        rmsnorm2_bwd_ms,
        stage_run_wot_bwd_ms,
        stage_sdpa_bwd1_ms,
        async_sdpa_bwd1_plus_dwo_ms,
        read_sdpa_bwd1_ms,
        stage_run_sdpa_bwd2_ms,
        rope_bwd_ms,
        stage_q_bwd_ms,
        async_q_bwd_plus_dw_ms,
        stage_run_kv_bwd_ms,
        rmsnorm1_bwd_ms,
        merge_dx_ms,
        total_ms,
    };

    (dx, timings)
}

// ── Backward pass ──

/// Run backward pass for one transformer layer.
/// `dy` is gradient of loss w.r.t. layer output [DIM * SEQ].
/// Returns `dx` (gradient w.r.t. layer input) and fills `grads`.
/// Uses pre-allocated workspace to eliminate ~32 vec allocations per call.
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
) -> Vec<f32> {
    let mut dx = vec![0.0f32; cfg.dim * cfg.seq];
    backward_into(cfg, kernels, weights, cache, dy, grads, ws, &mut dx);
    dx
}

/// Backward pass writing dx into pre-allocated buffer (zero allocations).
pub(crate) fn backward_into_with_ffn_cache(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    metal_ffn: Option<&MetalFFN>,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
    ffn_cache: &mut FfnBwdW13tCache,
    dx_out: &mut [f32],
) {
    backward_into_impl(
        cfg,
        kernels,
        metal_ffn,
        weights,
        cache,
        dy,
        grads,
        ws,
        Some(ffn_cache),
        dx_out,
    );
}

pub fn backward_into(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
    dx_out: &mut [f32],
) {
    backward_into_impl(
        cfg, kernels, None, weights, cache, dy, grads, ws, None, dx_out,
    );
}

fn backward_into_impl(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    metal_ffn: Option<&MetalFFN>,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
    ws: &mut BackwardWorkspace,
    mut ffn_cache: Option<&mut FfnBwdW13tCache>,
    dx_out: &mut [f32],
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
    let profile_ffn_cache = profile_ffn_cache_enabled();

    // 1. Scale dy
    vdsp::vsmul(dy, alpha, &mut ws.dffn);

    // 2. ffnBwdW2t
    kernels.bufs.ffn_bwd_w2t_dffn.copy_from_f32(&ws.dffn);
    kernels.bufs.ffn_bwd_w2t_w2.copy_from_f32(&weights.w2);
    let refresh_ffn_weights = metal_ffn.is_none()
        && ffn_cache
            .as_ref()
            .map_or(true, |cache| cache.staged_generation != weights.generation);
    if metal_ffn.is_none() {
        if let Some(cache) = ffn_cache.as_deref_mut() {
            if refresh_ffn_weights {
                cache.stats.weight_refreshes += 1;
            } else {
                cache.stats.weight_reuse_hits += 1;
            }
        }
    }
    // ASYNC: ANE ffnBwdW2t || optional staged-weight refresh + sigmoid precompute.
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .ffn_bwd_w2t
                .run_cached_direct(
                    &[&kernels.bufs.ffn_bwd_w2t_dffn, &kernels.bufs.ffn_bwd_w2t_w2],
                    &[&kernels.bufs.ffn_bwd_w2t_out],
                )
                .expect("ANE eval failed");
        });
        if metal_ffn.is_none() && refresh_ffn_weights {
            let t_weights = profile_ffn_cache.then(Instant::now);
            if let Some(cache) = ffn_cache.as_deref_mut() {
                cache.w1t.copy_from_f32(&weights.w1t);
                cache.w3t.copy_from_f32(&weights.w3t);
                cache.staged_generation = weights.generation;
                if let Some(t0) = t_weights {
                    cache.stats.stage_weights_ms += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }
        if cfg.ffn_activation == FfnActivation::SwiGlu {
            // Pre-compute sigmoid(h1) — doesn't need dsilu_raw (ANE output), safe to overlap
            vdsp::vsmul(&cache.h1, -1.0, &mut ws.neg_h1);
            vdsp::expf(&ws.neg_h1, &mut ws.exp_neg);
            vdsp::vsadd(&ws.exp_neg, 1.0, &mut ws.neg_h1); // neg_h1 = 1 + exp(-h1)
            vdsp::recf_inplace(&mut ws.neg_h1); // neg_h1 = sig = 1/(1+exp(-h1))
        }
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        ws.dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }

    // 3. Gate activation backward scalar loop.
    let n = hidden * seq;
    if cfg.ffn_activation == FfnActivation::SwiGlu {
        for i in 0..n {
            let sig = ws.neg_h1[i];
            let act = ffn_gate_activation(&cfg, cache.h1[i]);
            let act_deriv = ffn_gate_activation_derivative(&cfg, cache.h1[i], Some(sig));
            ws.dh3[i] = ws.dsilu_raw[i] * act;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * act_deriv;
        }
    } else {
        for i in 0..n {
            let act = ffn_gate_activation(&cfg, cache.h1[i]);
            let act_deriv = ffn_gate_activation_derivative(&cfg, cache.h1[i], None);
            ws.dh3[i] = ws.dsilu_raw[i] * act;
            ws.dh1[i] = ws.dsilu_raw[i] * cache.h3[i] * act_deriv;
        }
    }

    if let Some(metal_ffn) = metal_ffn {
        metal_ffn.backward_dx_into(
            cfg,
            &ws.dh1,
            &ws.dh3,
            &weights.w1,
            &weights.w3,
            &mut ws.dx_ffn,
        );
        accumulate_dw(&cache.x2norm, dim, &ws.dh3, hidden, seq, &mut grads.dw3);
    } else {
        // 4. Stage ffnBwdW13t split inputs — flat copies avoid the packed-row write hotspot.
        if let Some(cache) = ffn_cache.as_deref_mut() {
            let t_acts = profile_ffn_cache.then(Instant::now);
            cache.dh1.copy_from_f32(&ws.dh1);
            cache.dh3.copy_from_f32(&ws.dh3);
            if let Some(t0) = t_acts {
                cache.stats.stage_activations_ms += t0.elapsed().as_secs_f64() * 1000.0;
            }
        } else {
            kernels.bufs.ffn_bwd_w13t_dh1.copy_from_f32(&ws.dh1);
            kernels.bufs.ffn_bwd_w13t_dh3.copy_from_f32(&ws.dh3);
            kernels.bufs.ffn_bwd_w13t_w1t.copy_from_f32(&weights.w1t);
            kernels.bufs.ffn_bwd_w13t_w3t.copy_from_f32(&weights.w3t);
        }

        // 5. ASYNC: ANE ffnBwdW13t || CPU dW3
        std::thread::scope(|s| {
            let ane_handle = s.spawn(|| {
                if let Some(cache) = ffn_cache.as_deref() {
                    kernels
                        .ffn_bwd_w13t_split
                        .run_cached_direct(
                            &[&cache.dh1, &cache.dh3, &cache.w1t, &cache.w3t],
                            &[&kernels.bufs.ffn_bwd_w13t_out],
                        )
                        .expect("ANE eval failed");
                } else {
                    kernels
                        .ffn_bwd_w13t_split
                        .run_cached_direct(
                            &[
                                &kernels.bufs.ffn_bwd_w13t_dh1,
                                &kernels.bufs.ffn_bwd_w13t_dh3,
                                &kernels.bufs.ffn_bwd_w13t_w1t,
                                &kernels.bufs.ffn_bwd_w13t_w3t,
                            ],
                            &[&kernels.bufs.ffn_bwd_w13t_out],
                        )
                        .expect("ANE eval failed");
                }
            });
            accumulate_dw(&cache.x2norm, dim, &ws.dh3, hidden, seq, &mut grads.dw3);
            ane_handle.join().expect("ANE thread panicked");
        });
        {
            let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
            ws.dx_ffn.copy_from_slice(&locked[..dim * seq]);
        }
    }

    // 6. RMSNorm2 backward
    rmsnorm::backward_channel_first(
        &ws.dx_ffn,
        &cache.x2,
        &weights.gamma2,
        &cache.rms_inv2,
        &mut ws.dx2,
        &mut grads.dgamma2,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );
    vdsp::vadd(&ws.dx2, dy, &mut ws.dx2_tmp);
    ws.dx2.copy_from_slice(&ws.dx2_tmp);

    // 7. wotBwd
    vdsp::vsmul(&ws.dx2, alpha, &mut ws.dx2_scaled);
    {
        kernels.bufs.wot_bwd_dx2.copy_from_f32(&ws.dx2_scaled);
        kernels.bufs.wot_bwd_wot.copy_from_f32(&weights.wot);
    }
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    // ASYNC: ANE wotBwd || pre-stage 3 of 4 sdpaBwd1 inputs (from forward cache)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .wot_bwd_split
                .run_cached_direct(
                    &[&kernels.bufs.wot_bwd_dx2, &kernels.bufs.wot_bwd_wot],
                    &[&kernels.bufs.wot_bwd_out],
                )
                .expect("ANE eval failed");
        });
        kernels.bufs.sdpa_bwd1_q.copy_from_f32(&cache.q_rope);
        kernels.bufs.sdpa_bwd1_k.copy_from_f32(&cache.k_rope);
        kernels.bufs.sdpa_bwd1_v.copy_from_f32(&cache.v);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        ws.da.copy_from_slice(&locked[..q_dim * seq]);
    }

    // 8. Stage remaining sdpaBwd1 input (da depends on wotBwd output)
    kernels.bufs.sdpa_bwd1_da.copy_from_f32(&ws.da);

    // 9. ASYNC: ANE sdpaBwd1 || CPU dWo + dW2 (moved from step 5 to rebalance CPU load)
    // Step 5 was CPU-bound at ~2.3ms (3 sgemm); step 9 had ~0.6ms ANE headroom.
    // dW2 = dffn @ gate^T — both available since step 1 (dffn) and forward cache (gate).
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd1
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_bwd1_q,
                        &kernels.bufs.sdpa_bwd1_k,
                        &kernels.bufs.sdpa_bwd1_v,
                        &kernels.bufs.sdpa_bwd1_da,
                    ],
                    &[&kernels.bufs.sdpa_bwd1_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(
            &cache.attn_out,
            q_dim,
            &ws.dx2_scaled,
            dim,
            seq,
            &mut grads.dwo,
        );
        accumulate_dw(&ws.dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        ane_handle.join().expect("ANE thread panicked");
    });

    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut ws.dv_full);
    }

    // 10. sdpaBwd2
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    kernels.bufs.sdpa_bwd2_q.copy_from_f32(&cache.q_rope);
    kernels.bufs.sdpa_bwd2_k.copy_from_f32(&cache.k_rope);
    // ASYNC: ANE sdpaBwd2 || dW1 (moved from step 5 to rebalance)
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .sdpa_bwd2
                .run_cached_direct(
                    &[
                        &kernels.bufs.sdpa_bwd1_out,
                        &kernels.bufs.sdpa_bwd2_q,
                        &kernels.bufs.sdpa_bwd2_k,
                    ],
                    &[&kernels.bufs.sdpa_bwd2_out],
                )
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.x2norm, dim, &ws.dh1, hidden, seq, &mut grads.dw1);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut ws.dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut ws.dk);
    }

    // 11. RoPE backward
    rope_backward_inplace(&mut ws.dq, heads, hd, seq, &kernels.rope);
    rope_backward_inplace(&mut ws.dk, heads, hd, seq, &kernels.rope);
    reduce_gqa_grads(cfg, &ws.dk, &ws.dv_full, &mut ws.dk_kv, &mut ws.dv_kv);

    // 11.5. Stage kvBwd early (dk post-RoPE)
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        for c in 0..kv_dim {
            let row = c * kv_bwd_sp;
            buf[row..row + seq].copy_from_slice(&ws.dk_kv[c * seq..c * seq + seq]);
            buf[row + seq..row + 2 * seq].copy_from_slice(&ws.dv_kv[c * seq..c * seq + seq]);
            buf[row + 2 * seq..row + 2 * seq + dim]
                .copy_from_slice(&weights.wkt[c * dim..c * dim + dim]);
            buf[row + 2 * seq + dim..row + 2 * seq + 2 * dim]
                .copy_from_slice(&weights.wvt[c * dim..c * dim + dim]);
        }
    }

    // 12. Stage qBwd
    kernels.bufs.q_bwd_dq.copy_from_f32(&ws.dq);
    kernels.bufs.q_bwd_wqt.copy_from_f32(&weights.wqt);

    // 13. ASYNC: ANE qBwd+kvBwd || CPU dWq+dWk+dWv
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels
                .q_bwd_split
                .run_cached_direct(
                    &[&kernels.bufs.q_bwd_dq, &kernels.bufs.q_bwd_wqt],
                    &[&kernels.bufs.q_bwd_out],
                )
                .expect("ANE eval failed");
            kernels
                .kv_bwd
                .run_cached_direct(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out])
                .expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &ws.dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &ws.dk_kv, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &ws.dv_kv, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        ws.dx_attn.copy_from_slice(&locked[..dim * seq]);
    }
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        ws.dx_kv.copy_from_slice(&locked[..dim * seq]);
    }
    vdsp::vadd(&ws.dx_attn, &ws.dx_kv, &mut ws.dx_merged);

    // 15. Merge + RMSNorm1 backward
    rmsnorm::backward_channel_first(
        &ws.dx_merged,
        &cache.x,
        &weights.gamma1,
        &cache.rms_inv1,
        &mut ws.dx_rms1,
        &mut grads.dgamma1,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );

    // 16. Final dx into pre-allocated buffer
    vdsp::vadd(&ws.dx_rms1, &ws.dx2, dx_out);
}

// ── CPU helpers ──

/// Accumulate weight gradient via BLAS: dW[a_ch, b_ch] += A[a_ch, seq] @ B[b_ch, seq]^T
/// `a` is [a_ch * seq] row-major, `b` is [b_ch * seq] row-major, `dw` is [a_ch * b_ch].
fn accumulate_dw(a: &[f32], a_ch: usize, b: &[f32], b_ch: usize, seq: usize, dw: &mut [f32]) {
    vdsp::sgemm_at(a, a_ch, seq, b, b_ch, dw);
}

fn profile_ffn_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUSTANE_PROFILE_FFN_CACHE")
            .ok()
            .map(|v| {
                matches!(
                    v.as_str(),
                    "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
                )
            })
            .unwrap_or(false)
    })
}

/// RoPE backward: inverse rotation applied in-place.
/// Uses cached cos/sin tables from CompiledKernels (computed once at init).
fn rope_backward_inplace(dx: &mut [f32], heads: usize, hd: usize, seq: usize, rope: &RopeTable) {
    let pairs = hd / 2;
    for h in 0..heads {
        for i in 0..pairs {
            let base0 = (h * hd + 2 * i) * seq;
            let base1 = (h * hd + 2 * i + 1) * seq;
            let tbase = i * seq;
            for p in 0..seq {
                let c = rope.cos[tbase + p];
                let s = rope.sin[tbase + p];
                let d0 = dx[base0 + p];
                let d1 = dx[base1 + p];
                dx[base0 + p] = c * d0 + s * d1;
                dx[base1 + p] = -s * d0 + c * d1;
            }
        }
    }
}

fn reduce_gqa_grads(
    cfg: &ModelConfig,
    dk_tiled: &[f32],
    dv_tiled: &[f32],
    dk_kv: &mut [f32],
    dv_kv: &mut [f32],
) {
    let kv_elems = cfg.kv_dim * cfg.seq;
    if cfg.gqa_ratio == 1 {
        dk_kv[..kv_elems].copy_from_slice(&dk_tiled[..kv_elems]);
        dv_kv[..kv_elems].copy_from_slice(&dv_tiled[..kv_elems]);
        return;
    }

    dk_kv[..kv_elems].fill(0.0);
    dv_kv[..kv_elems].fill(0.0);
    let head_span = cfg.hd * cfg.seq;
    for kv_head in 0..cfg.kv_heads {
        let dst = kv_head * head_span;
        for group in 0..cfg.gqa_ratio {
            let src = (kv_head * cfg.gqa_ratio + group) * head_span;
            for i in 0..head_span {
                dk_kv[dst + i] += dk_tiled[src + i];
                dv_kv[dst + i] += dv_tiled[src + i];
            }
        }
    }
}
