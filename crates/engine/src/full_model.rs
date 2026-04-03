//! Full transformer model: embedding → NL layers → final norm → logits → loss.
//!
//! gpt_karpathy: 6 layers, 48.8M params, SwiGLU, MHA, RoPE.
//! Tied embedding weights (embedding table = output projection transposed).

use crate::cpu::{cross_entropy, embedding, rmsnorm, vdsp};
use crate::layer::{
    self, BackwardWorkspace, CompiledKernels, FfnBwdW13tCache, FfnBwdW13tCacheStats, ForwardCache,
    LayerGrads, LayerWeights,
};
use crate::metal_adam::MetalAdam;
use crate::metal_ffn::MetalFFN;
use crate::metal_muon::MetalMuon;
use crate::model::ModelConfig;
use crate::training::LayerOptState;

/// Full model weights.
#[derive(Clone)]
pub struct ModelWeights {
    pub embed: Vec<f32>, // [VOCAB * DIM]
    pub layers: Vec<LayerWeights>,
    pub gamma_final: Vec<f32>, // [DIM]
}

/// Full model gradients.
pub struct ModelGrads {
    pub dembed: Vec<f32>, // [VOCAB * DIM]
    pub layers: Vec<LayerGrads>,
    pub dgamma_final: Vec<f32>,
}

/// Full model optimizer state.
pub struct ModelOptState {
    pub embed_m: Vec<f32>,
    pub embed_v: Vec<f32>,
    pub layers: Vec<LayerOptState>,
    pub gamma_final_m: Vec<f32>,
    pub gamma_final_v: Vec<f32>,
}

/// Training hyperparameters (from Obj-C reference).
pub struct TrainConfig {
    pub max_lr: f32,
    pub min_lr_frac: f32,
    pub warmup_steps: u32,
    pub total_steps: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub matrix_lr_scale: f32,
    pub embed_lr_scale: f32,
    pub loss_scale: f32,
    pub accum_steps: u32,
    pub grad_clip: f32,
    pub softcap: f32,
    pub ema_decay: Option<f32>,
    pub swa_interval: Option<u32>,
    pub swa_start_frac: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_lr: 3e-4,
            min_lr_frac: 0.1,
            warmup_steps: 100,
            total_steps: 10000,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            matrix_lr_scale: 0.05,
            embed_lr_scale: 5.0,
            loss_scale: 256.0,
            accum_steps: 10,
            grad_clip: 1.0,
            softcap: 0.0, // was 15.0 — A/B test showed identical loss, saves ~5ms/microbatch
            ema_decay: None,
            swa_interval: None,
            swa_start_frac: None,
        }
    }
}

/// Muon optimizer hyperparameters for matrix-shaped layer weights.
#[derive(Clone, Copy, Debug)]
pub struct MuonConfig {
    pub muon_lr: f32,
    pub muon_momentum: f32,
    pub newton_schulz_steps: u32,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            muon_lr: 0.025,
            muon_momentum: 0.95,
            newton_schulz_steps: 5,
        }
    }
}

/// Forward-path execution toggles.
#[derive(Clone, Copy, Default)]
pub struct ForwardOptions<'a> {
    pub use_gpu_ffn: bool,
    pub metal_ffn: Option<&'a MetalFFN>,
}

impl<'a> ForwardOptions<'a> {
    pub fn gpu_ffn(metal_ffn: &'a MetalFFN) -> Self {
        Self {
            use_gpu_ffn: true,
            metal_ffn: Some(metal_ffn),
        }
    }
}

/// Backward-path execution toggles.
#[derive(Clone, Copy, Default)]
pub struct BackwardOptions<'a> {
    pub use_gpu_ffn_dx: bool,
    pub use_ffn_weight_cache: bool,
    pub metal_ffn: Option<&'a MetalFFN>,
}

impl<'a> BackwardOptions<'a> {
    pub fn gpu_ffn_dx(metal_ffn: &'a MetalFFN) -> Self {
        Self {
            use_gpu_ffn_dx: true,
            metal_ffn: Some(metal_ffn),
            ..Self::default()
        }
    }

    pub fn with_ffn_weight_cache(mut self) -> Self {
        self.use_ffn_weight_cache = true;
        self
    }
}

/// Everything needed from forward pass for backward.
pub struct ForwardResult {
    pub loss: f32,
    pub caches: Vec<ForwardCache>,
    pub dlogits: Vec<f32>,       // [SEQ * VOCAB]
    pub x_final: Vec<f32>,       // [DIM * SEQ] channel-first, post-norm
    pub x_prenorm: Vec<f32>,     // [DIM * SEQ] channel-first, pre-final-norm
    pub rms_inv_final: Vec<f32>, // [SEQ]
    pub logits_capped: Vec<f32>, // [SEQ * VOCAB] post-softcap (empty if no softcap)
}

impl ModelWeights {
    pub fn random(cfg: &ModelConfig) -> Self {
        Self {
            embed: random_vec(cfg.vocab * cfg.dim, 0.005), // Obj-C E50 value
            layers: (0..cfg.nlayers)
                .map(|_| LayerWeights::random(cfg))
                .collect(),
            gamma_final: vec![1.0; cfg.dim],
        }
    }

    pub fn blend_towards(&mut self, src: &Self, amount: f32) {
        debug_assert!((0.0..=1.0).contains(&amount));
        if amount <= 0.0 {
            return;
        }
        if amount >= 1.0 {
            *self = src.clone();
            return;
        }

        blend_slice_towards(&mut self.embed, &src.embed, amount);
        blend_slice_towards(&mut self.gamma_final, &src.gamma_final, amount);
        for (dst_layer, src_layer) in self.layers.iter_mut().zip(&src.layers) {
            blend_slice_towards(&mut dst_layer.wq, &src_layer.wq, amount);
            blend_slice_towards(&mut dst_layer.wk, &src_layer.wk, amount);
            blend_slice_towards(&mut dst_layer.wv, &src_layer.wv, amount);
            blend_slice_towards(&mut dst_layer.wo, &src_layer.wo, amount);
            blend_slice_towards(&mut dst_layer.w1, &src_layer.w1, amount);
            blend_slice_towards(&mut dst_layer.w3, &src_layer.w3, amount);
            blend_slice_towards(&mut dst_layer.w2, &src_layer.w2, amount);
            blend_slice_towards(&mut dst_layer.wqt, &src_layer.wqt, amount);
            blend_slice_towards(&mut dst_layer.wkt, &src_layer.wkt, amount);
            blend_slice_towards(&mut dst_layer.wvt, &src_layer.wvt, amount);
            blend_slice_towards(&mut dst_layer.wot, &src_layer.wot, amount);
            blend_slice_towards(&mut dst_layer.w1t, &src_layer.w1t, amount);
            blend_slice_towards(&mut dst_layer.w3t, &src_layer.w3t, amount);
            blend_slice_towards(&mut dst_layer.gamma1, &src_layer.gamma1, amount);
            blend_slice_towards(&mut dst_layer.gamma2, &src_layer.gamma2, amount);
            dst_layer.generation = src_layer.generation;
        }
    }
}

impl ModelGrads {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            dembed: vec![0.0; cfg.vocab * cfg.dim],
            layers: (0..cfg.nlayers).map(|_| LayerGrads::zeros(cfg)).collect(),
            dgamma_final: vec![0.0; cfg.dim],
        }
    }

    pub fn zero_out(&mut self) {
        self.dembed.fill(0.0);
        self.dgamma_final.fill(0.0);
        for lg in &mut self.layers {
            lg.zero_out();
        }
    }
}

impl ModelOptState {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            embed_m: vec![0.0; cfg.vocab * cfg.dim],
            embed_v: vec![0.0; cfg.vocab * cfg.dim],
            layers: (0..cfg.nlayers)
                .map(|_| LayerOptState::zeros(cfg))
                .collect(),
            gamma_final_m: vec![0.0; cfg.dim],
            gamma_final_v: vec![0.0; cfg.dim],
        }
    }
}

/// Pre-allocated workspace for full model backward pass.
/// Eliminates ~1.2 GB of per-call allocations (dl, dx_final, dy, rms_dot_buf, layer workspace).
pub struct ModelBackwardWorkspace {
    pub dl: Vec<f32>,          // [SEQ * VOCAB]
    pub dx_final: Vec<f32>,    // [DIM * SEQ]
    pub dy: Vec<f32>,          // [DIM * SEQ]
    pub dy_buf: Vec<f32>,      // [DIM * SEQ] swap buffer for backward_into
    pub rms_dot_buf: Vec<f32>, // [SEQ]
    pub layer_ws: BackwardWorkspace,
    pub ffn_bwd_w13t: Vec<FfnBwdW13tCache>,
}

impl ModelBackwardWorkspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            dl: vec![0.0; cfg.seq * cfg.vocab],
            dx_final: vec![0.0; cfg.dim * cfg.seq],
            dy: vec![0.0; cfg.dim * cfg.seq],
            dy_buf: vec![0.0; cfg.dim * cfg.seq],
            rms_dot_buf: vec![0.0; cfg.seq],
            layer_ws: BackwardWorkspace::new(cfg),
            ffn_bwd_w13t: (0..cfg.nlayers)
                .map(|_| FfnBwdW13tCache::new(cfg))
                .collect(),
        }
    }

    pub fn reset_ffn_bwd_w13t_stats(&mut self) {
        for cache in &mut self.ffn_bwd_w13t {
            cache.reset_stats();
        }
    }

    pub fn ffn_bwd_w13t_stats(&self) -> FfnBwdW13tCacheStats {
        let mut total = FfnBwdW13tCacheStats::default();
        for cache in &self.ffn_bwd_w13t {
            let stats = cache.stats();
            total.weight_refreshes += stats.weight_refreshes;
            total.weight_reuse_hits += stats.weight_reuse_hits;
            total.stage_weights_ms += stats.stage_weights_ms;
            total.stage_activations_ms += stats.stage_activations_ms;
        }
        total
    }
}

/// Pre-allocated workspace for full model forward pass.
/// Eliminates ~144MB of per-step alloc/zero/free for layer caches +
/// ~37MB for model-level buffers (logits, dlogits, etc.).
pub struct ModelForwardWorkspace {
    pub caches: Vec<ForwardCache>,
    pub x_row: Vec<f32>,         // [SEQ * DIM]
    pub x_buf: Vec<f32>,         // [DIM * SEQ] current layer input
    pub x_next_buf: Vec<f32>,    // [DIM * SEQ] current layer output
    pub x_final: Vec<f32>,       // [DIM * SEQ] post final rmsnorm
    pub x_prenorm: Vec<f32>,     // [DIM * SEQ] pre final rmsnorm (= last layer output)
    pub rms_inv_final: Vec<f32>, // [SEQ]
    pub x_final_row: Vec<f32>,   // [SEQ * DIM]
    pub logits: Vec<f32>,        // [SEQ * VOCAB]
    pub logits_capped: Vec<f32>, // [SEQ * VOCAB] (for softcap backward)
    pub dlogits: Vec<f32>,       // [SEQ * VOCAB]
}

impl ModelForwardWorkspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let vocab = cfg.vocab;
        Self {
            caches: (0..cfg.nlayers).map(|_| ForwardCache::new(cfg)).collect(),
            x_row: vec![0.0; seq * dim],
            x_buf: vec![0.0; dim * seq],
            x_next_buf: vec![0.0; dim * seq],
            x_final: vec![0.0; dim * seq],
            x_prenorm: vec![0.0; dim * seq],
            rms_inv_final: vec![0.0; seq],
            x_final_row: vec![0.0; seq * dim],
            logits: vec![0.0; seq * vocab],
            logits_capped: vec![0.0; seq * vocab],
            dlogits: vec![0.0; seq * vocab],
        }
    }
}

/// Forward pass through embeddings/layers/final norm/output projection, writing logits into `ws`.
/// `tokens` must be length `cfg.seq`.
pub fn forward_logits_ws(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    softcap: f32,
    ws: &mut ModelForwardWorkspace,
) {
    forward_logits_ws_with_options(
        cfg,
        kernels,
        weights,
        tokens,
        softcap,
        ws,
        ForwardOptions::default(),
    )
}

/// Forward-only pass with optional GPU FFN acceleration, writing logits into `ws`.
pub fn forward_logits_ws_with_options(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    softcap: f32,
    ws: &mut ModelForwardWorkspace,
    opts: ForwardOptions<'_>,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;
    let use_gpu_ffn = opts.use_gpu_ffn && opts.metal_ffn.is_some();

    assert_eq!(tokens.len(), seq, "tokens must match cfg.seq");

    // 1. Embedding lookup → x_buf [DIM, SEQ]
    embedding::forward(&weights.embed, dim, tokens, &mut ws.x_row);
    vdsp::mtrans(&ws.x_row, dim, &mut ws.x_buf, seq, seq, dim);

    // 2. Forward through NL layers.
    if use_gpu_ffn {
        let metal_ffn = opts.metal_ffn.expect("gpu FFN requested");
        for l in 0..cfg.nlayers {
            layer::forward_into_gpu_ffn(
                cfg,
                kernels,
                metal_ffn,
                &weights.layers[l],
                &ws.x_buf,
                &mut ws.caches[l],
                &mut ws.x_next_buf,
            );
            std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
        }
    } else {
        // Each layer defers its h1/h3/gate readback (12MB). The NEXT layer reads them
        // during its sdpaFwd ANE overlap (step 3, ~1.7ms spare CPU time).
        for l in 0..cfg.nlayers {
            if l > 0 {
                let (prev_caches, curr_caches) = ws.caches.split_at_mut(l);
                layer::forward_into_pipelined(
                    cfg,
                    kernels,
                    &weights.layers[l],
                    &ws.x_buf,
                    &mut curr_caches[0],
                    &mut ws.x_next_buf,
                    Some(&mut prev_caches[l - 1]),
                );
            } else {
                layer::forward_into_pipelined(
                    cfg,
                    kernels,
                    &weights.layers[0],
                    &ws.x_buf,
                    &mut ws.caches[0],
                    &mut ws.x_next_buf,
                    None,
                );
            }
            std::mem::swap(&mut ws.x_buf, &mut ws.x_next_buf);
        }
        // Read last layer's deferred h1/h3/gate
        layer::read_ffn_cache(cfg, kernels, &mut ws.caches[cfg.nlayers - 1]);
    }

    // 3. x_buf now holds the last layer's output = x_prenorm
    ws.x_prenorm.copy_from_slice(&ws.x_buf);

    // 4. Final RMSNorm
    rmsnorm::forward_channel_first(
        &ws.x_prenorm,
        &weights.gamma_final,
        &mut ws.x_final,
        &mut ws.rms_inv_final,
        dim,
        seq,
    );

    // 5. Logits: x_final^T @ embed^T → [SEQ, VOCAB]
    //    sgemm_at uses beta=1.0 (C += A@B^T), so zero logits first to prevent
    //    accumulation across repeated inference calls.
    vdsp::mtrans(&ws.x_final, seq, &mut ws.x_final_row, dim, dim, seq);
    ws.logits.fill(0.0);
    vdsp::sgemm_at(
        &ws.x_final_row,
        seq,
        dim,
        &weights.embed,
        vocab,
        &mut ws.logits,
    );

    // 6. Softcap: logits = softcap * tanh(logits / softcap)
    //    Store unscaled tanh in logits_capped for backward (avoids extra pass + simplifies derivative).
    if softcap > 0.0 {
        vdsp::sscal(&mut ws.logits, 1.0 / softcap);
        vdsp::tanhf(&ws.logits, &mut ws.logits_capped);
        vdsp::vsmul(&ws.logits_capped, softcap, &mut ws.logits);
    }
}

/// Forward pass using pre-allocated workspace (zero heap allocations in steady state).
/// Returns loss. All intermediate data lives in `ws` for backward_ws to read.
pub fn forward_ws(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
    ws: &mut ModelForwardWorkspace,
) -> f32 {
    forward_ws_with_options(
        cfg,
        kernels,
        weights,
        tokens,
        targets,
        softcap,
        ws,
        ForwardOptions::default(),
    )
}

/// Forward pass using pre-allocated workspace and optional GPU FFN acceleration.
pub fn forward_ws_with_options(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
    ws: &mut ModelForwardWorkspace,
    opts: ForwardOptions<'_>,
) -> f32 {
    let vocab = cfg.vocab;
    let seq = cfg.seq;

    forward_logits_ws_with_options(cfg, kernels, weights, tokens, softcap, ws, opts);

    // 7. Cross-entropy
    let total_loss = cross_entropy::forward_backward_batch(
        &ws.logits,
        targets,
        vocab,
        &mut ws.dlogits,
        1.0 / seq as f32,
    );

    total_loss / seq as f32
}

/// Backward pass using pre-allocated workspaces (reads from fwd_ws, writes grads).
pub fn backward_ws(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    fwd_ws: &ModelForwardWorkspace,
    tokens: &[u32],
    softcap: f32,
    loss_scale: f32,
    grads: &mut ModelGrads,
    bwd_ws: &mut ModelBackwardWorkspace,
) {
    backward_ws_with_options(
        cfg,
        kernels,
        weights,
        fwd_ws,
        tokens,
        softcap,
        loss_scale,
        grads,
        bwd_ws,
        BackwardOptions::default(),
    )
}

pub fn backward_ws_with_options(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    fwd_ws: &ModelForwardWorkspace,
    tokens: &[u32],
    softcap: f32,
    loss_scale: f32,
    grads: &mut ModelGrads,
    bwd_ws: &mut ModelBackwardWorkspace,
    opts: BackwardOptions<'_>,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;
    let gpu_ffn_dx = if opts.use_gpu_ffn_dx {
        opts.metal_ffn
    } else {
        None
    };
    let use_ffn_weight_cache = opts.use_ffn_weight_cache;

    // 1+2. Fused scale + softcap backward
    //    logits_capped stores tanh(logits/softcap) — unscaled, so sech^2 = 1 - t^2
    if softcap > 0.0 {
        let dl = &mut bwd_ws.dl;
        let dlog = &fwd_ws.dlogits;
        let t = &fwd_ws.logits_capped; // tanh(logits/softcap), NOT scaled by softcap
        for i in 0..dl.len() {
            dl[i] = dlog[i] * loss_scale * (1.0 - t[i] * t[i]);
        }
    } else {
        vdsp::vsmul(&fwd_ws.dlogits, loss_scale, &mut bwd_ws.dl);
    }

    // 3a. dx_final = embed^T @ dl (needed for RMSNorm backward, sequential)
    unsafe {
        vdsp::cblas_sgemm(
            101,
            112,
            112,
            dim as i32,
            seq as i32,
            vocab as i32,
            1.0,
            weights.embed.as_ptr(),
            dim as i32,
            bwd_ws.dl.as_ptr(),
            vocab as i32,
            0.0,
            bwd_ws.dx_final.as_mut_ptr(),
            seq as i32,
        );
    }

    // 3b+4+5. ASYNC: dembed sgemm overlaps with RMSNorm backward + layer backward.
    // Safety: dembed is not accessed by rmsnorm backward (writes dy, dgamma_final)
    // or layer backward (writes grads.layers[l]) — disjoint struct fields.
    // Using usize casts for Send safety across thread boundary.
    let dl_addr = bwd_ws.dl.as_ptr() as usize;
    let xf_addr = fwd_ws.x_final.as_ptr() as usize;
    let de_addr = grads.dembed.as_mut_ptr() as usize;
    let (sv, sd, ss) = (vocab as i32, dim as i32, seq as i32);
    std::thread::scope(|s| {
        let sgemm_handle = s.spawn(move || unsafe {
            vdsp::cblas_sgemm(
                101,
                112,
                112,
                sv,
                sd,
                ss,
                1.0,
                dl_addr as *const f32,
                sv,
                xf_addr as *const f32,
                ss,
                1.0,
                de_addr as *mut f32,
                sd,
            );
        });

        // 4. Final RMSNorm backward (main thread)
        rmsnorm::backward_channel_first(
            &bwd_ws.dx_final,
            &fwd_ws.x_prenorm,
            &weights.gamma_final,
            &fwd_ws.rms_inv_final,
            &mut bwd_ws.dy,
            &mut grads.dgamma_final,
            dim,
            seq,
            &mut bwd_ws.rms_dot_buf,
        );

        // 5. Backward through NL layers (main thread)
        for l in (0..cfg.nlayers).rev() {
            if use_ffn_weight_cache || gpu_ffn_dx.is_some() {
                layer::backward_into_with_ffn_cache(
                    cfg,
                    kernels,
                    gpu_ffn_dx,
                    &weights.layers[l],
                    &fwd_ws.caches[l],
                    &bwd_ws.dy,
                    &mut grads.layers[l],
                    &mut bwd_ws.layer_ws,
                    &mut bwd_ws.ffn_bwd_w13t[l],
                    &mut bwd_ws.dy_buf,
                );
            } else {
                layer::backward_into(
                    cfg,
                    kernels,
                    &weights.layers[l],
                    &fwd_ws.caches[l],
                    &bwd_ws.dy,
                    &mut grads.layers[l],
                    &mut bwd_ws.layer_ws,
                    &mut bwd_ws.dy_buf,
                );
            }
            std::mem::swap(&mut bwd_ws.dy, &mut bwd_ws.dy_buf);
        }

        sgemm_handle.join().expect("dembed sgemm thread panicked");
    });

    // 6. Embedding backward (after dembed sgemm completes)
    embedding::backward_channel_first(&bwd_ws.dy, dim, tokens, &mut grads.dembed);
}

/// Cosine LR schedule with linear warmup.
pub fn learning_rate(step: u32, tc: &TrainConfig) -> f32 {
    if step < tc.warmup_steps {
        tc.max_lr * (step + 1) as f32 / tc.warmup_steps as f32
    } else {
        let decay = (step - tc.warmup_steps) as f32 / (tc.total_steps - tc.warmup_steps) as f32;
        let min_lr = tc.max_lr * tc.min_lr_frac;
        min_lr + 0.5 * (1.0 + (std::f32::consts::PI * decay).cos()) * (tc.max_lr - min_lr)
    }
}

pub fn learning_rate_scale(step: u32, tc: &TrainConfig) -> f32 {
    if tc.max_lr <= 0.0 {
        0.0
    } else {
        learning_rate(step, tc) / tc.max_lr
    }
}

pub fn should_collect_swa(step_completed: u32, tc: &TrainConfig) -> bool {
    let Some(interval) = tc.swa_interval else {
        return false;
    };
    if interval == 0 || step_completed == 0 || step_completed % interval != 0 {
        return false;
    }
    let lr_step = step_completed - 1;
    if lr_step < tc.warmup_steps {
        return false;
    }
    if let Some(start_frac) = tc.swa_start_frac {
        let decay_steps = tc.total_steps.saturating_sub(tc.warmup_steps);
        if decay_steps == 0 {
            return false;
        }
        let decay_progress = (lr_step - tc.warmup_steps) as f32 / decay_steps as f32;
        return decay_progress >= (1.0 - start_frac);
    }
    learning_rate_scale(lr_step, tc) < 0.5
}

/// Full forward pass: embedding → NL layers → final norm → logits → loss.
pub fn forward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],  // [SEQ] input token IDs
    targets: &[u32], // [SEQ] target token IDs
    softcap: f32,
) -> ForwardResult {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1. Embedding lookup → x [DIM, SEQ] (channel-first)
    let mut x_row = vec![0.0f32; seq * dim]; // [SEQ, DIM] row-major
    embedding::forward(&weights.embed, dim, tokens, &mut x_row);
    // Transpose to channel-first [DIM, SEQ] via vDSP_mtrans
    let mut x = vec![0.0f32; dim * seq];
    vdsp::mtrans(&x_row, dim, &mut x, seq, seq, dim);

    // 2. Forward through NL layers
    let mut caches = Vec::with_capacity(cfg.nlayers);
    for l in 0..cfg.nlayers {
        let (x_next, cache) = layer::forward(cfg, kernels, &weights.layers[l], &x);
        caches.push(cache);
        x = x_next;
    }

    // 3. Final RMSNorm (CPU) — channel-first, no transpose
    let x_prenorm = x;
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv_final = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(
        &x_prenorm,
        &weights.gamma_final,
        &mut x_final,
        &mut rms_inv_final,
        dim,
        seq,
    );

    // 4. Logits: x_final^T @ embed^T → [SEQ, VOCAB]
    let mut x_final_row = vec![0.0f32; seq * dim];
    vdsp::mtrans(&x_final, seq, &mut x_final_row, dim, dim, seq);
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);

    // 5. Logit softcap: logits = softcap * tanh(logits / softcap)
    //    Store unscaled tanh in logits_capped for backward (sech^2 = 1 - t^2).
    let has_softcap = softcap > 0.0;
    let mut logits_capped = Vec::new();
    if has_softcap {
        logits_capped = vec![0.0f32; seq * vocab];
        vdsp::sscal(&mut logits, 1.0 / softcap);
        vdsp::tanhf(&logits, &mut logits_capped); // logits_capped = tanh(logits/softcap)
        vdsp::vsmul(&logits_capped, softcap, &mut logits); // logits = softcap * tanh(...)
    }

    // 6. Cross-entropy loss (batched — single alloc for all positions)
    let mut dlogits = vec![0.0f32; seq * vocab];
    let total_loss = cross_entropy::forward_backward_batch(
        &logits,
        targets,
        vocab,
        &mut dlogits,
        1.0 / seq as f32,
    );

    ForwardResult {
        loss: total_loss / seq as f32,
        caches,
        dlogits,
        x_final,
        x_prenorm,
        rms_inv_final,
        logits_capped,
    }
}

/// Full backward pass: logits grad → softcap → final norm → NL layers (reverse) → embedding.
/// Takes a pre-allocated workspace to avoid per-call allocations (~1.2 GB churn).
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    fwd: &ForwardResult,
    tokens: &[u32], // input token IDs (for embedding backward)
    softcap: f32,
    loss_scale: f32,
    grads: &mut ModelGrads,
    ws: &mut ModelBackwardWorkspace,
) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1+2. Fused scale + softcap backward
    //    logits_capped stores tanh(logits/softcap) — unscaled, so sech^2 = 1 - t^2
    if softcap > 0.0 && !fwd.logits_capped.is_empty() {
        let dl = &mut ws.dl;
        let dlog = &fwd.dlogits;
        let t = &fwd.logits_capped;
        for i in 0..dl.len() {
            dl[i] = dlog[i] * loss_scale * (1.0 - t[i] * t[i]);
        }
    } else {
        vdsp::vsmul(&fwd.dlogits, loss_scale, &mut ws.dl);
    }

    // 3. Output projection gradients (tied embedding weights)
    //    dembed += dl^T @ x_final^T: use BLAS trans flags, no mtrans needed
    unsafe {
        vdsp::cblas_sgemm(
            101,
            112,
            112, // row-major, transA(dl), transB(x_final)
            vocab as i32,
            dim as i32,
            seq as i32,
            1.0,
            ws.dl.as_ptr(),
            vocab as i32,
            fwd.x_final.as_ptr(),
            seq as i32,
            1.0,
            grads.dembed.as_mut_ptr(),
            dim as i32,
        );
    }
    //    dx_final = embed^T @ dl^T → directly in [dim,seq] channel-first, no mtrans
    unsafe {
        vdsp::cblas_sgemm(
            101,
            112,
            112,
            dim as i32,
            seq as i32,
            vocab as i32,
            1.0,
            weights.embed.as_ptr(),
            dim as i32,
            ws.dl.as_ptr(),
            vocab as i32,
            0.0,
            ws.dx_final.as_mut_ptr(),
            seq as i32,
        );
    }

    // 4. Final RMSNorm backward (CPU) — channel-first, no transpose
    rmsnorm::backward_channel_first(
        &ws.dx_final,
        &fwd.x_prenorm,
        &weights.gamma_final,
        &fwd.rms_inv_final,
        &mut ws.dy,
        &mut grads.dgamma_final,
        dim,
        seq,
        &mut ws.rms_dot_buf,
    );

    // 5. Backward through NL layers (reverse order, pre-allocated workspace)
    for l in (0..cfg.nlayers).rev() {
        // layer::backward returns dx into a new Vec; copy into ws.dy for next layer
        let dx = layer::backward(
            cfg,
            kernels,
            &weights.layers[l],
            &fwd.caches[l],
            &ws.dy,
            &mut grads.layers[l],
            &mut ws.layer_ws,
        );
        ws.dy = dx;
    }

    // 6. Input embedding backward: channel-first scatter-add, no mtrans
    embedding::backward_channel_first(&ws.dy, dim, tokens, &mut grads.dembed);
}

/// Apply Adam to all model weights with split learning rates.
/// Uses 2-thread parallelism: step_fused is compute-bound (sqrt+div per element),
/// and single-threaded achieves only ~80 GB/s of M4 Max's 400 GB/s bandwidth.
/// `grad_scale` is applied to gradients inline (fused descale+clip).
/// Pass 1.0 for no scaling, or the combined descale/clip factor.
pub fn update_weights(
    cfg: &ModelConfig,
    weights: &mut ModelWeights,
    grads: &ModelGrads,
    opt: &mut ModelOptState,
    t: u32,
    lr: f32,
    tc: &TrainConfig,
    _metal_adam: &MetalAdam,
    grad_scale: f32,
) {
    use crate::cpu::adam::step_fused;

    let wd = tc.weight_decay;
    let matrix_lr = lr * tc.matrix_lr_scale;
    let embed_lr = lr * tc.embed_lr_scale;
    let (b1, b2, eps) = (tc.beta1, tc.beta2, tc.eps);

    // Destructure into independent borrows for safe parallel access.
    let ModelWeights {
        embed: ref mut w_embed,
        layers: ref mut w_layers,
        gamma_final: ref mut w_gf,
    } = *weights;
    let ModelGrads {
        dembed: ref g_embed,
        layers: ref g_layers,
        dgamma_final: ref g_gf,
    } = *grads;
    let ModelOptState {
        embed_m: ref mut o_em,
        embed_v: ref mut o_ev,
        layers: ref mut o_layers,
        gamma_final_m: ref mut o_gfm,
        gamma_final_v: ref mut o_gfv,
    } = *opt;

    // 6-way split: embed+gamma | g1 | g2 | g3 | g4 | g5.
    // Keep the 5 layer groups balanced and run the embedding update separately so
    // no worker carries both embed and a full layer shard.
    let nl = cfg.nlayers;
    let g1 = nl / 5;
    let g2 = (nl - g1) / 4;
    let g3 = (nl - g1 - g2) / 3;
    let g4 = (nl - g1 - g2 - g3) / 2;
    // g5 = nl - g1 - g2 - g3 - g4 (remainder on main thread)

    let (wl_a, wl_rest) = w_layers.split_at_mut(g1);
    let (wl_b, wl_rest2) = wl_rest.split_at_mut(g2);
    let (wl_c, wl_rest3) = wl_rest2.split_at_mut(g3);
    let (wl_d, wl_e) = wl_rest3.split_at_mut(g4);
    let (gl_a, gl_rest) = g_layers.split_at(g1);
    let (gl_b, gl_rest2) = gl_rest.split_at(g2);
    let (gl_c, gl_rest3) = gl_rest2.split_at(g3);
    let (gl_d, gl_e) = gl_rest3.split_at(g4);
    let (ol_a, ol_rest) = o_layers.split_at_mut(g1);
    let (ol_b, ol_rest2) = ol_rest.split_at_mut(g2);
    let (ol_c, ol_rest3) = ol_rest2.split_at_mut(g3);
    let (ol_d, ol_e) = ol_rest3.split_at_mut(g4);

    std::thread::scope(|s| {
        let h0 = s.spawn(move || {
            step_fused(
                w_embed, g_embed, o_em, o_ev, t, embed_lr, b1, b2, eps, 0.0, grad_scale,
            );
            step_fused(
                w_gf, g_gf, o_gfm, o_gfv, t, lr, b1, b2, eps, 0.0, grad_scale,
            );
        });

        // Thread 1: first layer group
        let h1 = s.spawn(move || {
            for l in 0..wl_a.len() {
                update_layer(
                    cfg,
                    &mut wl_a[l],
                    &gl_a[l],
                    &mut ol_a[l],
                    t,
                    matrix_lr,
                    b1,
                    b2,
                    eps,
                    wd,
                    lr,
                    grad_scale,
                );
            }
        });

        // Thread 2: second group
        let h2 = s.spawn(move || {
            for l in 0..wl_b.len() {
                update_layer(
                    cfg,
                    &mut wl_b[l],
                    &gl_b[l],
                    &mut ol_b[l],
                    t,
                    matrix_lr,
                    b1,
                    b2,
                    eps,
                    wd,
                    lr,
                    grad_scale,
                );
            }
        });

        // Thread 3: third layer group
        let h3 = s.spawn(move || {
            for l in 0..wl_c.len() {
                update_layer(
                    cfg,
                    &mut wl_c[l],
                    &gl_c[l],
                    &mut ol_c[l],
                    t,
                    matrix_lr,
                    b1,
                    b2,
                    eps,
                    wd,
                    lr,
                    grad_scale,
                );
            }
        });

        // Thread 4: fourth layer group
        let h4 = s.spawn(move || {
            for l in 0..wl_d.len() {
                update_layer(
                    cfg,
                    &mut wl_d[l],
                    &gl_d[l],
                    &mut ol_d[l],
                    t,
                    matrix_lr,
                    b1,
                    b2,
                    eps,
                    wd,
                    lr,
                    grad_scale,
                );
            }
        });

        // Main thread: fifth group (remainder)
        for l in 0..wl_e.len() {
            update_layer(
                cfg,
                &mut wl_e[l],
                &gl_e[l],
                &mut ol_e[l],
                t,
                matrix_lr,
                b1,
                b2,
                eps,
                wd,
                lr,
                grad_scale,
            );
        }

        h0.join().expect("Adam thread 0 panicked");
        h1.join().expect("Adam thread 1 panicked");
        h2.join().expect("Adam thread 2 panicked");
        h3.join().expect("Adam thread 3 panicked");
        h4.join().expect("Adam thread 4 panicked");
    });
}

/// Split optimizer path: Muon for layer matrices, Adam for embeddings and RMSNorm gammas.
pub fn update_weights_muon(
    cfg: &ModelConfig,
    weights: &mut ModelWeights,
    grads: &ModelGrads,
    opt: &mut ModelOptState,
    t: u32,
    lr: f32,
    tc: &TrainConfig,
    muon_cfg: &MuonConfig,
    _metal_adam: &MetalAdam,
    metal_muon: &MetalMuon,
    grad_scale: f32,
) {
    use crate::cpu::adam::step_fused;

    let wd = tc.weight_decay;
    let embed_lr = lr * tc.embed_lr_scale;
    let lr_mul = if tc.max_lr > 0.0 { lr / tc.max_lr } else { 1.0 };
    let matrix_lr = muon_cfg.muon_lr * lr_mul;
    let (b1, b2, eps) = (tc.beta1, tc.beta2, tc.eps);

    step_fused(
        &mut weights.embed,
        &grads.dembed,
        &mut opt.embed_m,
        &mut opt.embed_v,
        t,
        embed_lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    step_fused(
        &mut weights.gamma_final,
        &grads.dgamma_final,
        &mut opt.gamma_final_m,
        &mut opt.gamma_final_v,
        t,
        lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );

    for l in 0..cfg.nlayers {
        update_layer_muon(
            cfg,
            &mut weights.layers[l],
            &grads.layers[l],
            &mut opt.layers[l],
            t,
            lr,
            b1,
            b2,
            eps,
            wd,
            matrix_lr,
            muon_cfg,
            metal_muon,
            grad_scale,
        );
    }
}

/// Update all weight tensors in a single layer.
#[inline]
fn update_layer(
    cfg: &ModelConfig,
    w: &mut LayerWeights,
    g: &LayerGrads,
    o: &mut LayerOptState,
    t: u32,
    matrix_lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
    lr: f32,
    grad_scale: f32,
) {
    use crate::cpu::adam::step_fused;
    step_fused(
        &mut w.wq,
        &g.dwq,
        &mut o.m_wq,
        &mut o.v_wq,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.wk,
        &g.dwk,
        &mut o.m_wk,
        &mut o.v_wk,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.wv,
        &g.dwv,
        &mut o.m_wv,
        &mut o.v_wv,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.wo,
        &g.dwo,
        &mut o.m_wo,
        &mut o.v_wo,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.w1,
        &g.dw1,
        &mut o.m_w1,
        &mut o.v_w1,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.w3,
        &g.dw3,
        &mut o.m_w3,
        &mut o.v_w3,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.w2,
        &g.dw2,
        &mut o.m_w2,
        &mut o.v_w2,
        t,
        matrix_lr,
        b1,
        b2,
        eps,
        wd,
        grad_scale,
    );
    step_fused(
        &mut w.gamma1,
        &g.dgamma1,
        &mut o.m_gamma1,
        &mut o.v_gamma1,
        t,
        lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    step_fused(
        &mut w.gamma2,
        &g.dgamma2,
        &mut o.m_gamma2,
        &mut o.v_gamma2,
        t,
        lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    w.refresh_transposes(cfg);
    w.generation += 1;
}

#[allow(clippy::too_many_arguments)]
fn update_layer_muon(
    cfg: &ModelConfig,
    w: &mut LayerWeights,
    g: &LayerGrads,
    o: &mut LayerOptState,
    t: u32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
    matrix_lr: f32,
    muon_cfg: &MuonConfig,
    metal_muon: &MetalMuon,
    grad_scale: f32,
) {
    use crate::cpu::adam::step_fused;

    muon_step_matrix(
        &mut w.wq,
        &g.dwq,
        &mut o.m_wq,
        &mut o.v_wq,
        cfg.dim,
        cfg.q_dim,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );
    muon_step_matrix(
        &mut w.wk,
        &g.dwk,
        &mut o.m_wk,
        &mut o.v_wk,
        cfg.dim,
        cfg.kv_dim,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );
    muon_step_matrix(
        &mut w.wv,
        &g.dwv,
        &mut o.m_wv,
        &mut o.v_wv,
        cfg.dim,
        cfg.kv_dim,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );
    muon_step_matrix(
        &mut w.wo,
        &g.dwo,
        &mut o.m_wo,
        &mut o.v_wo,
        cfg.q_dim,
        cfg.dim,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );
    muon_step_matrix(
        &mut w.w1,
        &g.dw1,
        &mut o.m_w1,
        &mut o.v_w1,
        cfg.dim,
        cfg.hidden,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );
    muon_step_matrix(
        &mut w.w3,
        &g.dw3,
        &mut o.m_w3,
        &mut o.v_w3,
        cfg.dim,
        cfg.hidden,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );
    muon_step_matrix(
        &mut w.w2,
        &g.dw2,
        &mut o.m_w2,
        &mut o.v_w2,
        cfg.dim,
        cfg.hidden,
        matrix_lr,
        wd,
        muon_cfg,
        metal_muon,
        grad_scale,
    );

    step_fused(
        &mut w.gamma1,
        &g.dgamma1,
        &mut o.m_gamma1,
        &mut o.v_gamma1,
        t,
        lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    step_fused(
        &mut w.gamma2,
        &g.dgamma2,
        &mut o.m_gamma2,
        &mut o.v_gamma2,
        t,
        lr,
        b1,
        b2,
        eps,
        0.0,
        grad_scale,
    );
    w.refresh_transposes(cfg);
    w.generation += 1;
}

#[allow(clippy::too_many_arguments)]
fn muon_step_matrix(
    param: &mut [f32],
    grad: &[f32],
    momentum_buf: &mut [f32],
    scratch: &mut [f32],
    rows: usize,
    cols: usize,
    lr: f32,
    wd: f32,
    muon_cfg: &MuonConfig,
    metal_muon: &MetalMuon,
    grad_scale: f32,
) {
    debug_assert_eq!(param.len(), grad.len());
    debug_assert_eq!(param.len(), momentum_buf.len());
    debug_assert_eq!(param.len(), scratch.len());

    for i in 0..grad.len() {
        let g = grad[i] * grad_scale;
        momentum_buf[i] = muon_cfg.muon_momentum * momentum_buf[i] + g;
        scratch[i] = g + muon_cfg.muon_momentum * momentum_buf[i];
    }
    metal_muon.orthogonalize_inplace(rows, cols, scratch, muon_cfg.newton_schulz_steps);
    let scale = ((rows as f32 / cols as f32).max(1.0)).sqrt();
    for i in 0..param.len() {
        let decay = wd * param[i];
        param[i] -= lr * (scratch[i] * scale + decay);
    }
}

fn layer_grad_sq_sum(lg: &LayerGrads) -> f32 {
    let mut sum = 0.0f32;
    for g in [
        &lg.dwq,
        &lg.dwk,
        &lg.dwv,
        &lg.dwo,
        &lg.dw1,
        &lg.dw3,
        &lg.dw2,
        &lg.dgamma1,
        &lg.dgamma2,
    ] {
        sum += vdsp::svesq(g);
    }
    sum
}

fn layer_slice_grad_sq_sum(layers: &[LayerGrads]) -> f32 {
    let mut sum = 0.0f32;
    for lg in layers {
        sum += layer_grad_sq_sum(lg);
    }
    sum
}

/// Global gradient L2 norm (uses vDSP_svesq — single call per tensor, no scratch).
/// Split across 6 shards: embed+gamma and 5 balanced layer groups.
pub fn grad_norm(grads: &ModelGrads) -> f32 {
    let nl = grads.layers.len();
    let g1 = nl / 5;
    let g2 = (nl - g1) / 4;
    let g3 = (nl - g1 - g2) / 3;
    let g4 = (nl - g1 - g2 - g3) / 2;

    let (gl_a, gl_rest) = grads.layers.split_at(g1);
    let (gl_b, gl_rest2) = gl_rest.split_at(g2);
    let (gl_c, gl_rest3) = gl_rest2.split_at(g3);
    let (gl_d, gl_e) = gl_rest3.split_at(g4);

    let (sum0, sum1, sum2, sum3, sum4, sum5) = std::thread::scope(|s| {
        let h0 = s.spawn(|| vdsp::svesq(&grads.dembed) + vdsp::svesq(&grads.dgamma_final));
        let h1 = s.spawn(|| layer_slice_grad_sq_sum(gl_a));
        let h2 = s.spawn(|| layer_slice_grad_sq_sum(gl_b));
        let h3 = s.spawn(|| layer_slice_grad_sq_sum(gl_c));
        let h4 = s.spawn(|| layer_slice_grad_sq_sum(gl_d));
        let sum5 = layer_slice_grad_sq_sum(gl_e);
        (
            h0.join().expect("norm thread 0 panicked"),
            h1.join().expect("norm thread 1 panicked"),
            h2.join().expect("norm thread 2 panicked"),
            h3.join().expect("norm thread 3 panicked"),
            h4.join().expect("norm thread 4 panicked"),
            sum5,
        )
    });

    (sum0 + sum1 + sum2 + sum3 + sum4 + sum5).sqrt()
}

/// Clip all gradients by global L2 norm (in-place cblas_sscal, no scratch allocs).
pub fn clip_grads(grads: &mut ModelGrads, max_norm: f32) {
    let norm = grad_norm(grads);
    if norm > max_norm {
        let scale = max_norm / norm;
        vdsp::sscal(&mut grads.dembed, scale);
        vdsp::sscal(&mut grads.dgamma_final, scale);
        for lg in &mut grads.layers {
            vdsp::sscal(&mut lg.dwq, scale);
            vdsp::sscal(&mut lg.dwk, scale);
            vdsp::sscal(&mut lg.dwv, scale);
            vdsp::sscal(&mut lg.dwo, scale);
            vdsp::sscal(&mut lg.dw1, scale);
            vdsp::sscal(&mut lg.dw3, scale);
            vdsp::sscal(&mut lg.dw2, scale);
            vdsp::sscal(&mut lg.dgamma1, scale);
            vdsp::sscal(&mut lg.dgamma2, scale);
        }
    }
}

/// Simple LCG pseudo-random (same as layer.rs).
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

fn blend_slice_towards(dst: &mut [f32], src: &[f32], amount: f32) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += (s - *d) * amount;
    }
}

/// Run one full training step with gradient accumulation.
/// Returns average loss across the accumulation microbatches.
pub fn train_step(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &mut ModelWeights,
    grads: &mut ModelGrads,
    opt: &mut ModelOptState,
    data: &[u16], // all training tokens
    step: u32,
    tc: &TrainConfig,
    metal_adam: &MetalAdam,
    fwd_ws: &mut ModelForwardWorkspace,
    bwd_ws: &mut ModelBackwardWorkspace,
) -> f32 {
    let seq = cfg.seq;
    grads.zero_out();

    let mut total_loss = 0.0f32;
    let max_pos = data.len() - seq - 1;

    for _micro in 0..tc.accum_steps {
        // Random position (simple LCG)
        let pos = ((step as u64 * 7919 + _micro as u64 * 104729) % max_pos as u64) as usize;
        let input_tokens: Vec<u32> = data[pos..pos + seq].iter().map(|&t| t as u32).collect();
        let target_tokens: Vec<u32> = data[pos + 1..pos + seq + 1]
            .iter()
            .map(|&t| t as u32)
            .collect();

        let loss = forward_ws(
            cfg,
            kernels,
            weights,
            &input_tokens,
            &target_tokens,
            tc.softcap,
            fwd_ws,
        );
        total_loss += loss;

        let bwd_opts = if tc.accum_steps > 1 {
            BackwardOptions::default().with_ffn_weight_cache()
        } else {
            BackwardOptions::default()
        };
        backward_ws_with_options(
            cfg,
            kernels,
            weights,
            fwd_ws,
            &input_tokens,
            tc.softcap,
            tc.loss_scale,
            grads,
            bwd_ws,
            bwd_opts,
        );
    }

    // Compute grad norm for clip decision (single read pass over ~168MB)
    let gsc = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);
    let raw_norm = grad_norm(grads);
    let scaled_norm = raw_norm * gsc;
    let combined_scale = if scaled_norm > tc.grad_clip {
        tc.grad_clip / raw_norm
    } else {
        gsc
    };

    // LR schedule
    let lr = learning_rate(step, tc);

    // Weight update with fused grad scaling (GPU applies grad * combined_scale inline)
    // Eliminates separate CPU sscal pass over ~168MB
    update_weights(
        cfg,
        weights,
        grads,
        opt,
        step + 1,
        lr,
        tc,
        metal_adam,
        combined_scale,
    );

    total_loss / tc.accum_steps as f32
}

/// Forward-only pass returning per-token losses (for val_bpb computation).
pub fn forward_losses(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    tokens: &[u32],
    targets: &[u32],
    softcap: f32,
) -> Vec<f32> {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // Embedding
    let mut x_row = vec![0.0f32; seq * dim];
    embedding::forward(&weights.embed, dim, tokens, &mut x_row);
    let mut x = vec![0.0f32; dim * seq];
    vdsp::mtrans(&x_row, dim, &mut x, seq, seq, dim);

    // Layers
    for l in 0..cfg.nlayers {
        let (x_next, _) = layer::forward(cfg, kernels, &weights.layers[l], &x);
        x = x_next;
    }

    // Final RMSNorm — channel-first, no transpose
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv = vec![0.0f32; seq];
    rmsnorm::forward_channel_first(
        &x,
        &weights.gamma_final,
        &mut x_final,
        &mut rms_inv,
        dim,
        seq,
    );

    // Logits
    let mut x_final_row = vec![0.0f32; seq * dim];
    vdsp::mtrans(&x_final, seq, &mut x_final_row, dim, dim, seq);
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);

    // Softcap
    if softcap > 0.0 {
        vdsp::sscal(&mut logits, 1.0 / softcap);
        vdsp::tanhf_inplace(&mut logits);
        vdsp::sscal(&mut logits, softcap);
    }

    // Per-token cross-entropy
    let mut losses = vec![0.0f32; seq];
    for s in 0..seq {
        let tok_logits = &logits[s * vocab..(s + 1) * vocab];
        let (loss, _) = cross_entropy::forward(tok_logits, targets[s] as usize);
        losses[s] = loss;
    }
    losses
}

#[cfg(test)]
mod tests {
    use super::{ModelWeights, TrainConfig, should_collect_swa};
    use crate::model::ModelConfig;

    #[test]
    fn blend_towards_matches_expected_weighted_average() {
        let cfg = ModelConfig::gpt_karpathy();
        let mut dst = ModelWeights::random(&cfg);
        let mut src = dst.clone();

        dst.embed[0] = 2.0;
        dst.gamma_final[0] = 4.0;
        dst.layers[0].wq[0] = 6.0;
        dst.layers[0].wqt[0] = 8.0;
        dst.layers[0].gamma1[0] = 10.0;

        src.embed[0] = 10.0;
        src.gamma_final[0] = 20.0;
        src.layers[0].wq[0] = 30.0;
        src.layers[0].wqt[0] = 40.0;
        src.layers[0].gamma1[0] = 50.0;
        src.layers[0].generation = 7;

        dst.blend_towards(&src, 0.25);

        assert!((dst.embed[0] - 4.0).abs() < 1e-6);
        assert!((dst.gamma_final[0] - 8.0).abs() < 1e-6);
        assert!((dst.layers[0].wq[0] - 12.0).abs() < 1e-6);
        assert!((dst.layers[0].wqt[0] - 16.0).abs() < 1e-6);
        assert!((dst.layers[0].gamma1[0] - 20.0).abs() < 1e-6);
        assert_eq!(dst.layers[0].generation, 7);
    }

    #[test]
    fn swa_collection_waits_for_late_lr_phase_and_interval() {
        let tc = TrainConfig {
            max_lr: 1.0,
            min_lr_frac: 0.1,
            warmup_steps: 0,
            total_steps: 100,
            swa_interval: Some(10),
            ..Default::default()
        };

        assert!(!should_collect_swa(10, &tc));
        assert!(!should_collect_swa(50, &tc));
        assert!(should_collect_swa(70, &tc));
        assert!(!should_collect_swa(71, &tc));
    }

    #[test]
    fn swa_collection_can_use_decay_phase_fraction() {
        let tc = TrainConfig {
            max_lr: 1.0,
            min_lr_frac: 0.1,
            warmup_steps: 10,
            total_steps: 110,
            swa_interval: Some(10),
            swa_start_frac: Some(0.4),
            ..Default::default()
        };

        assert!(!should_collect_swa(60, &tc));
        assert!(!should_collect_swa(70, &tc));
        assert!(should_collect_swa(80, &tc));
        assert!(should_collect_swa(100, &tc));
    }
}
