//! ANE-accelerated MLA projections using Conv1x1 graphs.
//!
//! Replaces CPU sgemv calls in mla_attention.rs with ANE dispatches.
//! Each projection compiles once at model load (shape-sharing: all layers
//! use identical dimensions), then stages weights per-layer via IOSurface memcpy.
//!
//! Pattern from engine/src/kernels/dyn_matmul.rs:88 (build_conv) and
//! engine/src/layer.rs (TensorData staging + run_cached_direct).

use ane_bridge::ane::{Executable, Shape, TensorData};
use moe_kernels::mla::{build_mla_conv1x1, conv1x1_spatial_width, pad16};
use objc2_foundation::NSQualityOfService;

/// Pre-compiled ANE graphs + pre-allocated IOSurface buffers for MLA decode.
///
/// Compiled once at model load. Reused across all 61 layers — weights are
/// staged into the spatial dimension of the input IOSurface each layer.
pub struct AneMlaKernels {
    // Q LoRA: two projections (compress + expand)
    pub q_a_exec: Executable,
    pub q_a_in: TensorData,
    pub q_a_out: TensorData,

    pub q_b_exec: Executable,
    pub q_b_in: TensorData,
    pub q_b_out: TensorData,

    // KV compression: [hidden → kv_lora_rank + rope_dim]
    pub kv_a_exec: Executable,
    pub kv_a_in: TensorData,
    pub kv_a_out: TensorData,

    // O projection: [v_total → hidden]
    pub o_exec: Executable,
    pub o_in: TensorData,
    pub o_out: TensorData,

    // Dimensions (for staging)
    pub hidden: usize,
    pub q_lora_rank: usize,
    pub q_total: usize,
    pub kv_out_dim: usize, // kv_lora_rank + rope_dim
    pub v_total: usize,

    /// Track which layer's weights are currently staged to avoid re-staging.
    pub staged_layer: std::cell::Cell<Option<usize>>,

    /// Per-layer pre-staged input IOSurfaces (weights baked at load time).
    /// UnsafeCell because prestage_all_layers mutates during &self warmup.
    /// Indexed by layer. Each entry has [q_a_in, q_b_in, kv_a_in, o_in].
    pub layer_inputs: std::cell::UnsafeCell<Vec<[TensorData; 4]>>,
}

/// Compile all MLA Conv1x1 kernels for a given model configuration.
///
/// Total compilations: 4 (q_a, q_b, kv_a, o_proj) — well under 119 limit.
/// All layers share the same dimensions, so we compile once and stage weights.
pub fn compile_mla_kernels(
    hidden: usize,
    q_lora_rank: usize,
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
) -> Result<AneMlaKernels, String> {
    let qos = NSQualityOfService::UserInteractive;
    let seq = 1; // single-token decode

    let q_total = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
    let v_total = num_heads * v_head_dim;
    let rope_dim = qk_rope_head_dim; // single-head rope for KV (not broadcast)
    let kv_out_dim = kv_lora_rank + rope_dim;

    // Q LoRA compress: [hidden → q_lora_rank]
    let q_a_graph = build_mla_conv1x1(hidden, q_lora_rank, seq);
    let q_a_exec = q_a_graph.compile(qos).map_err(|e| format!("q_a compile: {e}"))?;
    let q_a_sp = conv1x1_spatial_width(seq, q_lora_rank);
    let q_a_in = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: q_a_sp });
    let q_a_out = TensorData::new(Shape { batch: 1, channels: q_lora_rank, height: 1, width: pad16(seq) });

    // Q LoRA expand: [q_lora_rank → q_total]
    let q_b_graph = build_mla_conv1x1(q_lora_rank, q_total, seq);
    let q_b_exec = q_b_graph.compile(qos).map_err(|e| format!("q_b compile: {e}"))?;
    let q_b_sp = conv1x1_spatial_width(seq, q_total);
    let q_b_in = TensorData::new(Shape { batch: 1, channels: q_lora_rank, height: 1, width: q_b_sp });
    let q_b_out = TensorData::new(Shape { batch: 1, channels: q_total, height: 1, width: pad16(seq) });

    // KV compression: [hidden → kv_lora_rank + rope_dim]
    let kv_a_graph = build_mla_conv1x1(hidden, kv_out_dim, seq);
    let kv_a_exec = kv_a_graph.compile(qos).map_err(|e| format!("kv_a compile: {e}"))?;
    let kv_a_sp = conv1x1_spatial_width(seq, kv_out_dim);
    let kv_a_in = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: kv_a_sp });
    let kv_a_out = TensorData::new(Shape { batch: 1, channels: kv_out_dim, height: 1, width: pad16(seq) });

    // O projection: [v_total → hidden]
    let o_graph = build_mla_conv1x1(v_total, hidden, seq);
    let o_exec = o_graph.compile(qos).map_err(|e| format!("o_proj compile: {e}"))?;
    let o_sp = conv1x1_spatial_width(seq, hidden);
    let o_in = TensorData::new(Shape { batch: 1, channels: v_total, height: 1, width: o_sp });
    let o_out = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: pad16(seq) });

    Ok(AneMlaKernels {
        q_a_exec, q_a_in, q_a_out,
        q_b_exec, q_b_in, q_b_out,
        kv_a_exec, kv_a_in, kv_a_out,
        o_exec, o_in, o_out,
        hidden, q_lora_rank, q_total, kv_out_dim, v_total,
        staged_layer: std::cell::Cell::new(None),
        layer_inputs: std::cell::UnsafeCell::new(Vec::new()),
    })
}

/// Pre-stage weights into an IOSurface buffer using copy_from_f32.
/// Prepares the full f32 buffer (activation zeros + transposed weights),
/// then converts the entire buffer to fp16 in one bulk NEON pass.
pub fn prestage_weights(
    input_buf: &TensorData,
    weights: &[f32],  // [oc, ic] row-major
    ic: usize,
    oc: usize,
) {
    let sp = input_buf.shape().width;
    let padded_seq = moe_kernels::mla::padded_seq(1);
    let total = ic * sp;

    // Build the full buffer in f32 (activation zeros + transposed weights)
    let mut full_buf = vec![0.0f32; total];
    for c in 0..ic {
        let dst = c * sp + padded_seq;
        for j in 0..oc {
            full_buf[dst + j] = weights[j * ic + c];
        }
    }
    // Bulk f32→fp16 conversion via NEON (one pass, not per-lock)
    input_buf.copy_from_f32(&full_buf);
}

/// Fast dispatch: stage activation via direct fp16 writes, weights already pre-staged.
/// Hot path: IC fp16 writes (7168 × 2 bytes = 14 KB for K2), not 46 MB f32 buffer conversion.
pub fn ane_projection_fast(
    exec: &Executable,
    input_buf: &TensorData,
    output_buf: &TensorData,
    activation: &[f32],
    ic: usize,
    oc: usize,
) -> Result<Vec<f32>, String> {
    let sp = input_buf.shape().width;

    // Direct fp16 write: only IC values at stride SP (bypasses full buffer conversion)
    unsafe {
        ane_bridge::stage_activation_fp16(input_buf, activation, ic, sp);
    }

    exec.run_cached_direct(&[input_buf], &[output_buf])
        .map_err(|e| format!("ANE dispatch: {e}"))?;

    let out_sp = output_buf.shape().width;
    let out_buf = output_buf.as_f32_slice();
    let mut result = vec![0.0f32; oc];
    for c in 0..oc {
        result[c] = out_buf[c * out_sp];
    }
    Ok(result)
}

/// Stage activation + weights into an IOSurface input buffer, then dispatch.
/// Slow path (transposes weights inline). For tests and one-off calls.
/// Production decode should use prestage_weights() + ane_projection_fast().
pub fn ane_projection(
    exec: &Executable,
    input_buf: &TensorData,
    output_buf: &TensorData,
    activation: &[f32],  // [ic]
    weights: &[f32],     // [oc, ic] row-major (CPU layout) or [ic * oc] flattened
    ic: usize,
    oc: usize,
) -> Result<Vec<f32>, String> {
    let sp = input_buf.shape().width;
    let padded_seq = moe_kernels::mla::padded_seq(1); // ANE needs ≥ 64 spatial

    // Stage into IOSurface (channel-interleaved layout).
    // Layout: buf[c * sp + s] = value at channel c, spatial position s
    //
    // Activation at spatial[0], zero-padded through spatial[padded_seq-1].
    // Weights at spatial[padded_seq .. padded_seq + oc], transposed.
    {
        let mut buf = input_buf.as_f32_slice_mut();
        // Zero the entire buffer (pad regions + unused positions must be zero)
        for v in buf.iter_mut() { *v = 0.0; }

        // Write activation at spatial position 0: buf[c * sp + 0] = activation[c]
        for c in 0..ic {
            buf[c * sp] = activation[c];
        }
        // Write weights at spatial offset padded_seq.
        // W_original is [OC, IC] row-major: W[j][c] = weights[j * ic + c]
        // We need W^T[IC, OC] in channel-interleaved layout:
        //   buf[c * sp + padded_seq + j] = W[j][c] for all c,j
        //
        // If weights are pre-transposed [IC, OC] row-major, use fast memcpy path.
        // Otherwise fall back to element-wise transpose.
        if weights.len() == ic * oc {
            // Stage using contiguous copies per channel (fast path).
            // Pre-transpose: W^T[c] = [W[0][c], W[1][c], ..., W[oc-1][c]]
            // We do the transpose inline but minimize cache misses by
            // iterating output-major (sequential writes).
            for c in 0..ic {
                let dst_off = c * sp + padded_seq;
                for j in 0..oc {
                    buf[dst_off + j] = weights[j * ic + c];
                }
            }
        }
    }

    // Dispatch on ANE
    exec.run_cached_direct(&[input_buf], &[output_buf])
        .map_err(|e| format!("ANE dispatch: {e}"))?;

    // Read output: [1, OC, 1, padded_seq] — result at spatial position 0
    let out_sp = output_buf.shape().width;
    let out_buf = output_buf.as_f32_slice();
    let mut result = vec![0.0f32; oc];
    for c in 0..oc {
        result[c] = out_buf[c * out_sp]; // position 0 of each output channel
    }
    Ok(result)
}

/// Profile flag: emit per-dispatch timing.
fn ane_profile_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("RUSTANE_ANE_PROFILE").is_ok())
}

/// Transpose a weight matrix from [oc, ic] row-major to [ic, oc] row-major.
/// Suitable for pre-computing at model load time.
pub fn transpose_weights(weights: &[f32], ic: usize, oc: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; ic * oc];
    for c in 0..ic {
        for j in 0..oc {
            transposed[c * oc + j] = weights[j * ic + c];
        }
    }
    transposed
}

/// Fast weight prestaging from pre-transposed [IC, OC] row-major weights.
/// Uses contiguous copy_from_slice per channel (memcpy speed, no cache misses).
pub fn prestage_weights_transposed(
    input_buf: &TensorData,
    weights_t: &[f32],  // [ic, oc] row-major (pre-transposed)
    ic: usize,
    oc: usize,
) {
    let sp = input_buf.shape().width;
    let padded_seq = moe_kernels::mla::padded_seq(1);
    let total = ic * sp;
    let mut full_buf = vec![0.0f32; total];
    // Contiguous copy: each channel's oc weights are sequential in weights_t
    for c in 0..ic {
        let dst = c * sp + padded_seq;
        let src = c * oc;
        full_buf[dst..dst + oc].copy_from_slice(&weights_t[src..src + oc]);
    }
    input_buf.copy_from_f32(&full_buf);
}

/// Pre-stage all 4 MLA projection weights for one layer.
/// Accepts pre-transposed weights for fast staging (contiguous memcpy).
pub fn prestage_mla_layer_transposed(
    kernels: &AneMlaKernels,
    w_qa_t: &[f32],   // [hidden, q_lora_rank] pre-transposed
    w_qb_t: &[f32],   // [q_lora_rank, q_total] pre-transposed
    w_kv_a_t: &[f32], // [hidden, kv_out_dim] pre-transposed
    w_o_t: &[f32],    // [v_total, hidden] pre-transposed
) {
    prestage_weights_transposed(&kernels.q_a_in, w_qa_t, kernels.hidden, kernels.q_lora_rank);
    prestage_weights_transposed(&kernels.q_b_in, w_qb_t, kernels.q_lora_rank, kernels.q_total);
    prestage_weights_transposed(&kernels.kv_a_in, w_kv_a_t, kernels.hidden, kernels.kv_out_dim);
    prestage_weights_transposed(&kernels.o_in, w_o_t, kernels.v_total, kernels.hidden);
}

/// Pre-stage all 4 MLA projection weights for one layer (slow path).
/// Transposes inline. Use prestage_mla_layer_transposed for production.
pub fn prestage_mla_layer(
    kernels: &AneMlaKernels,
    w_qa: &[f32],
    w_qb: &[f32],
    w_kv_a: &[f32],
    w_o: &[f32],
) {
    prestage_weights(&kernels.q_a_in, w_qa, kernels.hidden, kernels.q_lora_rank);
    prestage_weights(&kernels.q_b_in, w_qb, kernels.q_lora_rank, kernels.q_total);
    prestage_weights(&kernels.kv_a_in, w_kv_a, kernels.hidden, kernels.kv_out_dim);
    prestage_weights(&kernels.o_in, w_o, kernels.v_total, kernels.hidden);
}

/// Pre-stage all layers' weights into per-layer IOSurface buffers.
/// Called once at model load. Memory: ~11 GB for 61 layers (K2).
/// After this, decode only writes activation fp16 (14 KB per dispatch).
/// # Safety
/// Must not be called concurrently with decode (single-threaded access).
pub fn prestage_all_layers(
    kernels: &AneMlaKernels,
    get_layer_weights: impl Fn(usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    num_layers: usize,
) {
    let seq = 1;
    let q_a_sp = conv1x1_spatial_width(seq, kernels.q_lora_rank);
    let q_b_sp = conv1x1_spatial_width(seq, kernels.q_total);
    let kv_a_sp = conv1x1_spatial_width(seq, kernels.kv_out_dim);
    let o_sp = conv1x1_spatial_width(seq, kernels.hidden);

    let t = std::time::Instant::now();
    let layer_inputs = unsafe { &mut *kernels.layer_inputs.get() };
    layer_inputs.clear();

    for layer in 0..num_layers {
        let (w_qa, w_qb, w_kv_a, w_o) = get_layer_weights(layer);

        // Allocate per-layer IOSurfaces
        let q_a_in = TensorData::new(Shape { batch: 1, channels: kernels.hidden, height: 1, width: q_a_sp });
        let q_b_in = TensorData::new(Shape { batch: 1, channels: kernels.q_lora_rank, height: 1, width: q_b_sp });
        let kv_a_in = TensorData::new(Shape { batch: 1, channels: kernels.hidden, height: 1, width: kv_a_sp });
        let o_in = TensorData::new(Shape { batch: 1, channels: kernels.v_total, height: 1, width: o_sp });

        // Pre-transpose + stage weights
        let w_qa_t = transpose_weights(&w_qa, kernels.hidden, kernels.q_lora_rank);
        let w_qb_t = transpose_weights(&w_qb, kernels.q_lora_rank, kernels.q_total);
        let w_kv_a_t = transpose_weights(&w_kv_a, kernels.hidden, kernels.kv_out_dim);
        let w_o_t = transpose_weights(&w_o, kernels.v_total, kernels.hidden);

        prestage_weights_transposed(&q_a_in, &w_qa_t, kernels.hidden, kernels.q_lora_rank);
        prestage_weights_transposed(&q_b_in, &w_qb_t, kernels.q_lora_rank, kernels.q_total);
        prestage_weights_transposed(&kv_a_in, &w_kv_a_t, kernels.hidden, kernels.kv_out_dim);
        prestage_weights_transposed(&o_in, &w_o_t, kernels.v_total, kernels.hidden);

        layer_inputs.push([q_a_in, q_b_in, kv_a_in, o_in]);
    }

    let elapsed = t.elapsed().as_secs_f64();
    eprintln!("[ANE] Pre-staged {num_layers} layers in {elapsed:.1}s ({:.0} MB total)",
        num_layers as f64 * 4.0 * q_a_sp as f64 * kernels.hidden as f64 * 4.0 / 1e6);
}

/// Dispatch using per-layer pre-staged IOSurface (0ms staging, only activation write).
pub fn ane_projection_layered(
    exec: &Executable,
    layer_input: &TensorData,  // per-layer, weights pre-staged
    output_buf: &TensorData,
    activation: &[f32],
    ic: usize,
    oc: usize,
) -> Result<Vec<f32>, String> {
    let sp = layer_input.shape().width;
    // Direct fp16 activation write (14 KB, not 46 MB)
    unsafe {
        ane_bridge::stage_activation_fp16(layer_input, activation, ic, sp);
    }
    exec.run_cached_direct(&[layer_input], &[output_buf])
        .map_err(|e| format!("ANE dispatch: {e}"))?;
    let out_sp = output_buf.shape().width;
    let out_buf = output_buf.as_f32_slice();
    let mut result = vec![0.0f32; oc];
    for c in 0..oc {
        result[c] = out_buf[c * out_sp];
    }
    Ok(result)
}

/// Run Q LoRA projection on ANE (fast path — weights pre-staged).
pub fn ane_q_lora_fast(
    kernels: &AneMlaKernels,
    x: &[f32],
    qa_norm: &[f32],
    rms_eps: f32,
) -> Result<Vec<f32>, String> {
    let q_latent = ane_projection_fast(
        &kernels.q_a_exec, &kernels.q_a_in, &kernels.q_a_out,
        x, kernels.hidden, kernels.q_lora_rank,
    )?;
    let q_latent_normed = crate::rmsnorm::rmsnorm(&q_latent, qa_norm, rms_eps);
    // Prestage q_b activation (q_latent_normed) — q_b weights already staged
    let q = ane_projection_fast(
        &kernels.q_b_exec, &kernels.q_b_in, &kernels.q_b_out,
        &q_latent_normed, kernels.q_lora_rank, kernels.q_total,
    )?;
    Ok(q)
}

/// Run KV compression on ANE (fast path — weights pre-staged).
pub fn ane_kv_compress_fast(
    kernels: &AneMlaKernels,
    x: &[f32],
) -> Result<Vec<f32>, String> {
    ane_projection_fast(
        &kernels.kv_a_exec, &kernels.kv_a_in, &kernels.kv_a_out,
        x, kernels.hidden, kernels.kv_out_dim,
    )
}

/// Run O projection on ANE (fast path — weights pre-staged).
pub fn ane_o_proj_fast(
    kernels: &AneMlaKernels,
    v_concat: &[f32],
) -> Result<Vec<f32>, String> {
    ane_projection_fast(
        &kernels.o_exec, &kernels.o_in, &kernels.o_out,
        v_concat, kernels.v_total, kernels.hidden,
    )
}

/// Run Q LoRA projection on ANE: x → q_a → layernorm → q_b → q[q_total]
/// Slow path (stages weights per call). Use ane_q_lora_fast for decode.
pub fn ane_q_lora(
    kernels: &AneMlaKernels,
    x: &[f32],
    w_qa: &[f32],
    qa_norm: &[f32],
    w_qb: &[f32],
    rms_eps: f32,
) -> Result<Vec<f32>, String> {
    let q_latent = ane_projection(
        &kernels.q_a_exec, &kernels.q_a_in, &kernels.q_a_out,
        x, w_qa, kernels.hidden, kernels.q_lora_rank,
    )?;
    let q_latent_normed = crate::rmsnorm::rmsnorm(&q_latent, qa_norm, rms_eps);
    let q = ane_projection(
        &kernels.q_b_exec, &kernels.q_b_in, &kernels.q_b_out,
        &q_latent_normed, w_qb, kernels.q_lora_rank, kernels.q_total,
    )?;
    Ok(q)
}

/// Run KV compression on ANE: x → kv_out[kv_lora_rank + rope_dim]
pub fn ane_kv_compress(
    kernels: &AneMlaKernels,
    x: &[f32],
    w_kv_a: &[f32],   // [kv_out_dim, hidden] row-major
) -> Result<Vec<f32>, String> {
    ane_projection(
        &kernels.kv_a_exec, &kernels.kv_a_in, &kernels.kv_a_out,
        x, w_kv_a, kernels.hidden, kernels.kv_out_dim,
    )
}

/// Run O projection on ANE: v_concat[v_total] → output[hidden]
pub fn ane_o_proj(
    kernels: &AneMlaKernels,
    v_concat: &[f32],
    w_o: &[f32],      // [hidden, v_total] row-major
) -> Result<Vec<f32>, String> {
    ane_projection(
        &kernels.o_exec, &kernels.o_in, &kernels.o_out,
        v_concat, w_o, kernels.v_total, kernels.hidden,
    )
}

// ---------------------------------------------------------------------------
// Shared expert FFN on ANE (E06-D)
//
// Shared expert: gate[hidden→inter] + up[hidden→inter] + SiLU fusion + down[inter→hidden]
// Each projection fits in 32MB SRAM individually:
//   7168 × 2048 × 2 bytes = 28.7 MB (under 32 MB cliff)
// SiLU stays on CPU (element-wise, no ANE advantage for dynamic weights).
// ---------------------------------------------------------------------------

/// Pre-compiled ANE graphs for shared expert FFN projections.
pub struct AneSharedFfnKernels {
    pub gate_exec: Executable,
    pub gate_in: TensorData,
    pub gate_out: TensorData,

    pub up_exec: Executable,
    pub up_in: TensorData,
    pub up_out: TensorData,

    pub down_exec: Executable,
    pub down_in: TensorData,
    pub down_out: TensorData,

    pub hidden: usize,
    pub inter: usize,
}

/// Compile shared expert FFN Conv1x1 kernels.
/// Total compilations: 3 (gate, up, down). Running total with MLA: 7/119.
pub fn compile_shared_ffn_kernels(
    hidden: usize,
    inter: usize,
) -> Result<AneSharedFfnKernels, String> {
    let qos = NSQualityOfService::UserInteractive;
    let seq = 1;

    // gate: [hidden → inter]
    let gate_graph = build_mla_conv1x1(hidden, inter, seq);
    let gate_exec = gate_graph.compile(qos).map_err(|e| format!("gate compile: {e}"))?;
    let gate_sp = conv1x1_spatial_width(seq, inter);
    let gate_in = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: gate_sp });
    let gate_out = TensorData::new(Shape { batch: 1, channels: inter, height: 1, width: seq });

    // up: [hidden → inter] (same shape as gate, but separate exec for clarity)
    let up_graph = build_mla_conv1x1(hidden, inter, seq);
    let up_exec = up_graph.compile(qos).map_err(|e| format!("up compile: {e}"))?;
    let up_in = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: gate_sp });
    let up_out = TensorData::new(Shape { batch: 1, channels: inter, height: 1, width: seq });

    // down: [inter → hidden]
    let down_graph = build_mla_conv1x1(inter, hidden, seq);
    let down_exec = down_graph.compile(qos).map_err(|e| format!("down compile: {e}"))?;
    let down_sp = conv1x1_spatial_width(seq, hidden);
    let down_in = TensorData::new(Shape { batch: 1, channels: inter, height: 1, width: down_sp });
    let down_out = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });

    Ok(AneSharedFfnKernels {
        gate_exec, gate_in, gate_out,
        up_exec, up_in, up_out,
        down_exec, down_in, down_out,
        hidden, inter,
    })
}

/// Run shared expert FFN on ANE: gate + up → SiLU (CPU) → down.
pub fn ane_shared_ffn(
    kernels: &AneSharedFfnKernels,
    x: &[f32],
    w_gate: &[f32],  // [inter, hidden] row-major
    w_up: &[f32],    // [inter, hidden] row-major
    w_down: &[f32],  // [hidden, inter] row-major
) -> Result<Vec<f32>, String> {
    // gate: x[hidden] → gate_out[inter]
    let gate_out = ane_projection(
        &kernels.gate_exec, &kernels.gate_in, &kernels.gate_out,
        x, w_gate, kernels.hidden, kernels.inter,
    )?;

    // up: x[hidden] → up_out[inter]
    let up_out = ane_projection(
        &kernels.up_exec, &kernels.up_in, &kernels.up_out,
        x, w_up, kernels.hidden, kernels.inter,
    )?;

    // SiLU fusion (CPU — element-wise, trivial)
    let mut gated = vec![0.0f32; kernels.inter];
    for i in 0..kernels.inter {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gated[i] = silu * up_out[i];
    }

    // down: gated[inter] → out[hidden]
    let out = ane_projection(
        &kernels.down_exec, &kernels.down_in, &kernels.down_out,
        &gated, w_down, kernels.inter, kernels.hidden,
    )?;

    Ok(out)
}
