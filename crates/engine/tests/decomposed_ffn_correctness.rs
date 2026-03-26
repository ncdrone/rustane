//! Correctness test: verify that the decomposed FFN path produces results
//! matching the fused path within fp16 tolerance.
//!
//! Strategy: run both fused and decomposed FFN on the gpt_karpathy config
//! (which normally uses fused) and compare outputs.

use ane_bridge::ane::{Shape, TensorData};
use engine::cpu::{silu, vdsp};
use engine::kernels::{dyn_matmul, ffn_fused};
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;

/// Helper: stage activations into IOSurface spatial layout.
fn stage_spatial(
    dst: &mut [f32],
    channels: usize,
    sp_width: usize,
    src: &[f32],
    src_width: usize,
    sp_offset: usize,
) {
    for c in 0..channels {
        let d = c * sp_width + sp_offset;
        let s = c * src_width;
        dst[d..d + src_width].copy_from_slice(&src[s..s + src_width]);
    }
}

/// Read contiguous channels from output buffer.
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

/// Deterministic pseudo-random data.
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

#[test]
fn decomposed_ffn_matches_fused() {
    let cfg = ModelConfig::gpt_karpathy();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
    let qos = NSQualityOfService::UserInteractive;

    // Confirm gpt_karpathy doesn't actually need decomposition
    assert!(
        !ffn_fused::needs_decomposition(&cfg),
        "gpt_karpathy should use fused path (sp={})",
        ffn_fused::input_spatial_width(&cfg)
    );

    // Generate test data
    let x2norm = random_vec(dim * seq, 0.1);
    let x2 = random_vec(dim * seq, 0.5);
    let w1 = random_vec(dim * hidden, 0.02);
    let w3 = random_vec(dim * hidden, 0.02);
    let w2 = random_vec(dim * hidden, 0.02);

    // ═══ Path A: Fused FFN on ANE ═══
    let fused_graph = ffn_fused::build(&cfg);
    let fused_exe = fused_graph.compile(qos).expect("fused compile");

    let ffn_sp = ffn_fused::input_spatial_width(&cfg);
    let ffn_out_ch = ffn_fused::output_channels(&cfg);
    let mut fused_in = vec![0.0f32; dim * ffn_sp];
    stage_spatial(&mut fused_in, dim, ffn_sp, &x2norm, seq, 0);
    stage_spatial(&mut fused_in, dim, ffn_sp, &x2, seq, seq);
    stage_spatial(&mut fused_in, dim, ffn_sp, &w1, hidden, 2 * seq);
    stage_spatial(&mut fused_in, dim, ffn_sp, &w3, hidden, 2 * seq + hidden);
    stage_spatial(
        &mut fused_in,
        dim,
        ffn_sp,
        &w2,
        hidden,
        2 * seq + 2 * hidden,
    );

    let in_td = TensorData::with_f32(
        &fused_in,
        Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: ffn_sp,
        },
    );
    let out_td = TensorData::new(Shape {
        batch: 1,
        channels: ffn_out_ch,
        height: 1,
        width: seq,
    });
    fused_exe.run(&[&in_td], &[&out_td]).expect("fused eval");

    let fused_out = out_td.as_f32_slice().to_vec();
    let mut fused_x_next = vec![0.0f32; dim * seq];
    let mut fused_h1 = vec![0.0f32; hidden * seq];
    let mut fused_h3 = vec![0.0f32; hidden * seq];
    let mut fused_gate = vec![0.0f32; hidden * seq];
    read_channels_into(&fused_out, ffn_out_ch, seq, 0, dim, &mut fused_x_next);
    read_channels_into(&fused_out, ffn_out_ch, seq, dim, hidden, &mut fused_h1);
    read_channels_into(
        &fused_out,
        ffn_out_ch,
        seq,
        dim + hidden,
        hidden,
        &mut fused_h3,
    );
    read_channels_into(
        &fused_out,
        ffn_out_ch,
        seq,
        dim + 2 * hidden,
        hidden,
        &mut fused_gate,
    );

    // ═══ Path B: Decomposed FFN (3 dyn_matmul + CPU) ═══
    let w13_graph = dyn_matmul::build(dim, hidden, seq);
    let w2_graph = dyn_matmul::build(hidden, dim, seq);
    let w13_exe = w13_graph.compile(qos).expect("w13 compile");
    let w2_exe = w2_graph.compile(qos).expect("w2 compile");

    let w13_sp = dyn_matmul::spatial_width(seq, hidden);
    let w2_sp = dyn_matmul::spatial_width(seq, dim);

    // W1 fwd: h1 = x2norm @ W1
    let mut w13_in_buf = vec![0.0f32; dim * w13_sp];
    stage_spatial(&mut w13_in_buf, dim, w13_sp, &x2norm, seq, 0);
    stage_spatial(&mut w13_in_buf, dim, w13_sp, &w1, hidden, seq);

    let w13_in_td = TensorData::with_f32(
        &w13_in_buf,
        Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: w13_sp,
        },
    );
    let w13_out_td = TensorData::new(Shape {
        batch: 1,
        channels: hidden,
        height: 1,
        width: seq,
    });
    w13_exe
        .run(&[&w13_in_td], &[&w13_out_td])
        .expect("W1 fwd eval");
    let decomp_h1: Vec<f32> = w13_out_td.as_f32_slice().to_vec();

    // W3 fwd: h3 = x2norm @ W3
    stage_spatial(&mut w13_in_buf, dim, w13_sp, &w3, hidden, seq);
    let w13_in_td2 = TensorData::with_f32(
        &w13_in_buf,
        Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: w13_sp,
        },
    );
    let w13_out_td2 = TensorData::new(Shape {
        batch: 1,
        channels: hidden,
        height: 1,
        width: seq,
    });
    w13_exe
        .run(&[&w13_in_td2], &[&w13_out_td2])
        .expect("W3 fwd eval");
    let decomp_h3: Vec<f32> = w13_out_td2.as_f32_slice().to_vec();

    // CPU SiLU gate
    let mut decomp_gate = vec![0.0f32; hidden * seq];
    silu::silu_gate(&decomp_h1, &decomp_h3, &mut decomp_gate);

    // Transpose W2: [DIM, HIDDEN] → [HIDDEN, DIM]
    let mut w2t = vec![0.0f32; hidden * dim];
    vdsp::mtrans(&w2, hidden, &mut w2t, dim, dim, hidden);

    // W2 fwd: ffn_out = gate @ W2^T
    let mut w2_in_buf = vec![0.0f32; hidden * w2_sp];
    stage_spatial(&mut w2_in_buf, hidden, w2_sp, &decomp_gate, seq, 0);
    stage_spatial(&mut w2_in_buf, hidden, w2_sp, &w2t, dim, seq);

    let w2_in_td = TensorData::with_f32(
        &w2_in_buf,
        Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: w2_sp,
        },
    );
    let w2_out_td = TensorData::new(Shape {
        batch: 1,
        channels: dim,
        height: 1,
        width: seq,
    });
    w2_exe
        .run(&[&w2_in_td], &[&w2_out_td])
        .expect("W2 fwd eval");
    let ffn_out: Vec<f32> = w2_out_td.as_f32_slice().to_vec();

    // Residual: x_next = x2 + alpha * ffn_out
    let mut decomp_x_next = vec![0.0f32; dim * seq];
    vdsp::vsma(&ffn_out, alpha, &x2, &mut decomp_x_next);

    // ═══ Compare h1, h3, gate, x_next ═══
    let n_h = hidden * seq;
    let n_d = dim * seq;

    // h1 and h3 should match closely (both are single matmul, same graph structure)
    let h1_max_err = (0..n_h)
        .map(|i| (fused_h1[i] - decomp_h1[i]).abs())
        .fold(0.0f32, f32::max);
    let h3_max_err = (0..n_h)
        .map(|i| (fused_h3[i] - decomp_h3[i]).abs())
        .fold(0.0f32, f32::max);

    // gate and x_next accumulate more error through SiLU and matmul chain
    let gate_max_err = (0..n_h)
        .map(|i| (fused_gate[i] - decomp_gate[i]).abs())
        .fold(0.0f32, f32::max);
    let xnext_max_err = (0..n_d)
        .map(|i| (fused_x_next[i] - decomp_x_next[i]).abs())
        .fold(0.0f32, f32::max);

    println!("h1 max abs error:    {h1_max_err:.6}");
    println!("h3 max abs error:    {h3_max_err:.6}");
    println!("gate max abs error:  {gate_max_err:.6}");
    println!("x_next max abs error: {xnext_max_err:.6}");

    // fp16 tolerance: ~1e-3 for single matmul, ~5e-3 for chained operations
    assert!(h1_max_err < 0.01, "h1 error too large: {h1_max_err}");
    assert!(h3_max_err < 0.01, "h3 error too large: {h3_max_err}");
    assert!(gate_max_err < 0.05, "gate error too large: {gate_max_err}");
    assert!(
        xnext_max_err < 0.1,
        "x_next error too large: {xnext_max_err}"
    );
}
