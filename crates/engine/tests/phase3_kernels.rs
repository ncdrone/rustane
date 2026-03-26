//! Phase 3 integration tests: compile and run forward kernels on ANE hardware.
//!
//! These tests verify that kernel graphs compile on the ANE and produce outputs
//! with the correct shapes. They use random-ish data — correctness validation
//! against reference implementations comes later.

use ane_bridge::ane::{Graph, Shape, TensorData};
use engine::kernels::{dyn_matmul, ffn_fused, sdpa_bwd, sdpa_fwd};
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;

fn compile_and_eval(
    graph: Graph,
    input_data: &[f32],
    input_shape: Shape,
    output_shape: Shape,
) -> Vec<f32> {
    let exe = graph
        .compile(NSQualityOfService::UserInteractive)
        .expect("ANE compilation failed");

    let input_td = TensorData::with_f32(input_data, input_shape);
    let output_td = TensorData::new(output_shape);

    exe.run(&[&input_td], &[&output_td])
        .expect("ANE eval failed");

    let locked = output_td.as_f32_slice();
    locked.to_vec()
}

fn compile_and_eval_multi(
    graph: Graph,
    inputs: &[(&[f32], Shape)],
    output_shapes: &[Shape],
) -> Vec<Vec<f32>> {
    let exe = graph
        .compile(NSQualityOfService::UserInteractive)
        .expect("ANE compilation failed");

    let input_tds: Vec<TensorData> = inputs
        .iter()
        .map(|(data, shape)| TensorData::with_f32(data, *shape))
        .collect();
    let output_tds: Vec<TensorData> = output_shapes
        .iter()
        .map(|shape| TensorData::new(*shape))
        .collect();
    let input_refs: Vec<&TensorData> = input_tds.iter().collect();
    let output_refs: Vec<&TensorData> = output_tds.iter().collect();

    exe.run_cached_direct(&input_refs, &output_refs)
        .expect("ANE eval failed");

    output_tds
        .iter()
        .map(|td| td.as_f32_slice().to_vec())
        .collect()
}

#[test]
fn wo_fwd_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    // woFwd: attn_out[Q_DIM,SEQ] @ Wo → [DIM,SEQ]
    // IC=Q_DIM=768, OC=DIM=768
    let graph = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq);

    let sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    let input_shape = Shape {
        batch: 1,
        channels: cfg.q_dim,
        height: 1,
        width: sp,
    };
    let output_shape = Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.seq,
    };

    // Fill with small values to avoid fp16 overflow
    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);

    assert_eq!(output.len(), output_shape.total_elements());
    // Verify output is not all zeros (actual computation happened)
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "output should not be all zeros");
    println!("woFwd: compiled ✓, ran ✓, output sum={sum:.2}");
}

#[test]
fn ffn_bwd_w2t_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    // ffnBwdW2t: dffn @ W2 → dsilu (IC=DIM=768, OC=HIDDEN=2048)
    let graph = dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq);

    let sp = dyn_matmul::spatial_width(cfg.seq, cfg.hidden);
    let input_shape = Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: sp,
    };
    let output_shape = Shape {
        batch: 1,
        channels: cfg.hidden,
        height: 1,
        width: cfg.seq,
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);
    assert_eq!(output.len(), output_shape.total_elements());
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0);
    println!("ffnBwdW2t: compiled ✓, ran ✓, output sum={sum:.2}");
}

#[test]
fn dual_dyn_matmul_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    // kvBwd test: IC=KV_DIM=768, OC=DIM=768
    let graph = dyn_matmul::build_dual(cfg.kv_dim, cfg.dim, cfg.seq);

    let sp = dyn_matmul::dual_spatial_width(cfg.seq, cfg.dim);
    let input_shape = Shape {
        batch: 1,
        channels: cfg.kv_dim,
        height: 1,
        width: sp,
    };
    let output_shape = Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.seq,
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);
    assert_eq!(output.len(), output_shape.total_elements());
    println!("DualDynMatmul: compiled ✓, ran ✓");
}

#[test]
fn sdpa_fwd_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    let graph = sdpa_fwd::build(&cfg);

    let xnorm_shape = sdpa_fwd::xnorm_shape(&cfg);
    let wq_shape = sdpa_fwd::wq_shape(&cfg);
    let wk_shape = sdpa_fwd::wk_shape(&cfg);
    let wv_shape = sdpa_fwd::wv_shape(&cfg);
    let attn_shape = sdpa_fwd::attn_out_shape(&cfg);
    let q_shape = sdpa_fwd::q_rope_shape(&cfg);
    let k_shape = sdpa_fwd::k_rope_shape(&cfg);
    let v_shape = sdpa_fwd::v_shape(&cfg);

    let xnorm_data: Vec<f32> = (0..xnorm_shape.total_elements())
        .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
        .collect();
    let wq_data: Vec<f32> = (0..wq_shape.total_elements())
        .map(|i| ((i % 128) as f32 - 64.0) * 0.0005)
        .collect();
    let wk_data: Vec<f32> = (0..wk_shape.total_elements())
        .map(|i| ((i % 127) as f32 - 63.0) * 0.0005)
        .collect();
    let wv_data: Vec<f32> = (0..wv_shape.total_elements())
        .map(|i| ((i % 129) as f32 - 64.0) * 0.0005)
        .collect();

    let outputs = compile_and_eval_multi(
        graph,
        &[
            (&xnorm_data, xnorm_shape),
            (&wq_data, wq_shape),
            (&wk_data, wk_shape),
            (&wv_data, wv_shape),
        ],
        &[attn_shape, q_shape, k_shape, v_shape],
    );
    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0].len(), attn_shape.total_elements());
    assert_eq!(outputs[1].len(), q_shape.total_elements());
    assert_eq!(outputs[2].len(), k_shape.total_elements());
    assert_eq!(outputs[3].len(), v_shape.total_elements());

    let attn_sum: f32 = outputs[0].iter().map(|x| x.abs()).sum();
    assert!(attn_sum > 0.0, "attn_out should not be all zeros");

    let qk_max_diff = outputs[1]
        .iter()
        .zip(outputs[2].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let qv_max_diff = outputs[1]
        .iter()
        .zip(outputs[3].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        qk_max_diff > 0.0 || qv_max_diff > 0.0,
        "sdpa outputs appear cross-wired or identical"
    );

    println!(
        "sdpaFwd: compiled ✓, ran ✓, attn_out sum={attn_sum:.2}, qk_max_diff={qk_max_diff:.6}, qv_max_diff={qv_max_diff:.6}"
    );
}

#[test]
fn ffn_fused_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    let graph = ffn_fused::build(&cfg);

    let sp_in = ffn_fused::input_spatial_width(&cfg);
    let out_ch = ffn_fused::output_channels(&cfg);
    let input_shape = Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: sp_in,
    };
    let output_shape = Shape {
        batch: 1,
        channels: out_ch,
        height: 1,
        width: cfg.seq,
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);
    assert_eq!(output.len(), output_shape.total_elements());
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "ffn output should not be all zeros");
    println!("ffnFused: compiled ✓, ran ✓, out_ch={out_ch}, output sum={sum:.2}");
}

#[test]
fn sdpa_bwd1_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    let graph = sdpa_bwd::build_bwd1(&cfg);

    let in_ch = sdpa_bwd::bwd1_input_channels(&cfg);
    let out_ch = sdpa_bwd::bwd1_output_channels(&cfg);
    let input_shape = Shape {
        batch: 1,
        channels: in_ch,
        height: 1,
        width: cfg.seq,
    };
    let output_shape = Shape {
        batch: 1,
        channels: out_ch,
        height: 1,
        width: cfg.seq,
    };

    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| ((i % 200) as f32 - 100.0) * 0.0001)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);
    assert_eq!(output.len(), output_shape.total_elements());
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "sdpaBwd1 output should not be all zeros");
    println!("sdpaBwd1: compiled ✓, ran ✓, in_ch={in_ch}, out_ch={out_ch}, sum={sum:.2}");
}

#[test]
fn sdpa_bwd2_compiles_and_runs() {
    let cfg = ModelConfig::gpt_karpathy();
    let graph = sdpa_bwd::build_bwd2(&cfg);

    let in_ch = sdpa_bwd::bwd2_input_channels(&cfg);
    let out_ch = sdpa_bwd::bwd2_output_channels(&cfg);
    let input_shape = Shape {
        batch: 1,
        channels: in_ch,
        height: 1,
        width: cfg.seq,
    };
    let output_shape = Shape {
        batch: 1,
        channels: out_ch,
        height: 1,
        width: cfg.seq,
    };

    // Use small positive values (probs should be softmax-like, dp small)
    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| ((i % 100) as f32 + 1.0) * 0.0001)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);
    assert_eq!(output.len(), output_shape.total_elements());
    println!("sdpaBwd2: compiled ✓, ran ✓, in_ch={in_ch}, out_ch={out_ch}");
}

#[test]
fn all_forward_kernels_1000_iters() {
    let cfg = ModelConfig::gpt_karpathy();

    // Compile all 3 forward kernels
    let wo_graph = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq);
    let sdpa_graph = sdpa_fwd::build(&cfg);
    let ffn_graph = ffn_fused::build(&cfg);

    let qos = NSQualityOfService::UserInteractive;
    let wo_exe = wo_graph.compile(qos).expect("woFwd compile");
    let sdpa_exe = sdpa_graph.compile(qos).expect("sdpaFwd compile");
    let ffn_exe = ffn_graph.compile(qos).expect("ffnFused compile");

    // Allocate IOSurfaces
    let wo_sp = dyn_matmul::spatial_width(cfg.seq, cfg.dim);
    let wo_in = TensorData::new(Shape {
        batch: 1,
        channels: cfg.q_dim,
        height: 1,
        width: wo_sp,
    });
    let wo_out = TensorData::new(Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.seq,
    });

    let sdpa_xnorm_in = TensorData::new(sdpa_fwd::xnorm_shape(&cfg));
    let sdpa_wq_in = TensorData::new(sdpa_fwd::wq_shape(&cfg));
    let sdpa_wk_in = TensorData::new(sdpa_fwd::wk_shape(&cfg));
    let sdpa_wv_in = TensorData::new(sdpa_fwd::wv_shape(&cfg));
    let sdpa_attn_out = TensorData::new(sdpa_fwd::attn_out_shape(&cfg));
    let sdpa_q_out = TensorData::new(sdpa_fwd::q_rope_shape(&cfg));
    let sdpa_k_out = TensorData::new(sdpa_fwd::k_rope_shape(&cfg));
    let sdpa_v_out = TensorData::new(sdpa_fwd::v_shape(&cfg));

    let ffn_sp = ffn_fused::input_spatial_width(&cfg);
    let ffn_out_ch = ffn_fused::output_channels(&cfg);
    let ffn_in = TensorData::new(Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: ffn_sp,
    });
    let ffn_out = TensorData::new(Shape {
        batch: 1,
        channels: ffn_out_ch,
        height: 1,
        width: cfg.seq,
    });

    // Run 1000 iterations
    for i in 0..1000 {
        wo_exe.run(&[&wo_in], &[&wo_out]).unwrap();
        sdpa_exe
            .run(
                &[&sdpa_xnorm_in, &sdpa_wq_in, &sdpa_wk_in, &sdpa_wv_in],
                &[&sdpa_attn_out, &sdpa_q_out, &sdpa_k_out, &sdpa_v_out],
            )
            .unwrap();
        ffn_exe.run(&[&ffn_in], &[&ffn_out]).unwrap();

        if i == 0 || i == 999 {
            println!("iter {i}: all 3 forward kernels evaluated ✓");
        }
    }
}
