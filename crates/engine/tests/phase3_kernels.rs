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

    let sp_in = sdpa_fwd::input_spatial_width(&cfg);
    let out_ch = sdpa_fwd::output_channels(&cfg);
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

    // Small random-ish values
    let input_data: Vec<f32> = (0..input_shape.total_elements())
        .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
        .collect();

    let output = compile_and_eval(graph, &input_data, input_shape, output_shape);
    assert_eq!(output.len(), output_shape.total_elements());
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "sdpa output should not be all zeros");
    println!("sdpaFwd: compiled ✓, ran ✓, out_ch={out_ch}, output sum={sum:.2}");
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

    let sdpa_sp = sdpa_fwd::input_spatial_width(&cfg);
    let sdpa_out_ch = sdpa_fwd::output_channels(&cfg);
    let sdpa_in = TensorData::new(Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: sdpa_sp,
    });
    let sdpa_out = TensorData::new(Shape {
        batch: 1,
        channels: sdpa_out_ch,
        height: 1,
        width: cfg.seq,
    });

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
        sdpa_exe.run(&[&sdpa_in], &[&sdpa_out]).unwrap();
        ffn_exe.run(&[&ffn_in], &[&ffn_out]).unwrap();

        if i == 0 || i == 999 {
            println!("iter {i}: all 3 forward kernels evaluated ✓");
        }
    }
}
