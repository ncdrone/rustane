use engine::cpu::vdsp;
use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelWeights,
};
use engine::layer::{self, CompiledKernels, ForwardCache, LayerWeights};
use engine::metal_ffn::MetalFFN;
use engine::model::ModelConfig;
use half::f16;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

fn clone_layer_weights(w: &LayerWeights) -> LayerWeights {
    LayerWeights {
        wq: w.wq.clone(),
        wk: w.wk.clone(),
        wv: w.wv.clone(),
        wo: w.wo.clone(),
        w1: w.w1.clone(),
        w3: w.w3.clone(),
        w2: w.w2.clone(),
        wqt: w.wqt.clone(),
        wkt: w.wkt.clone(),
        wvt: w.wvt.clone(),
        wot: w.wot.clone(),
        w1t: w.w1t.clone(),
        w3t: w.w3t.clone(),
        gamma1: w.gamma1.clone(),
        gamma2: w.gamma2.clone(),
        generation: w.generation,
    }
}

fn clone_model_weights(w: &ModelWeights) -> ModelWeights {
    ModelWeights {
        embed: w.embed.clone(),
        layers: w.layers.iter().map(clone_layer_weights).collect(),
        gamma_final: w.gamma_final.clone(),
    }
}

fn cpu_ffn_forward(
    cfg: &ModelConfig,
    x2norm: &[f32],
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
    x2: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    let mut h1 = vec![0.0; hidden * seq];
    let mut h3 = vec![0.0; hidden * seq];
    let mut gate = vec![0.0; hidden * seq];
    let mut x_next = x2.to_vec();

    vdsp::sgemm_ta(w1, hidden, dim, x2norm, seq, &mut h1);
    vdsp::sgemm_ta(w3, hidden, dim, x2norm, seq, &mut h3);

    for i in 0..gate.len() {
        let x = h1[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        gate[i] = x * sig * h3[i];
    }

    unsafe {
        vdsp::cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            dim as i32,
            seq as i32,
            hidden as i32,
            alpha,
            w2.as_ptr(),
            hidden as i32,
            gate.as_ptr(),
            seq as i32,
            1.0,
            x_next.as_mut_ptr(),
            seq as i32,
        );
    }

    (x_next, h1, h3, gate)
}

fn cpu_ffn_backward_dx(
    cfg: &ModelConfig,
    dh1: &[f32],
    dh3: &[f32],
    w1: &[f32],
    w3: &[f32],
) -> Vec<f32> {
    let mut dx = vec![0.0; cfg.dim * cfg.seq];
    unsafe {
        vdsp::cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cfg.dim as i32,
            cfg.seq as i32,
            cfg.hidden as i32,
            1.0,
            w1.as_ptr(),
            cfg.hidden as i32,
            dh1.as_ptr(),
            cfg.seq as i32,
            0.0,
            dx.as_mut_ptr(),
            cfg.seq as i32,
        );
        vdsp::cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cfg.dim as i32,
            cfg.seq as i32,
            cfg.hidden as i32,
            1.0,
            w3.as_ptr(),
            cfg.hidden as i32,
            dh3.as_ptr(),
            cfg.seq as i32,
            1.0,
            dx.as_mut_ptr(),
            cfg.seq as i32,
        );
    }
    dx
}

fn roundtrip_f16_vec(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

#[test]
#[ignore]
fn metal_ffn_matches_cpu_reference() {
    let cfg = ModelConfig::gpt_karpathy();
    let metal_ffn = match MetalFFN::new(&cfg) {
        Some(ffn) => ffn,
        None => {
            eprintln!("Metal unavailable, skipping");
            return;
        }
    };
    let weights = LayerWeights::random(&cfg);

    let x2norm: Vec<f32> = (0..cfg.dim * cfg.seq)
        .map(|i| ((i * 17 + 3) % 1000) as f32 * 0.001 - 0.5)
        .collect();
    let x2: Vec<f32> = (0..cfg.dim * cfg.seq)
        .map(|i| ((i * 29 + 11) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    let (cpu_x_next, cpu_h1, cpu_h3, cpu_gate) =
        cpu_ffn_forward(&cfg, &x2norm, &weights.w1, &weights.w3, &weights.w2, &x2);

    let mut gpu_h1 = vec![0.0; cfg.hidden * cfg.seq];
    let mut gpu_h3 = vec![0.0; cfg.hidden * cfg.seq];
    let mut gpu_gate = vec![0.0; cfg.hidden * cfg.seq];
    let mut gpu_x_next = vec![0.0; cfg.dim * cfg.seq];
    metal_ffn.forward_into(
        &cfg,
        &x2norm,
        &weights.w1,
        &weights.w3,
        &weights.w2,
        &x2,
        &mut gpu_h1,
        &mut gpu_h3,
        &mut gpu_gate,
        &mut gpu_x_next,
    );

    let h1_diff = max_abs_diff(&cpu_h1, &gpu_h1);
    let h3_diff = max_abs_diff(&cpu_h3, &gpu_h3);
    let gate_diff = max_abs_diff(&cpu_gate, &gpu_gate);
    let x_next_diff = max_abs_diff(&cpu_x_next, &gpu_x_next);

    println!(
        "cpu vs metal: h1={h1_diff:.6} h3={h3_diff:.6} gate={gate_diff:.6} x_next={x_next_diff:.6}"
    );

    assert!(h1_diff < 0.01, "h1 diff too large: {h1_diff}");
    assert!(h3_diff < 0.01, "h3 diff too large: {h3_diff}");
    assert!(gate_diff < 0.01, "gate diff too large: {gate_diff}");
    assert!(x_next_diff < 0.01, "x_next diff too large: {x_next_diff}");
}

#[test]
#[ignore]
fn gpu_ffn_layer_path_matches_ane_ffn() {
    let cfg = ModelConfig::gpt_karpathy();
    let metal_ffn = match MetalFFN::new(&cfg) {
        Some(ffn) => ffn,
        None => {
            eprintln!("Metal unavailable, skipping");
            return;
        }
    };
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x: Vec<f32> = (0..cfg.dim * cfg.seq)
        .map(|i| ((i * 13 + 5) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    let (ane_x_next, ane_cache) = layer::forward(&cfg, &kernels, &weights, &x);
    let mut gpu_cache = ForwardCache::new(&cfg);
    let mut gpu_x_next = vec![0.0; cfg.dim * cfg.seq];
    layer::forward_into_gpu_ffn(
        &cfg,
        &kernels,
        &metal_ffn,
        &weights,
        &x,
        &mut gpu_cache,
        &mut gpu_x_next,
    );

    let x2norm_diff = max_abs_diff(&ane_cache.x2norm, &gpu_cache.x2norm);
    let h1_diff = max_abs_diff(&ane_cache.h1, &gpu_cache.h1);
    let h3_diff = max_abs_diff(&ane_cache.h3, &gpu_cache.h3);
    let gate_diff = max_abs_diff(&ane_cache.gate, &gpu_cache.gate);
    let x_next_diff = max_abs_diff(&ane_x_next, &gpu_x_next);

    println!(
        "ane vs hybrid: x2norm={x2norm_diff:.6} h1={h1_diff:.6} h3={h3_diff:.6} gate={gate_diff:.6} x_next={x_next_diff:.6}"
    );

    assert!(x2norm_diff < 1e-6, "x2norm diff too large: {x2norm_diff}");
    assert!(h1_diff < 0.01, "h1 diff too large: {h1_diff}");
    assert!(h3_diff < 0.01, "h3 diff too large: {h3_diff}");
    assert!(gate_diff < 0.01, "gate diff too large: {gate_diff}");
    assert!(x_next_diff < 0.01, "x_next diff too large: {x_next_diff}");
}

#[test]
#[ignore]
fn metal_ffn_backward_dx_matches_cpu_reference() {
    let cfg = ModelConfig::gpt_karpathy();
    let metal_ffn = match MetalFFN::new(&cfg) {
        Some(ffn) => ffn,
        None => {
            eprintln!("Metal unavailable, skipping");
            return;
        }
    };
    let weights = LayerWeights::random(&cfg);
    let dh1: Vec<f32> = (0..cfg.hidden * cfg.seq)
        .map(|i| ((i * 19 + 7) % 1000) as f32 * 0.001 - 0.5)
        .collect();
    let dh3: Vec<f32> = (0..cfg.hidden * cfg.seq)
        .map(|i| ((i * 23 + 5) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    let cpu_dx = cpu_ffn_backward_dx(&cfg, &dh1, &dh3, &weights.w1, &weights.w3);
    let mut gpu_dx = vec![0.0; cfg.dim * cfg.seq];
    metal_ffn.backward_dx_into(&cfg, &dh1, &dh3, &weights.w1, &weights.w3, &mut gpu_dx);
    let dx_diff = max_abs_diff(&cpu_dx, &gpu_dx);
    println!("cpu vs metal backward_dx: diff={dx_diff:.6}");
    assert!(dx_diff < 0.01, "dx diff too large: {dx_diff}");
}

#[test]
#[ignore]
fn ane_ffn_600m_is_closer_to_fp16_quantized_cpu() {
    let cfg = ModelConfig::target_600m();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let x: Vec<f32> = (0..cfg.dim * cfg.seq)
        .map(|i| ((i * 13 + 5) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    let (ane_x_next, ane_cache) = layer::forward(&cfg, &kernels, &weights, &x);

    let (cpu_x_next, _cpu_h1, _cpu_h3, cpu_gate) = cpu_ffn_forward(
        &cfg,
        &ane_cache.x2norm,
        &weights.w1,
        &weights.w3,
        &weights.w2,
        &ane_cache.x2,
    );

    let x2norm_q = roundtrip_f16_vec(&ane_cache.x2norm);
    let x2_q = roundtrip_f16_vec(&ane_cache.x2);
    let w1_q = roundtrip_f16_vec(&weights.w1);
    let w3_q = roundtrip_f16_vec(&weights.w3);
    let w2_q = roundtrip_f16_vec(&weights.w2);
    let (cpu_q_x_next, _cpu_q_h1, _cpu_q_h3, cpu_q_gate) =
        cpu_ffn_forward(&cfg, &x2norm_q, &w1_q, &w3_q, &w2_q, &x2_q);

    let gate_exact = max_abs_diff(&ane_cache.gate, &cpu_gate);
    let gate_q = max_abs_diff(&ane_cache.gate, &cpu_q_gate);
    let x_next_exact = max_abs_diff(&ane_x_next, &cpu_x_next);
    let x_next_q = max_abs_diff(&ane_x_next, &cpu_q_x_next);

    println!(
        "ane vs cpu exact/quantized: gate={gate_exact:.6}/{gate_q:.6} x_next={x_next_exact:.6}/{x_next_q:.6}"
    );
}

#[test]
#[ignore]
fn gpu_ffn_full_model_forward_matches_ane() {
    let cfg = ModelConfig::gpt_karpathy();
    let metal_ffn = match MetalFFN::new(&cfg) {
        Some(ffn) => ffn,
        None => {
            eprintln!("Metal unavailable, skipping");
            return;
        }
    };
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    let mut ane_ws = ModelForwardWorkspace::new(&cfg);
    let ane_loss = full_model::forward_ws(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut ane_ws,
    );

    let mut gpu_ws = ModelForwardWorkspace::new(&cfg);
    let gpu_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut gpu_ws,
        full_model::ForwardOptions::gpu_ffn(&metal_ffn),
    );

    let logits_diff = max_abs_diff(&ane_ws.logits, &gpu_ws.logits);
    let loss_diff = (ane_loss - gpu_loss).abs();

    println!("full model: loss_diff={loss_diff:.6} logits_diff={logits_diff:.6}");

    assert!(loss_diff < 0.01, "loss diff too large: {loss_diff}");
    assert!(logits_diff < 0.01, "logits diff too large: {logits_diff}");
}

#[test]
#[ignore]
fn gpu_ffn_600m_backward_matches_ane() {
    let cfg = ModelConfig::target_600m();
    let metal_ffn = match MetalFFN::new(&cfg) {
        Some(ffn) => ffn,
        None => {
            eprintln!("Metal unavailable, skipping");
            return;
        }
    };
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let weights_gpu = clone_model_weights(&weights);
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();

    let mut ane_ws = ModelForwardWorkspace::new(&cfg);
    let ane_loss = full_model::forward_ws(
        &cfg,
        &kernels,
        &weights,
        &tokens,
        &targets,
        0.0,
        &mut ane_ws,
    );
    let mut grads_ane = ModelGrads::zeros(&cfg);
    let mut bwd_ws_ane = ModelBackwardWorkspace::new(&cfg);
    full_model::backward_ws(
        &cfg,
        &kernels,
        &weights,
        &ane_ws,
        &tokens,
        0.0,
        256.0,
        &mut grads_ane,
        &mut bwd_ws_ane,
    );

    let mut gpu_ws = ModelForwardWorkspace::new(&cfg);
    let gpu_loss = full_model::forward_ws_with_options(
        &cfg,
        &kernels,
        &weights_gpu,
        &tokens,
        &targets,
        0.0,
        &mut gpu_ws,
        full_model::ForwardOptions::gpu_ffn(&metal_ffn),
    );
    let mut grads_gpu = ModelGrads::zeros(&cfg);
    let mut bwd_ws_gpu = ModelBackwardWorkspace::new(&cfg);
    let bwd_opts = if std::env::var("RUSTANE_USE_GPU_FFN_BWD_DX")
        .ok()
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
    {
        full_model::BackwardOptions::gpu_ffn_dx(&metal_ffn)
    } else {
        full_model::BackwardOptions::default()
    };
    full_model::backward_ws_with_options(
        &cfg,
        &kernels,
        &weights_gpu,
        &gpu_ws,
        &tokens,
        0.0,
        256.0,
        &mut grads_gpu,
        &mut bwd_ws_gpu,
        bwd_opts,
    );

    let loss_diff = (ane_loss - gpu_loss).abs();
    let dembed_diff = max_abs_diff(&grads_ane.dembed, &grads_gpu.dembed);
    let l0_dw1_diff = max_abs_diff(&grads_ane.layers[0].dw1, &grads_gpu.layers[0].dw1);
    let l0_dw2_diff = max_abs_diff(&grads_ane.layers[0].dw2, &grads_gpu.layers[0].dw2);
    let l19_dw2_diff = max_abs_diff(
        &grads_ane.layers[cfg.nlayers - 1].dw2,
        &grads_gpu.layers[cfg.nlayers - 1].dw2,
    );
    let gate0_diff = max_abs_diff(&ane_ws.caches[0].gate, &gpu_ws.caches[0].gate);
    let gate19_diff = max_abs_diff(
        &ane_ws.caches[cfg.nlayers - 1].gate,
        &gpu_ws.caches[cfg.nlayers - 1].gate,
    );
    let l0_dw2_max = max_abs(&grads_ane.layers[0].dw2);
    let l19_dw2_max = max_abs(&grads_ane.layers[cfg.nlayers - 1].dw2);

    println!(
        "600m backward: loss_diff={loss_diff:.6} dembed={dembed_diff:.6} gate0={gate0_diff:.6} gate19={gate19_diff:.6} l0.dw1={l0_dw1_diff:.6} l0.dw2={l0_dw2_diff:.6}/{l0_dw2_max:.6} l19.dw2={l19_dw2_diff:.6}/{l19_dw2_max:.6}"
    );

    assert!(loss_diff < 0.01, "loss diff too large: {loss_diff}");
    assert!(dembed_diff < 0.01, "dembed diff too large: {dembed_diff}");
    assert!(l0_dw1_diff < 0.01, "l0.dw1 diff too large: {l0_dw1_diff}");
    assert!(l0_dw2_diff < 0.01, "l0.dw2 diff too large: {l0_dw2_diff}");
    assert!(
        l19_dw2_diff < 0.01,
        "l19.dw2 diff too large: {l19_dw2_diff}"
    );
}
