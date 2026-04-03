//! Profile individual operations inside ONE layer forward+backward.
//! Run: cargo test -p engine --test profile_single_layer --release -- --ignored --nocapture

use engine::layer::{self, CompiledKernels, LayerGrads, LayerWeights};
use engine::model::ModelConfig;

#[test]
#[ignore]
fn profile_layer_ops() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = LayerWeights::random(&cfg);
    let mut grads = LayerGrads::zeros(&cfg);

    let dim = cfg.dim;
    let seq = cfg.seq;

    // Random input
    let x: Vec<f32> = (0..dim * seq)
        .map(|i| ((i * 17 + 3) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    // Warmup
    let mut bwd_ws = layer::BackwardWorkspace::new(&cfg);
    {
        let (x_next, cache) = layer::forward(&cfg, &kernels, &weights, &x);
        let _ = layer::backward(
            &cfg,
            &kernels,
            &weights,
            &cache,
            &x_next,
            &mut grads,
            &mut bwd_ws,
        );
    }

    println!("\n=== Single Layer Op-Level Profiling (3 runs) ===");
    println!("Model: 768D, 512S, 12H, 2048 hidden\n");

    for run in 0..3 {
        grads.zero_out();

        println!("--- Run {} ---", run + 1);

        // Forward with timing
        println!("  FORWARD:");
        let (x_next, cache, fwd_t) = layer::forward_timed(&cfg, &kernels, &weights, &x);
        fwd_t.print();

        // Backward with timing
        println!("  BACKWARD:");
        let (_dx, bwd_t) = layer::backward_timed(
            &cfg,
            &kernels,
            &weights,
            &cache,
            &x_next,
            &mut grads,
            &mut bwd_ws,
        );
        bwd_t.print();

        println!(
            "  {:<35} {:>6.2}ms\n",
            "FWD+BWD TOTAL",
            fwd_t.total_ms + bwd_t.total_ms
        );
    }

    // Summary: categorize time into ANE vs CPU vs staging
    println!("=== Category Summary (last run averages) ===");
    let (x_next, cache, fwd_t) = layer::forward_timed(&cfg, &kernels, &weights, &x);
    grads.zero_out();
    let (_dx, bwd_t) = layer::backward_timed(
        &cfg,
        &kernels,
        &weights,
        &cache,
        &x_next,
        &mut grads,
        &mut bwd_ws,
    );

    let ane_fwd = fwd_t.ane_sdpa_ms + fwd_t.ane_wo_ms + fwd_t.ane_ffn_ms;
    let stage_fwd = fwd_t.stage_sdpa_ms
        + fwd_t.stage_wo_ms
        + fwd_t.stage_ffn_ms
        + fwd_t.read_sdpa_ms
        + fwd_t.read_wo_ms
        + fwd_t.read_ffn_ms;
    let cpu_fwd = fwd_t.rmsnorm1_ms + fwd_t.residual_rmsnorm2_ms;

    let ane_bwd = bwd_t.stage_run_ffn_bwd_w2t_ms
        + bwd_t.async_ffn_bwd_w13t_plus_dw_ms
        + bwd_t.stage_run_wot_bwd_ms
        + bwd_t.async_sdpa_bwd1_plus_dwo_ms
        + bwd_t.stage_run_sdpa_bwd2_ms
        + bwd_t.async_q_bwd_plus_dw_ms
        + bwd_t.stage_run_kv_bwd_ms;
    let cpu_bwd = bwd_t.scale_dy_ms
        + bwd_t.silu_deriv_ms
        + bwd_t.rmsnorm2_bwd_ms
        + bwd_t.rope_bwd_ms
        + bwd_t.rmsnorm1_bwd_ms
        + bwd_t.merge_dx_ms;
    let stage_bwd = bwd_t.stage_ffn_bwd_w13t_ms
        + bwd_t.stage_sdpa_bwd1_ms
        + bwd_t.stage_q_bwd_ms
        + bwd_t.read_sdpa_bwd1_ms;

    println!(
        "  FORWARD:  ANE {ane_fwd:.1}ms | staging {stage_fwd:.1}ms | CPU {cpu_fwd:.1}ms | total {:.1}ms",
        fwd_t.total_ms
    );
    println!(
        "  BACKWARD: ANE+dW {ane_bwd:.1}ms | staging {stage_bwd:.1}ms | CPU {cpu_bwd:.1}ms | total {:.1}ms",
        bwd_t.total_ms
    );
    println!("  COMBINED: {:.1}ms", fwd_t.total_ms + bwd_t.total_ms);
}
