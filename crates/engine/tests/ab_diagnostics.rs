//! A/B diagnostic tests to isolate where time is actually spent.
//! Run: cargo test -p engine --test ab_diagnostics --release -- --ignored --nocapture

use engine::cpu::vdsp;
use engine::full_model::{self, ModelBackwardWorkspace, ModelGrads, ModelWeights, TrainConfig};
use engine::layer::{self, BackwardWorkspace, CompiledKernels, LayerGrads, LayerWeights};
use engine::model::ModelConfig;
use std::time::Instant;

#[test]
#[ignore]
fn ab_diagnose_bottlenecks() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let tc = TrainConfig::default();
    let tokens: Vec<u32> = (0..cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let targets: Vec<u32> = (1..=cfg.seq)
        .map(|i| ((i * 31 + 7) % cfg.vocab) as u32)
        .collect();
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    // Warmup (2 full steps)
    for _ in 0..2 {
        grads.zero_out();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        full_model::backward(
            &cfg,
            &kernels,
            &weights,
            &fwd,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut bwd_ws,
        );
    }

    println!("\n=== A/B DIAGNOSTIC: Isolating Bottlenecks ===");
    println!("Model: 6L 768D 512S 8192V\n");

    let dim = cfg.dim;
    let hidden = cfg.hidden;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;
    let n_runs = 20;

    // ──────────────────────────────────────────────
    // Hypothesis 1: Weight mtrans cost in backward
    // ──────────────────────────────────────────────
    println!("── H1: Weight transpose cost ──");

    let mut w1t = vec![0.0f32; hidden * dim];
    let mut w3t = vec![0.0f32; hidden * dim];
    let mut wot = vec![0.0f32; dim * q_dim];
    let mut wqt = vec![0.0f32; q_dim * dim];
    let mut wkt = vec![0.0f32; kv_dim * dim];
    let mut wvt = vec![0.0f32; kv_dim * dim];

    // Warmup transposes
    vdsp::mtrans(&weights.layers[0].w1, hidden, &mut w1t, dim, dim, hidden);

    let mut total_mtrans = 0.0f32;
    for _ in 0..n_runs {
        let t = Instant::now();
        vdsp::mtrans(&weights.layers[0].w1, hidden, &mut w1t, dim, dim, hidden);
        vdsp::mtrans(&weights.layers[0].w3, hidden, &mut w3t, dim, dim, hidden);
        vdsp::mtrans(&weights.layers[0].wo, dim, &mut wot, q_dim, q_dim, dim);
        vdsp::mtrans(&weights.layers[0].wq, q_dim, &mut wqt, dim, dim, q_dim);
        vdsp::mtrans(&weights.layers[0].wk, kv_dim, &mut wkt, dim, dim, kv_dim);
        vdsp::mtrans(&weights.layers[0].wv, kv_dim, &mut wvt, dim, dim, kv_dim);
        total_mtrans += t.elapsed().as_secs_f32() * 1000.0;
    }
    let mtrans_per_layer = total_mtrans / n_runs as f32;
    println!(
        "  6 weight mtrans/layer:  {:.2}ms × 6 layers = {:.1}ms",
        mtrans_per_layer,
        mtrans_per_layer * 6.0
    );

    // ──────────────────────────────────────────────
    // Hypothesis 2: Forward allocation overhead
    // ──────────────────────────────────────────────
    println!("\n── H2: Forward allocation overhead ──");

    let alloc_sizes: Vec<usize> = vec![
        dim * seq,
        dim * seq,
        seq,
        q_dim * seq,
        kv_dim * seq,
        kv_dim * seq,
        q_dim * seq,
        dim * seq,
        dim * seq,
        dim * seq,
        seq,
        hidden * seq,
        hidden * seq,
        hidden * seq,
    ];

    // A: Time allocation of all forward cache buffers for 6 layers
    let t = Instant::now();
    for _ in 0..6 {
        let mut _vecs: Vec<Vec<f32>> = alloc_sizes.iter().map(|&s| vec![0.0f32; s]).collect();
        std::hint::black_box(&mut _vecs);
    }
    let alloc_ms = t.elapsed().as_secs_f32() * 1000.0;
    println!("  Alloc 14 Vecs × 6 layers:  {:.2}ms", alloc_ms);

    // B: Same data but pre-allocated (just zero-fill)
    let mut pre_alloc: Vec<Vec<f32>> = alloc_sizes.iter().map(|&s| vec![0.0f32; s]).collect();
    let t = Instant::now();
    for _ in 0..6 {
        for v in pre_alloc.iter_mut() {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
        std::hint::black_box(&mut pre_alloc);
    }
    let zero_ms = t.elapsed().as_secs_f32() * 1000.0;
    println!("  Zero 14 Vecs × 6 layers:   {:.2}ms", zero_ms);
    println!("  Allocation overhead:        {:.2}ms", alloc_ms - zero_ms);

    // ──────────────────────────────────────────────
    // Hypothesis 3: 1 layer timing breakdown (fwd vs bwd components)
    // ──────────────────────────────────────────────
    println!("\n── H3: Per-layer component timing ──");

    let layer_w = LayerWeights::random(&cfg);
    let mut layer_g = LayerGrads::zeros(&cfg);
    let x: Vec<f32> = (0..dim * seq)
        .map(|i| ((i * 17 + 3) % 1000) as f32 * 0.001 - 0.5)
        .collect();
    let mut bwd_ws = BackwardWorkspace::new(&cfg);

    // Warmup
    let (x_next, cache) = layer::forward(&cfg, &kernels, &layer_w, &x);
    layer_g.zero_out();
    let _ = layer::backward(
        &cfg,
        &kernels,
        &layer_w,
        &cache,
        &x_next,
        &mut layer_g,
        &mut bwd_ws,
    );

    // Single layer timing (n_runs)
    let mut fwd_1 = 0.0f32;
    let mut bwd_1 = 0.0f32;
    for _ in 0..n_runs {
        let t = Instant::now();
        let (x_next, cache) = layer::forward(&cfg, &kernels, &layer_w, &x);
        fwd_1 += t.elapsed().as_secs_f32() * 1000.0;

        layer_g.zero_out();
        let t = Instant::now();
        let _ = layer::backward(
            &cfg,
            &kernels,
            &layer_w,
            &cache,
            &x_next,
            &mut layer_g,
            &mut bwd_ws,
        );
        bwd_1 += t.elapsed().as_secs_f32() * 1000.0;
    }
    fwd_1 /= n_runs as f32;
    bwd_1 /= n_runs as f32;
    println!(
        "  1 layer fwd:  {:.2}ms  bwd: {:.2}ms  total: {:.2}ms",
        fwd_1,
        bwd_1,
        fwd_1 + bwd_1
    );

    // Timed variants — use built-in print() for full breakdown
    let (x_next_t, cache_t, fwd_times) = layer::forward_timed(&cfg, &kernels, &layer_w, &x);
    println!("  Forward breakdown (total: {:.2}ms):", fwd_times.total_ms);
    fwd_times.print();

    // Categorize: staging vs ANE compute vs CPU vs readback
    let fwd_staging = fwd_times.stage_sdpa_ms + fwd_times.stage_wo_ms + fwd_times.stage_ffn_ms;
    let fwd_ane = fwd_times.ane_sdpa_ms + fwd_times.ane_wo_ms + fwd_times.ane_ffn_ms;
    let fwd_read = fwd_times.read_sdpa_ms + fwd_times.read_wo_ms + fwd_times.read_ffn_ms;
    let fwd_cpu = fwd_times.rmsnorm1_ms + fwd_times.residual_rmsnorm2_ms;
    println!(
        "  → Staging:  {:.2}ms ({:.0}%)",
        fwd_staging,
        fwd_staging / fwd_times.total_ms * 100.0
    );
    println!(
        "  → ANE:      {:.2}ms ({:.0}%)",
        fwd_ane,
        fwd_ane / fwd_times.total_ms * 100.0
    );
    println!(
        "  → Readback: {:.2}ms ({:.0}%)",
        fwd_read,
        fwd_read / fwd_times.total_ms * 100.0
    );
    println!(
        "  → CPU:      {:.2}ms ({:.0}%)",
        fwd_cpu,
        fwd_cpu / fwd_times.total_ms * 100.0
    );

    layer_g.zero_out();
    let (_, bwd_times) = layer::backward_timed(
        &cfg,
        &kernels,
        &layer_w,
        &cache_t,
        &x_next_t,
        &mut layer_g,
        &mut bwd_ws,
    );
    println!("  Backward breakdown (total: {:.2}ms):", bwd_times.total_ms);
    bwd_times.print();

    // Categorize backward
    let bwd_staging = bwd_times.stage_run_ffn_bwd_w2t_ms
        + bwd_times.stage_ffn_bwd_w13t_ms
        + bwd_times.stage_run_wot_bwd_ms
        + bwd_times.stage_sdpa_bwd1_ms
        + bwd_times.stage_run_sdpa_bwd2_ms
        + bwd_times.stage_q_bwd_ms
        + bwd_times.stage_run_kv_bwd_ms;
    let bwd_async = bwd_times.async_ffn_bwd_w13t_plus_dw_ms
        + bwd_times.async_sdpa_bwd1_plus_dwo_ms
        + bwd_times.async_q_bwd_plus_dw_ms;
    let bwd_cpu = bwd_times.scale_dy_ms
        + bwd_times.silu_deriv_ms
        + bwd_times.rmsnorm2_bwd_ms
        + bwd_times.rope_bwd_ms
        + bwd_times.rmsnorm1_bwd_ms
        + bwd_times.merge_dx_ms;
    let bwd_read = bwd_times.read_sdpa_bwd1_ms;
    println!(
        "  → Stage+ANE: {:.2}ms ({:.0}%)",
        bwd_staging,
        bwd_staging / bwd_times.total_ms * 100.0
    );
    println!(
        "  → Async ANE+CPU: {:.2}ms ({:.0}%)",
        bwd_async,
        bwd_async / bwd_times.total_ms * 100.0
    );
    println!(
        "  → Pure CPU:  {:.2}ms ({:.0}%)",
        bwd_cpu,
        bwd_cpu / bwd_times.total_ms * 100.0
    );
    println!(
        "  → Readback:  {:.2}ms ({:.0}%)",
        bwd_read,
        bwd_read / bwd_times.total_ms * 100.0
    );

    // ──────────────────────────────────────────────
    // Hypothesis 4: Per-layer scaling (cache pressure, 1 vs 6)
    // ──────────────────────────────────────────────
    println!("\n── H4: Per-layer scaling (cache pressure) ──");

    let mut model_bwd_ws = ModelBackwardWorkspace::new(&cfg);

    // 6 layers sequential (using real model forward/backward)
    grads.zero_out();
    let mut fwd_6 = 0.0f32;
    let mut bwd_6 = 0.0f32;
    for _ in 0..n_runs {
        let t = Instant::now();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        fwd_6 += t.elapsed().as_secs_f32() * 1000.0;

        grads.zero_out();
        let t = Instant::now();
        full_model::backward(
            &cfg,
            &kernels,
            &weights,
            &fwd,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut model_bwd_ws,
        );
        bwd_6 += t.elapsed().as_secs_f32() * 1000.0;
    }
    fwd_6 /= n_runs as f32;
    bwd_6 /= n_runs as f32;

    let fwd_model_overhead = fwd_6 - fwd_1 * 6.0;
    let bwd_model_overhead = bwd_6 - bwd_1 * 6.0;
    println!(
        "  6 layer fwd:  {:.1}ms  ({:.2}ms/layer, {:.1}ms model overhead)",
        fwd_6,
        fwd_6 / 6.0,
        fwd_model_overhead
    );
    println!(
        "  6 layer bwd:  {:.1}ms  ({:.2}ms/layer, {:.1}ms model overhead)",
        bwd_6,
        bwd_6 / 6.0,
        bwd_model_overhead
    );
    println!(
        "  Scale factor fwd: {:.2}×  bwd: {:.2}×",
        (fwd_6 / 6.0) / fwd_1,
        (bwd_6 / 6.0) / bwd_1
    );

    // ──────────────────────────────────────────────
    // Hypothesis 5: Cross-entropy cost breakdown
    // ──────────────────────────────────────────────
    println!("\n── H5: Cross-entropy cost breakdown ──");

    let logits_row = vec![0.1f32; vocab];
    let mut exp_out = vec![0.0f32; vocab];

    // Warmup
    vdsp::expf(&logits_row, &mut exp_out);

    let t = Instant::now();
    for _ in 0..(seq * n_runs) {
        vdsp::expf(&logits_row, &mut exp_out);
    }
    let exp_per_pos = t.elapsed().as_secs_f32() * 1000.0 / n_runs as f32;
    println!(
        "  {} expf calls on [{}]:  {:.2}ms ({:.3}ms/call)",
        seq,
        vocab,
        exp_per_pos,
        exp_per_pos / seq as f32
    );
    println!("  CE does 2× expf:       {:.2}ms (est)", exp_per_pos * 2.0);

    // ──────────────────────────────────────────────
    // Hypothesis 6: BLAS matmul cost (model-level embed/proj)
    // ──────────────────────────────────────────────
    println!("\n── H6: Model-level BLAS matmul cost ──");

    // embed @ x_final: [vocab, dim] × [dim, seq] → [vocab, seq]
    let a = vec![0.1f32; vocab * dim];
    let b = vec![0.1f32; dim * seq];
    let mut c_buf = vec![0.0f32; vocab * seq];

    // Warmup
    vdsp::sgemm_at(&b, seq, dim, &a, vocab, &mut c_buf);

    let t = Instant::now();
    for _ in 0..n_runs {
        for x in c_buf.iter_mut() {
            *x = 0.0;
        }
        vdsp::sgemm_at(&b, seq, dim, &a, vocab, &mut c_buf);
    }
    let proj_fwd_ms = t.elapsed().as_secs_f32() * 1000.0 / n_runs as f32;
    println!(
        "  embed proj fwd [{}×{}]@[{}×{}]: {:.2}ms",
        seq, dim, dim, vocab, proj_fwd_ms
    );

    // grad matmul: [vocab, seq]^T @ [seq, dim] → [vocab, dim]
    let dl = vec![0.1f32; vocab * seq];
    let mut dembed = vec![0.0f32; vocab * dim];

    let t = Instant::now();
    for _ in 0..n_runs {
        for x in dembed.iter_mut() {
            *x = 0.0;
        }
        vdsp::sgemm_ta(&dl, vocab, seq, &b, dim, &mut dembed);
    }
    let proj_bwd_ms = t.elapsed().as_secs_f32() * 1000.0 / n_runs as f32;
    println!(
        "  embed grad [{}×{}]^T@[{}×{}]:   {:.2}ms",
        vocab, seq, seq, dim, proj_bwd_ms
    );

    // ──────────────────────────────────────────────
    // Hypothesis 7: Full step breakdown (fwd + bwd + adam)
    // ──────────────────────────────────────────────
    println!("\n── H7: Full training step timing ──");

    let mut full_step = 0.0f32;
    for _ in 0..n_runs {
        grads.zero_out();
        let t = Instant::now();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        full_model::backward(
            &cfg,
            &kernels,
            &weights,
            &fwd,
            &tokens,
            tc.softcap,
            tc.loss_scale,
            &mut grads,
            &mut model_bwd_ws,
        );
        full_step += t.elapsed().as_secs_f32() * 1000.0;
    }
    full_step /= n_runs as f32;
    let fwd_pct = fwd_6 / full_step * 100.0;
    let bwd_pct = bwd_6 / full_step * 100.0;
    let other_pct = 100.0 - fwd_pct - bwd_pct;
    println!("  Full step (fwd+bwd):   {:.1}ms", full_step);
    println!("  Forward:  {:.1}ms ({:.0}%)", fwd_6, fwd_pct);
    println!("  Backward: {:.1}ms ({:.0}%)", bwd_6, bwd_pct);
    println!(
        "  Gap (alloc/staging):   {:.1}ms ({:.0}%)",
        full_step - fwd_6 - bwd_6,
        other_pct
    );

    // ──────────────────────────────────────────────
    // Summary
    // ──────────────────────────────────────────────
    println!("\n=== SUMMARY: Where to Focus ===");
    println!(
        "  Weight mtrans:     {:.1}ms/step ({:.0}% of {:.0}ms)",
        mtrans_per_layer * 6.0,
        mtrans_per_layer * 6.0 / full_step * 100.0,
        full_step
    );
    println!(
        "  Fwd allocations:   {:.1}ms/step ({:.0}%)",
        alloc_ms,
        alloc_ms / full_step * 100.0
    );
    println!(
        "  CE expf:           {:.1}ms/step ({:.0}%)",
        exp_per_pos * 2.0,
        exp_per_pos * 2.0 / full_step * 100.0
    );
    println!(
        "  BLAS proj fwd+bwd: {:.1}ms ({:.0}%)",
        proj_fwd_ms + proj_bwd_ms,
        (proj_fwd_ms + proj_bwd_ms) / full_step * 100.0
    );
    println!(
        "  Model overhead:    fwd {:.1}ms + bwd {:.1}ms",
        fwd_model_overhead, bwd_model_overhead
    );
    println!(
        "  1 layer:           fwd {:.2}ms + bwd {:.2}ms = {:.2}ms",
        fwd_1,
        bwd_1,
        fwd_1 + bwd_1
    );
    println!(
        "  6 layer scaling:   fwd {:.2}×  bwd {:.2}×",
        (fwd_6 / 6.0) / fwd_1,
        (bwd_6 / 6.0) / bwd_1
    );
}
