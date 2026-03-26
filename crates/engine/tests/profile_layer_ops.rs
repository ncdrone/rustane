//! Profile model-level forward and backward breakdown.
//! Run: cargo test -p engine --test profile_layer_ops --release -- --ignored --nocapture

use engine::cpu::{cross_entropy, embedding, rmsnorm, vdsp};
use engine::full_model::{self, ModelBackwardWorkspace, ModelGrads, ModelWeights, TrainConfig};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;
use std::time::Instant;

#[test]
#[ignore]
fn profile_full_step_breakdown() {
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

    // Warmup
    {
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

    println!("\n=== Full Step Profiling (3 runs) ===");
    println!("Model: 6L 768D 512S 8192V\n");

    for run in 0..3 {
        grads.zero_out();
        println!("--- Run {} ---", run + 1);

        let t0 = Instant::now();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        let fwd_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
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
        let bwd_ms = t1.elapsed().as_secs_f32() * 1000.0;

        let t2 = Instant::now();
        // Grad norm only (scale fused into Adam GPU kernel)
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let _scaled_norm = raw_norm * gsc;
        let _combined_scale = if _scaled_norm > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let clip_ms = t2.elapsed().as_secs_f32() * 1000.0;

        println!("  forward:     {:>7.1}ms", fwd_ms);
        println!("  backward:    {:>7.1}ms", bwd_ms);
        println!("  scale+clip:  {:>7.1}ms", clip_ms);
        println!("  total:       {:>7.1}ms\n", fwd_ms + bwd_ms + clip_ms);
    }

    // Detailed forward breakdown
    println!("=== Forward Pass Component Breakdown ===");
    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // Embedding + transpose
    let t = Instant::now();
    let mut x_row = vec![0.0f32; seq * dim];
    embedding::forward(&weights.embed, dim, &tokens, &mut x_row);
    let mut x = vec![0.0f32; dim * seq];
    vdsp::mtrans(&x_row, dim, &mut x, seq, seq, dim);
    println!(
        "  embed + mtrans:   {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // 6 layers
    let t = Instant::now();
    let mut caches = Vec::with_capacity(cfg.nlayers);
    for l in 0..cfg.nlayers {
        let (x_next, cache) = engine::layer::forward(&cfg, &kernels, &weights.layers[l], &x);
        caches.push(cache);
        x = x_next;
    }
    let layer_ms = t.elapsed().as_secs_f32() * 1000.0;
    println!(
        "  6 layers:         {:>6.2}ms ({:.2}ms/layer)",
        layer_ms,
        layer_ms / 6.0
    );

    // Final RMSNorm (batch)
    let t = Instant::now();
    let x_prenorm = x;
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv_final = vec![0.0f32; seq];
    {
        let mut x_t = vec![0.0f32; seq * dim];
        let mut xfinal_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&x_prenorm, seq, &mut x_t, dim, dim, seq);
        rmsnorm::forward_batch(
            &x_t,
            &weights.gamma_final,
            &mut xfinal_t,
            &mut rms_inv_final,
            dim,
            seq,
        );
        vdsp::mtrans(&xfinal_t, dim, &mut x_final, seq, seq, dim);
    }
    println!(
        "  final rmsnorm:    {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Logits matmul
    let t = Instant::now();
    let mut x_final_row = vec![0.0f32; seq * dim];
    vdsp::mtrans(&x_final, seq, &mut x_final_row, dim, dim, seq);
    let mut logits = vec![0.0f32; seq * vocab];
    vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);
    println!(
        "  logits sgemm:     {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Softcap
    let t = Instant::now();
    if tc.softcap > 0.0 {
        let inv_cap = 1.0 / tc.softcap;
        let mut scaled = vec![0.0f32; seq * vocab];
        vdsp::vsmul(&logits, inv_cap, &mut scaled);
        vdsp::tanhf(&scaled, &mut logits);
        vdsp::sscal(&mut logits, tc.softcap);
    }
    println!(
        "  softcap:          {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Cross-entropy (batched)
    let t = Instant::now();
    let mut dlogits = vec![0.0f32; seq * vocab];
    let _loss = cross_entropy::forward_backward_batch(
        &logits,
        &targets,
        vocab,
        &mut dlogits,
        1.0 / seq as f32,
    );
    println!(
        "  cross-entropy:    {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Backward model-level costs
    println!("\n=== Backward Pass Component Breakdown ===");
    grads.zero_out();
    let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);

    // Scale + softcap backward
    let t = Instant::now();
    let mut dl = vec![0.0f32; seq * vocab];
    vdsp::vsmul(&fwd.dlogits, tc.loss_scale, &mut dl);
    if tc.softcap > 0.0 && !fwd.logits_capped.is_empty() {
        for i in 0..dl.len() {
            let t = fwd.logits_capped[i]; // already unscaled tanh
            dl[i] *= 1.0 - t * t;
        }
    }
    println!(
        "  scale+softcap:    {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Output projection gradients (2 large sgemm)
    let t = Instant::now();
    let mut x_final_row2 = vec![0.0f32; seq * dim];
    vdsp::mtrans(&fwd.x_final, seq, &mut x_final_row2, dim, dim, seq);
    // dembed += dl^T @ x_final_row
    unsafe {
        vdsp::cblas_sgemm(
            101,
            112,
            111,
            vocab as i32,
            dim as i32,
            seq as i32,
            1.0,
            dl.as_ptr(),
            vocab as i32,
            x_final_row2.as_ptr(),
            dim as i32,
            1.0,
            grads.dembed.as_mut_ptr(),
            dim as i32,
        );
    }
    // dx_final_row = dl @ embed
    let mut dx_final_row = vec![0.0f32; seq * dim];
    unsafe {
        vdsp::cblas_sgemm(
            101,
            111,
            111,
            seq as i32,
            dim as i32,
            vocab as i32,
            1.0,
            dl.as_ptr(),
            vocab as i32,
            weights.embed.as_ptr(),
            dim as i32,
            0.0,
            dx_final_row.as_mut_ptr(),
            dim as i32,
        );
    }
    println!(
        "  proj grads sgemm: {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Final RMSNorm backward
    let t = Instant::now();
    let mut dx_final = vec![0.0f32; dim * seq];
    vdsp::mtrans(&dx_final_row, dim, &mut dx_final, seq, seq, dim);
    let mut dy = vec![0.0f32; dim * seq];
    {
        let mut dx_final_t = vec![0.0f32; seq * dim];
        let mut x_prenorm_t2 = vec![0.0f32; seq * dim];
        let mut dy_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&dx_final, seq, &mut dx_final_t, dim, dim, seq);
        vdsp::mtrans(&fwd.x_prenorm, seq, &mut x_prenorm_t2, dim, dim, seq);
        rmsnorm::backward_batch(
            &dx_final_t,
            &x_prenorm_t2,
            &weights.gamma_final,
            &fwd.rms_inv_final,
            &mut dy_t,
            &mut grads.dgamma_final,
            dim,
            seq,
        );
        vdsp::mtrans(&dy_t, dim, &mut dy, seq, seq, dim);
    }
    println!(
        "  final rmsnorm bwd:{:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // 6 layers backward
    let t = Instant::now();
    let mut bwd_ws = engine::layer::BackwardWorkspace::new(&cfg);
    for l in (0..cfg.nlayers).rev() {
        dy = engine::layer::backward(
            &cfg,
            &kernels,
            &weights.layers[l],
            &fwd.caches[l],
            &dy,
            &mut grads.layers[l],
            &mut bwd_ws,
        );
    }
    let layer_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;
    println!(
        "  6 layers bwd:     {:>6.2}ms ({:.2}ms/layer)",
        layer_bwd_ms,
        layer_bwd_ms / 6.0
    );

    // Embedding backward
    let t = Instant::now();
    let mut dy_row = vec![0.0f32; seq * dim];
    vdsp::mtrans(&dy, seq, &mut dy_row, dim, dim, seq);
    embedding::backward(&dy_row, dim, &tokens, &mut grads.dembed);
    println!(
        "  embed bwd:        {:>6.2}ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    // Grad norm + scale
    let t = Instant::now();
    let raw_norm = full_model::grad_norm(&grads);
    println!(
        "  grad_norm:        {:>6.2}ms (norm={:.4})",
        t.elapsed().as_secs_f32() * 1000.0,
        raw_norm
    );
}
