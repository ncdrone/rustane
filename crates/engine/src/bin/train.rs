//! Full model training loop.
//!
//! Usage: cargo run -p engine --bin train --release -- \
//!   --data /path/to/train_karpathy.bin \
//!   --val /path/to/val_karpathy.bin \
//!   --token-bytes /path/to/token_bytes.bin \
//!   [--steps 72000] [--val-interval 500] [--val-steps 20]

use engine::data::{TokenData, TokenBytes, compute_bpb};
use engine::full_model::{self, ModelWeights, ModelGrads, ModelOptState, ModelForwardWorkspace, ModelBackwardWorkspace, TrainConfig};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: String,
    data_path: PathBuf,
    val_path: Option<PathBuf>,
    token_bytes_path: Option<PathBuf>,
    total_steps: u32,
    warmup_steps: u32,
    max_lr: f32,
    accum_steps: u32,
    loss_scale: f32,
    grad_clip: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    embed_lr_scale: f32,
    min_lr_frac: f32,
    matrix_lr_scale: f32,
    val_interval: u32,
    val_steps: u32,
    checkpoint_interval: u32,
    checkpoint_dir: Option<PathBuf>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = "gpt_karpathy".to_string();
    let mut data_path = None;
    let mut val_path = None;
    let mut token_bytes_path = None;
    let mut total_steps = 72000u32;
    let mut warmup_steps = 0u32; // 0 = auto (2% of total)
    let mut max_lr = 0.0f32;     // 0 = auto
    let mut accum_steps = 0u32;  // 0 = default (10)
    let mut loss_scale = 0.0f32; // 0 = default (256)
    let mut grad_clip = 0.0f32;  // 0 = default (1.0)
    let mut beta2 = 0.0f32;     // 0 = default
    let mut eps = 0.0f32;       // 0 = default
    let mut weight_decay = -1.0f32; // -1 = default (allows 0.0 as explicit value)
    let mut embed_lr_scale = -1.0f32; // -1 = default
    let mut min_lr_frac = -1.0f32;    // -1 = default
    let mut matrix_lr_scale = -1.0f32; // -1 = default
    let mut val_interval = 500u32;
    let mut val_steps = 20u32;
    let mut checkpoint_interval = 1000u32;
    let mut checkpoint_dir = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = args[i+1].clone(); i += 2; }
            "--data" => { data_path = Some(PathBuf::from(&args[i+1])); i += 2; }
            "--val" => { val_path = Some(PathBuf::from(&args[i+1])); i += 2; }
            "--token-bytes" => { token_bytes_path = Some(PathBuf::from(&args[i+1])); i += 2; }
            "--steps" => { total_steps = args[i+1].parse().unwrap(); i += 2; }
            "--warmup" => { warmup_steps = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { max_lr = args[i+1].parse().unwrap(); i += 2; }
            "--accum" => { accum_steps = args[i+1].parse().unwrap(); i += 2; }
            "--loss-scale" => { loss_scale = args[i+1].parse().unwrap(); i += 2; }
            "--grad-clip" => { grad_clip = args[i+1].parse().unwrap(); i += 2; }
            "--beta2" => { beta2 = args[i+1].parse().unwrap(); i += 2; }
            "--eps" => { eps = args[i+1].parse().unwrap(); i += 2; }
            "--wd" => { weight_decay = args[i+1].parse().unwrap(); i += 2; }
            "--embed-lr" => { embed_lr_scale = args[i+1].parse().unwrap(); i += 2; }
            "--min-lr-frac" => { min_lr_frac = args[i+1].parse().unwrap(); i += 2; }
            "--matrix-lr" => { matrix_lr_scale = args[i+1].parse().unwrap(); i += 2; }
            "--val-interval" => { val_interval = args[i+1].parse().unwrap(); i += 2; }
            "--val-steps" => { val_steps = args[i+1].parse().unwrap(); i += 2; }
            "--ckpt-interval" => { checkpoint_interval = args[i+1].parse().unwrap(); i += 2; }
            "--ckpt-dir" => { checkpoint_dir = Some(PathBuf::from(&args[i+1])); i += 2; }
            other => { eprintln!("Unknown arg: {other}"); std::process::exit(1); }
        }
    }

    // Auto warmup: 2% of total steps, min 100
    if warmup_steps == 0 {
        warmup_steps = (total_steps / 50).max(100);
    }
    // Auto LR: 3e-4 for small models, scale down for larger
    if max_lr == 0.0 {
        max_lr = 3e-4;
    }

    Args {
        model,
        data_path: data_path.expect("--data required"),
        val_path,
        token_bytes_path,
        total_steps,
        warmup_steps,
        max_lr,
        accum_steps,
        loss_scale,
        grad_clip,
        beta2,
        eps,
        weight_decay,
        embed_lr_scale,
        min_lr_frac,
        matrix_lr_scale,
        val_interval,
        val_steps,
        checkpoint_interval,
        checkpoint_dir,
    }
}

fn validate(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    val_data: &TokenData,
    token_bytes: Option<&TokenBytes>,
    steps: u32,
    softcap: f32,
) -> (f32, f32) {
    let seq = cfg.seq;
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let mut all_losses = Vec::new();
    let mut all_targets = Vec::new();
    let max_pos = val_data.len() - seq - 1;

    for vs in 0..steps {
        let pos = (vs as usize * seq) % max_pos;
        let input = val_data.tokens(pos, seq);
        let target = val_data.tokens(pos + 1, seq);

        let losses = full_model::forward_losses(cfg, kernels, weights, &input, &target, softcap);
        total_loss += losses.iter().sum::<f32>();
        total_tokens += seq;

        if token_bytes.is_some() {
            all_losses.extend_from_slice(&losses);
            all_targets.extend_from_slice(&target);
        }
    }

    let avg_loss = total_loss / total_tokens as f32;
    let bpb = if let Some(tb) = token_bytes {
        let (bpb, _, _) = compute_bpb(&all_losses, &all_targets, tb);
        bpb
    } else {
        0.0
    };

    (avg_loss, bpb)
}

fn save_checkpoint(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    _opt: &ModelOptState,
    step: u32,
    dir: &std::path::Path,
) {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("ckpt_{step:06}.bin"));

    let mut buf: Vec<u8> = Vec::new();
    // Header: magic + version + step + config
    buf.extend_from_slice(b"RSTK"); // magic
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&step.to_le_bytes());
    buf.extend_from_slice(&(cfg.dim as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.nlayers as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.vocab as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.seq as u32).to_le_bytes());

    // Weights as f32
    write_f32_vec(&mut buf, &weights.embed);
    write_f32_vec(&mut buf, &weights.gamma_final);
    for l in &weights.layers {
        for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w1, &l.w3, &l.w2, &l.gamma1, &l.gamma2] {
            write_f32_vec(&mut buf, w);
        }
    }

    std::fs::write(&path, &buf).unwrap();
    println!("  saved checkpoint: {} ({:.1} MB)", path.display(), buf.len() as f64 / 1e6);
}

fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    for &x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn main() {
    let args = parse_args();
    let cfg = if args.model.starts_with("custom:") {
        // custom:dim,hidden,nlayers,seq  e.g. custom:1024,2816,12,256
        let parts: Vec<usize> = args.model[7..].split(',').map(|s| s.parse().unwrap()).collect();
        let (dim, hidden, nl, seq) = (parts[0], parts[1], parts[2], parts[3]);
        let heads = dim / 128;
        ModelConfig {
            dim, hidden, heads, kv_heads: heads, hd: 128, seq, nlayers: nl,
            vocab: 8192, q_dim: dim, kv_dim: dim, gqa_ratio: 1,
        }
    } else {
        match args.model.as_str() {
            "gpt_karpathy" => ModelConfig::gpt_karpathy(),
            "gpt_1024" => ModelConfig::gpt_1024(),
            "mha_28l" => ModelConfig::mha_28l(),
            other => { eprintln!("Unknown model: {other}"); std::process::exit(1); }
        }
    };

    let per_layer = cfg.dim * cfg.q_dim + cfg.dim * cfg.kv_dim * 2
        + cfg.q_dim * cfg.dim + cfg.dim * cfg.hidden * 3 + cfg.dim * 2;
    let total_params = cfg.vocab * cfg.dim + cfg.nlayers * per_layer + cfg.dim;

    println!("=== Rustane Trainer ===");
    println!("model: {} ({}L, {}D, {}V, {}S)", args.model, cfg.nlayers, cfg.dim, cfg.vocab, cfg.seq);
    println!("params: {:.1}M", total_params as f64 / 1e6);

    // Load data
    println!("\nLoading training data...");
    let train_data = TokenData::open(&args.data_path);
    println!("  {} tokens ({:.1} MB)", train_data.len(), train_data.len() as f64 * 2.0 / 1e6);

    let val_data = args.val_path.as_ref().map(|p| {
        println!("Loading validation data...");
        let d = TokenData::open(p);
        println!("  {} tokens ({:.1} MB)", d.len(), d.len() as f64 * 2.0 / 1e6);
        d
    });

    let token_bytes = args.token_bytes_path.as_ref().map(|p| {
        println!("Loading token_bytes...");
        TokenBytes::load(p)
    });

    // Compile kernels
    println!("\nCompiling 10 ANE kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    println!("  compiled in {:.1}s", t0.elapsed().as_secs_f32());

    // Init model + Metal Adam optimizer
    let metal_adam = MetalAdam::new().expect("Metal GPU required for training");
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let mut tc = TrainConfig::default();
    tc.total_steps = args.total_steps;
    tc.warmup_steps = args.warmup_steps;
    tc.max_lr = args.max_lr;
    if args.accum_steps > 0 { tc.accum_steps = args.accum_steps; }
    if args.loss_scale > 0.0 { tc.loss_scale = args.loss_scale; }
    if args.grad_clip > 0.0 { tc.grad_clip = args.grad_clip; }
    if args.beta2 > 0.0 { tc.beta2 = args.beta2; }
    if args.eps > 0.0 { tc.eps = args.eps; }
    if args.weight_decay >= 0.0 { tc.weight_decay = args.weight_decay; }
    if args.embed_lr_scale >= 0.0 { tc.embed_lr_scale = args.embed_lr_scale; }
    if args.min_lr_frac >= 0.0 { tc.min_lr_frac = args.min_lr_frac; }
    if args.matrix_lr_scale >= 0.0 { tc.matrix_lr_scale = args.matrix_lr_scale; }

    println!("\nTraining config:");
    println!("  lr: {:.0e} (warmup {} → cosine, min_lr_frac: {})", tc.max_lr, tc.warmup_steps, tc.min_lr_frac);
    println!("  accum: {} microbatches, loss_scale: {}", tc.accum_steps, tc.loss_scale);
    println!("  beta1: {}, beta2: {}, eps: {:.0e}", tc.beta1, tc.beta2, tc.eps);
    println!("  weight_decay: {}, grad_clip: {}", tc.weight_decay, tc.grad_clip);
    println!("  embed_lr_scale: {}, matrix_lr_scale: {}", tc.embed_lr_scale, tc.matrix_lr_scale);
    println!("  softcap: {}", tc.softcap);
    println!("  total steps: {}", tc.total_steps);
    println!();

    // Initial validation
    if let Some(ref vd) = val_data {
        let (val_loss, val_bpb) = validate(
            &cfg, &kernels, &weights, vd,
            token_bytes.as_ref(), args.val_steps, tc.softcap,
        );
        println!("step 0: val_loss = {val_loss:.4}, val_bpb = {val_bpb:.4}");
    }

    // Training loop
    let train_start = Instant::now();
    let mut best_bpb = f32::MAX;

    for step in 0..tc.total_steps {
        let step_t0 = Instant::now();

        // Train step (using mmap'd data via TokenData)
        let seq = cfg.seq;
        let max_pos = train_data.len() - seq - 1;
        grads.zero_out();

        let mut total_loss = 0.0f32;
        for micro in 0..tc.accum_steps {
            let pos = ((step as u64 * 7919 + micro as u64 * 104729) % max_pos as u64) as usize;
            let input_tokens = train_data.tokens(pos, seq);
            let target_tokens = train_data.tokens(pos + 1, seq);

            let loss = full_model::forward_ws(
                &cfg, &kernels, &weights, &input_tokens, &target_tokens, tc.softcap, &mut fwd_ws,
            );
            total_loss += loss;
            full_model::backward_ws(
                &cfg, &kernels, &weights, &fwd_ws, &input_tokens, tc.softcap, tc.loss_scale, &mut grads, &mut bwd_ws,
            );
        }

        // Grad norm + clip (descaling fused into Adam GPU kernel)
        let gsc = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);
        let raw_norm = full_model::grad_norm(&grads);
        let gnorm = raw_norm * gsc;
        let combined_scale = if gnorm > tc.grad_clip { tc.grad_clip / raw_norm } else { gsc };
        let lr = full_model::learning_rate(step, &tc);
        full_model::update_weights(&cfg, &mut weights, &grads, &mut opt, step + 1, lr, &tc, &metal_adam, combined_scale);

        let avg_loss = total_loss / tc.accum_steps as f32;
        let step_time = step_t0.elapsed().as_secs_f32();

        // Log every step
        let elapsed = train_start.elapsed().as_secs_f32();
        println!(
            "step {step:>5}: loss = {avg_loss:.4}, lr = {lr:.2e}, gnorm = {gnorm:.2}, time = {step_time:.2}s, elapsed = {elapsed:.0}s"
        );

        // Validation
        if let Some(ref vd) = val_data {
            if (step + 1) % args.val_interval == 0 || step + 1 == tc.total_steps {
                let (val_loss, val_bpb) = validate(
                    &cfg, &kernels, &weights, vd,
                    token_bytes.as_ref(), args.val_steps, tc.softcap,
                );
                println!(
                    "  >>> val_loss = {val_loss:.4}, val_bpb = {val_bpb:.4}{}",
                    if val_bpb < best_bpb { best_bpb = val_bpb; " (best)" } else { "" }
                );
            }
        }

        // Checkpoint
        if let Some(ref dir) = args.checkpoint_dir {
            if (step + 1) % args.checkpoint_interval == 0 || step + 1 == tc.total_steps {
                save_checkpoint(&cfg, &weights, &opt, step + 1, dir);
            }
        }
    }

    let total_time = train_start.elapsed().as_secs_f32();
    println!("\n=== Training complete ===");
    println!("  steps: {}", tc.total_steps);
    println!("  time: {:.1}s ({:.1} min)", total_time, total_time / 60.0);
    if best_bpb < f32::MAX {
        println!("  best val_bpb: {best_bpb:.4}");
    }
}


