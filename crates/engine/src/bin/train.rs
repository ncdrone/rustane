//! Full model training loop.
//!
//! Usage: cargo run -p engine --bin train --release -- \
//!   --data /path/to/train_karpathy.bin \
//!   --val /path/to/val_karpathy.bin \
//!   --token-bytes /path/to/token_bytes.bin \
//!   [--steps 72000] [--val-interval 500] [--val-steps 20]

use engine::cpu::cross_entropy;
use engine::data::{
    FineWebTrainStream, FineWebValidationData, SentencePieceBpbLut, TokenBytes, TokenData,
    compute_bpb, compute_sentencepiece_bpb,
};
use engine::full_model::{
    self, ModelBackwardWorkspace, ModelForwardWorkspace, ModelGrads, ModelOptState, ModelWeights,
    MuonConfig, TrainConfig,
};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::metal_ffn::MetalFFN;
use engine::metal_muon::MetalMuon;
use engine::model::{FfnActivation, ModelConfig};
use std::path::PathBuf;
use std::process::Command;
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
    ema_decay: Option<f32>,
    swa_interval: Option<u32>,
    swa_start_frac: Option<f32>,
    use_muon: bool,
    muon_lr: Option<f32>,
    muon_momentum: Option<f32>,
    muon_steps: Option<u32>,
    fineweb_eval: bool,
    fineweb_eval_stride: usize,
    fineweb_variant: Option<String>,
    fineweb_dir: Option<PathBuf>,
    fineweb_tokenizer_path: Option<PathBuf>,
    parameter_golf_repo: Option<PathBuf>,
}

enum TrainData {
    Flat(TokenData),
    FineWeb(FineWebTrainStream),
}

impl TrainData {
    fn open(path: &std::path::Path) -> Self {
        if path.is_dir() {
            Self::FineWeb(
                FineWebTrainStream::open_dir(path)
                    .unwrap_or_else(|e| panic!("open fineweb train dir {}: {e}", path.display())),
            )
        } else {
            Self::Flat(TokenData::open(path))
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Flat(data) => data.len(),
            Self::FineWeb(data) => data.len(),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Flat(_) => "flat_u16",
            Self::FineWeb(_) => "fineweb_train_shards",
        }
    }

    fn next_training_pair(&mut self, step: u32, micro: u32, seq: usize) -> (Vec<u32>, Vec<u32>) {
        match self {
            Self::Flat(data) => {
                let max_pos = data.len() - seq - 1;
                let pos = ((step as u64 * 7919 + micro as u64 * 104729) % max_pos as u64) as usize;
                (data.tokens(pos, seq), data.tokens(pos + 1, seq))
            }
            Self::FineWeb(data) => {
                let chunk = data.next_tokens(seq + 1);
                (chunk[..seq].to_vec(), chunk[1..].to_vec())
            }
        }
    }
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = "gpt_karpathy".to_string();
    let mut data_path = None;
    let mut val_path = None;
    let mut token_bytes_path = None;
    let mut total_steps = 72000u32;
    let mut warmup_steps = 0u32; // 0 = auto (2% of total)
    let mut max_lr = 0.0f32; // 0 = auto
    let mut accum_steps = 0u32; // 0 = default (10)
    let mut loss_scale = 0.0f32; // 0 = default (256)
    let mut grad_clip = 0.0f32; // 0 = default (1.0)
    let mut beta2 = 0.0f32; // 0 = default
    let mut eps = 0.0f32; // 0 = default
    let mut weight_decay = -1.0f32; // -1 = default (allows 0.0 as explicit value)
    let mut embed_lr_scale = -1.0f32; // -1 = default
    let mut min_lr_frac = -1.0f32; // -1 = default
    let mut matrix_lr_scale = -1.0f32; // -1 = default
    let mut val_interval = 500u32;
    let mut val_steps = 20u32;
    let mut checkpoint_interval = 1000u32;
    let mut checkpoint_dir = None;
    let mut ema_decay = None;
    let mut swa_interval = None;
    let mut swa_start_frac = None;
    let mut use_muon = false;
    let mut muon_lr = None;
    let mut muon_momentum = None;
    let mut muon_steps = None;
    let mut fineweb_eval = false;
    let mut fineweb_eval_stride = 64usize;
    let mut fineweb_variant = None;
    let mut fineweb_dir = None;
    let mut fineweb_tokenizer_path = None;
    let mut parameter_golf_repo = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = args[i + 1].clone();
                i += 2;
            }
            "--data" => {
                data_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--val" => {
                val_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--token-bytes" => {
                token_bytes_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--steps" => {
                total_steps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--warmup" => {
                warmup_steps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--lr" => {
                max_lr = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--accum" => {
                accum_steps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--loss-scale" => {
                loss_scale = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--grad-clip" => {
                grad_clip = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--beta2" => {
                beta2 = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--eps" => {
                eps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--wd" => {
                weight_decay = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--embed-lr" => {
                embed_lr_scale = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--min-lr-frac" => {
                min_lr_frac = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--matrix-lr" => {
                matrix_lr_scale = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--val-interval" => {
                val_interval = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--val-steps" => {
                val_steps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ckpt-interval" => {
                checkpoint_interval = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ckpt-dir" => {
                checkpoint_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--ema-decay" => {
                ema_decay = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--swa-interval" => {
                swa_interval = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--swa-start-frac" => {
                swa_start_frac = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--muon" => {
                use_muon = true;
                i += 1;
            }
            "--muon-lr" => {
                muon_lr = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--muon-momentum" => {
                muon_momentum = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--muon-steps" => {
                muon_steps = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--fineweb-eval" | "--eval-fineweb" => {
                fineweb_eval = true;
                i += 1;
            }
            "--fineweb-eval-stride" => {
                fineweb_eval_stride = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--fineweb-variant" => {
                fineweb_variant = Some(args[i + 1].clone());
                i += 2;
            }
            "--fineweb-dir" => {
                fineweb_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--fineweb-tokenizer" => {
                fineweb_tokenizer_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--parameter-golf-repo" => {
                parameter_golf_repo = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
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
        ema_decay,
        swa_interval,
        swa_start_frac,
        use_muon,
        muon_lr,
        muon_momentum,
        muon_steps,
        fineweb_eval,
        fineweb_eval_stride,
        fineweb_variant,
        fineweb_dir,
        fineweb_tokenizer_path,
        parameter_golf_repo,
    }
}

fn select_eval_weights<'a>(
    live: &'a ModelWeights,
    ema: Option<&'a ModelWeights>,
    swa: Option<&'a ModelWeights>,
) -> (&'a ModelWeights, &'static str) {
    if let Some(swa_weights) = swa {
        (swa_weights, "swa")
    } else if let Some(ema_weights) = ema {
        (ema_weights, "ema")
    } else {
        (live, "live")
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
    fwd_opts: full_model::ForwardOptions<'_>,
) -> (f32, f32) {
    let seq = cfg.seq;
    let vocab = cfg.vocab;
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let mut all_losses = Vec::new();
    let mut all_targets = Vec::new();
    let max_pos = val_data.len() - seq - 1;
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);

    for vs in 0..steps {
        let pos = (vs as usize * seq) % max_pos;
        let input = val_data.tokens(pos, seq);
        let target = val_data.tokens(pos + 1, seq);

        let _ = full_model::forward_ws_with_options(
            cfg,
            kernels,
            weights,
            &input,
            &target,
            softcap,
            &mut fwd_ws,
            fwd_opts,
        );
        let mut losses = Vec::with_capacity(seq);
        for s in 0..seq {
            let tok_logits = &fwd_ws.logits[s * vocab..(s + 1) * vocab];
            let (loss, _) = cross_entropy::forward(tok_logits, target[s] as usize);
            total_loss += loss;
            losses.push(loss);
        }
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

fn validate_fineweb(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &ModelWeights,
    val_data: &FineWebValidationData,
    bpb_lut: &SentencePieceBpbLut,
    stride: usize,
    softcap: f32,
    fwd_opts: full_model::ForwardOptions<'_>,
) -> (f32, f32) {
    let seq = cfg.seq;
    let vocab = cfg.vocab;
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let mut total_bytes = 0usize;
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let all_tokens = val_data.tokens();
    let total_eval_tokens = all_tokens.len().saturating_sub(1);
    let stride = stride.max(1);

    for ws in (0..total_eval_tokens).step_by(stride) {
        let end = (ws + seq).min(total_eval_tokens);
        let wlen = end.saturating_sub(ws);
        if wlen == 0 {
            continue;
        }
        let mut input = vec![0u32; seq];
        let mut target = vec![0u32; seq];
        input[..wlen].copy_from_slice(&all_tokens[ws..end]);
        target[..wlen].copy_from_slice(&all_tokens[ws + 1..end + 1]);

        let _ = full_model::forward_ws_with_options(
            cfg,
            kernels,
            weights,
            &input,
            &target,
            softcap,
            &mut fwd_ws,
            fwd_opts,
        );

        let score_start = if ws == 0 {
            0
        } else {
            wlen.saturating_sub(stride)
        };
        let mut losses = vec![0.0f32; wlen - score_start];
        for s in score_start..wlen {
            let tok_logits = &fwd_ws.logits[s * vocab..(s + 1) * vocab];
            let (loss, _) = cross_entropy::forward(tok_logits, target[s] as usize);
            total_loss += loss;
            total_tokens += 1;
            losses[s - score_start] = loss;
        }

        let (_, _, bytes) = compute_sentencepiece_bpb(
            &losses,
            &input[score_start..wlen],
            &target[score_start..wlen],
            bpb_lut,
        );
        total_bytes += bytes;
    }

    let avg_loss = total_loss / total_tokens as f32;
    let bpb = if total_bytes > 0 {
        total_loss / (std::f32::consts::LN_2 * total_bytes as f32)
    } else {
        0.0
    };
    (avg_loss, bpb)
}

fn default_parameter_golf_repo() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("PARAMETER_GOLF_REPO") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    let rustane_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .parent()?
        .to_path_buf();
    let candidate = rustane_root
        .parent()?
        .parent()?
        .join("openai/parameter-golf");
    candidate.exists().then_some(candidate)
}

fn dataset_dir_for_variant(variant: &str) -> Option<String> {
    if variant == "byte260" {
        Some("fineweb10B_byte260".to_string())
    } else if let Some(rest) = variant.strip_prefix("sp") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            Some(format!("fineweb10B_{variant}"))
        } else {
            None
        }
    } else {
        None
    }
}

fn ensure_fineweb_variant_cached(pg_repo: &std::path::Path, variant: &str) -> Result<(), String> {
    let script = pg_repo.join("data/cached_challenge_fineweb.py");
    if !script.exists() {
        return Err(format!(
            "parameter-golf downloader not found at {}",
            script.display()
        ));
    }
    let output = Command::new("python3")
        .arg(&script)
        .arg("--variant")
        .arg(variant)
        .arg("--train-shards")
        .arg("0")
        .current_dir(pg_repo)
        .output()
        .map_err(|e| format!("failed to start {}: {e}", script.display()))?;
    if !output.status.success() {
        return Err(format!(
            "fineweb download failed via {}: {}",
            script.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn resolve_fineweb_assets(
    cfg: &ModelConfig,
    args: &Args,
) -> Result<Option<(FineWebValidationData, SentencePieceBpbLut, PathBuf, PathBuf)>, String> {
    if !args.fineweb_eval {
        return Ok(None);
    }

    let (dataset_dir, tokenizer_path) = if let (Some(dir), Some(tok)) =
        (&args.fineweb_dir, &args.fineweb_tokenizer_path)
    {
        (dir.clone(), tok.clone())
    } else {
        let variant = args
            .fineweb_variant
            .clone()
            .unwrap_or_else(|| format!("sp{}", cfg.vocab));
        let dataset_name = dataset_dir_for_variant(&variant)
            .ok_or_else(|| format!("unsupported fineweb variant {variant}"))?;
        if variant == "byte260" {
            return Err(
                "byte260 fineweb eval is not yet supported; use a SentencePiece variant".into(),
            );
        }
        let pg_repo = args
            .parameter_golf_repo
            .clone()
            .or_else(default_parameter_golf_repo)
            .ok_or_else(|| {
                "could not resolve parameter-golf repo; pass --parameter-golf-repo or set PARAMETER_GOLF_REPO"
                    .to_string()
            })?;
        ensure_fineweb_variant_cached(&pg_repo, &variant)?;
        let dir = pg_repo.join("data/datasets").join(dataset_name);
        let tokenizer_path = pg_repo
            .join("data/tokenizers")
            .join(format!("fineweb_{}_bpe.model", &variant[2..]));
        (dir, tokenizer_path)
    };

    if !dataset_dir.exists() {
        return Err(format!(
            "fineweb dataset dir not found: {}",
            dataset_dir.display()
        ));
    }
    if !tokenizer_path.exists() {
        return Err(format!(
            "fineweb tokenizer not found: {}",
            tokenizer_path.display()
        ));
    }

    let val_data = FineWebValidationData::load_dir(&dataset_dir).map_err(|e| {
        format!(
            "failed to load fineweb validation from {}: {e}",
            dataset_dir.display()
        )
    })?;
    let bpb_lut = SentencePieceBpbLut::from_python(&tokenizer_path, cfg.vocab).map_err(|e| {
        format!(
            "failed to build SentencePiece BPB LUT from {}: {e}",
            tokenizer_path.display()
        )
    })?;
    Ok(Some((val_data, bpb_lut, dataset_dir, tokenizer_path)))
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
    buf.extend_from_slice(&2u32.to_le_bytes()); // version
    buf.extend_from_slice(&step.to_le_bytes());
    buf.extend_from_slice(&(cfg.dim as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.nlayers as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.vocab as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.seq as u32).to_le_bytes());
    buf.extend_from_slice(&cfg.ffn_activation.checkpoint_code().to_le_bytes());

    // Weights as f32
    write_f32_vec(&mut buf, &weights.embed);
    write_f32_vec(&mut buf, &weights.gamma_final);
    for l in &weights.layers {
        for w in [
            &l.wq, &l.wk, &l.wv, &l.wo, &l.w1, &l.w3, &l.w2, &l.gamma1, &l.gamma2,
        ] {
            write_f32_vec(&mut buf, w);
        }
    }

    std::fs::write(&path, &buf).unwrap();
    println!(
        "  saved checkpoint: {} ({:.1} MB)",
        path.display(),
        buf.len() as f64 / 1e6
    );
}

fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    for &x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn main() {
    let args = parse_args();
    let cfg = if args.model.starts_with("custom:") {
        // custom:dim,hidden,nlayers,seq  e.g. custom:1024,2816,12,256
        let parts: Vec<usize> = args.model[7..]
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let (dim, hidden, nl, seq) = (parts[0], parts[1], parts[2], parts[3]);
        let heads = dim / 128;
        ModelConfig {
            dim,
            hidden,
            heads,
            kv_heads: heads,
            hd: 128,
            seq,
            nlayers: nl,
            vocab: 8192,
            q_dim: dim,
            kv_dim: dim,
            gqa_ratio: 1,
            ffn_activation: FfnActivation::SwiGlu,
        }
    } else {
        match args.model.as_str() {
            "gpt_karpathy" => ModelConfig::gpt_karpathy(),
            "gpt_karpathy_pg" => ModelConfig::gpt_karpathy_pg(),
            "gpt_karpathy_leaky_relu_sq" => ModelConfig::gpt_karpathy_leaky_relu_sq(),
            "gpt_karpathy_gqa" => ModelConfig::gpt_karpathy_gqa(),
            "gpt_1024" => ModelConfig::gpt_1024(),
            "mha_28l" => ModelConfig::mha_28l(),
            other => {
                eprintln!("Unknown model: {other}");
                std::process::exit(1);
            }
        }
    };

    let per_layer = cfg.dim * cfg.q_dim
        + cfg.dim * cfg.kv_dim * 2
        + cfg.q_dim * cfg.dim
        + cfg.dim * cfg.hidden * 3
        + cfg.dim * 2;
    let total_params = cfg.vocab * cfg.dim + cfg.nlayers * per_layer + cfg.dim;

    println!("=== Rustane Trainer ===");
    println!(
        "model: {} ({}L, {}D, {}V, {}S)",
        args.model, cfg.nlayers, cfg.dim, cfg.vocab, cfg.seq
    );
    println!("params: {:.1}M", total_params as f64 / 1e6);

    let fineweb_eval = resolve_fineweb_assets(&cfg, &args).unwrap_or_else(|e| {
        eprintln!("FineWeb eval setup failed: {e}");
        std::process::exit(1);
    });

    // Load data
    println!("\nLoading training data...");
    let mut train_data = TrainData::open(&args.data_path);
    println!(
        "  {} tokens ({:.1} MB, {})",
        train_data.len(),
        train_data.len() as f64 * 2.0 / 1e6,
        train_data.kind(),
    );

    let val_data = args.val_path.as_ref().map(|p| {
        println!("Loading validation data...");
        let d = TokenData::open(p);
        println!(
            "  {} tokens ({:.1} MB)",
            d.len(),
            d.len() as f64 * 2.0 / 1e6
        );
        d
    });

    let token_bytes = args.token_bytes_path.as_ref().map(|p| {
        println!("Loading token_bytes...");
        TokenBytes::load(p)
    });
    if let Some((ref fineweb_data, _, ref dataset_dir, ref tokenizer_path)) = fineweb_eval {
        println!("Loading FineWeb validation data...");
        println!(
            "  dataset: {} ({} tokens, {} sequences)",
            dataset_dir.display(),
            fineweb_data.len(),
            (fineweb_data.len() - 1) / cfg.seq
        );
        println!("  tokenizer: {}", tokenizer_path.display());
    }

    // Compile kernels
    println!("\nCompiling 10 ANE kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    println!("  compiled in {:.1}s", t0.elapsed().as_secs_f32());

    let requested_gpu_ffn = env_flag("RUSTANE_USE_GPU_FFN");
    let requested_gpu_ffn_bwd_dx = env_flag("RUSTANE_USE_GPU_FFN_BWD_DX");
    let metal_ffn = if requested_gpu_ffn {
        MetalFFN::new(&cfg)
    } else {
        None
    };
    let fwd_opts = if let Some(ref metal_ffn) = metal_ffn {
        full_model::ForwardOptions::gpu_ffn(metal_ffn)
    } else {
        full_model::ForwardOptions::default()
    };
    let mut bwd_opts = if requested_gpu_ffn_bwd_dx {
        metal_ffn
            .as_ref()
            .map(full_model::BackwardOptions::gpu_ffn_dx)
            .unwrap_or_default()
    } else {
        full_model::BackwardOptions::default()
    };

    // Init model + Metal Adam optimizer
    let metal_adam = MetalAdam::new().expect("Metal GPU required for training");
    let mut muon_cfg = MuonConfig::default();
    if let Some(v) = args.muon_lr {
        muon_cfg.muon_lr = v;
    }
    if let Some(v) = args.muon_momentum {
        muon_cfg.muon_momentum = v;
    }
    if let Some(v) = args.muon_steps {
        muon_cfg.newton_schulz_steps = v;
    }
    let use_muon = args.use_muon
        || args.muon_lr.is_some()
        || args.muon_momentum.is_some()
        || args.muon_steps.is_some();
    let metal_muon = if use_muon {
        Some(MetalMuon::new().expect("Metal GPU required for Muon"))
    } else {
        None
    };
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let mut fwd_ws = ModelForwardWorkspace::new(&cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(&cfg);

    let mut tc = TrainConfig::default();
    tc.total_steps = args.total_steps;
    tc.warmup_steps = args.warmup_steps;
    tc.max_lr = args.max_lr;
    if args.accum_steps > 0 {
        tc.accum_steps = args.accum_steps;
    }
    if args.loss_scale > 0.0 {
        tc.loss_scale = args.loss_scale;
    }
    if args.grad_clip > 0.0 {
        tc.grad_clip = args.grad_clip;
    }
    if args.beta2 > 0.0 {
        tc.beta2 = args.beta2;
    }
    if args.eps > 0.0 {
        tc.eps = args.eps;
    }
    if args.weight_decay >= 0.0 {
        tc.weight_decay = args.weight_decay;
    }
    if args.embed_lr_scale >= 0.0 {
        tc.embed_lr_scale = args.embed_lr_scale;
    }
    if args.min_lr_frac >= 0.0 {
        tc.min_lr_frac = args.min_lr_frac;
    }
    if args.matrix_lr_scale >= 0.0 {
        tc.matrix_lr_scale = args.matrix_lr_scale;
    }
    if let Some(ema_decay) = args.ema_decay {
        if !(0.0..1.0).contains(&ema_decay) {
            eprintln!("--ema-decay must be in [0, 1)");
            std::process::exit(1);
        }
        tc.ema_decay = Some(ema_decay);
    }
    if let Some(swa_interval) = args.swa_interval {
        if swa_interval == 0 {
            eprintln!("--swa-interval must be > 0");
            std::process::exit(1);
        }
        tc.swa_interval = Some(swa_interval);
    }
    if let Some(swa_start_frac) = args.swa_start_frac {
        if !(0.0..=1.0).contains(&swa_start_frac) || swa_start_frac == 0.0 {
            eprintln!("--swa-start-frac must be in (0, 1]");
            std::process::exit(1);
        }
        tc.swa_start_frac = Some(swa_start_frac);
    }
    if tc.accum_steps > 1 {
        bwd_opts = bwd_opts.with_ffn_weight_cache();
    }
    let mut ema_weights = tc.ema_decay.map(|_| weights.clone());
    let mut swa_weights: Option<ModelWeights> = None;
    let mut swa_count = 0u32;

    println!("\nTraining config:");
    println!(
        "  lr: {:.0e} (warmup {} → cosine, min_lr_frac: {})",
        tc.max_lr, tc.warmup_steps, tc.min_lr_frac
    );
    println!(
        "  accum: {} microbatches, loss_scale: {}",
        tc.accum_steps, tc.loss_scale
    );
    println!(
        "  beta1: {}, beta2: {}, eps: {:.0e}",
        tc.beta1, tc.beta2, tc.eps
    );
    println!(
        "  weight_decay: {}, grad_clip: {}",
        tc.weight_decay, tc.grad_clip
    );
    println!(
        "  embed_lr_scale: {}, matrix_lr_scale: {}",
        tc.embed_lr_scale, tc.matrix_lr_scale
    );
    println!("  softcap: {}", tc.softcap);
    println!(
        "  optimizer: {}",
        if use_muon { "muon+adam" } else { "adam" }
    );
    if use_muon {
        println!(
            "  muon: lr {}, momentum {}, steps {}",
            muon_cfg.muon_lr, muon_cfg.muon_momentum, muon_cfg.newton_schulz_steps
        );
    }
    println!(
        "  ema_decay: {}",
        tc.ema_decay
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "disabled".to_string())
    );
    println!(
        "  swa_interval: {}",
        tc.swa_interval
            .map(|v| v.to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );
    println!(
        "  swa_start_frac: {}",
        tc.swa_start_frac
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "disabled".to_string())
    );
    println!(
        "  fineweb_eval: {}",
        if fineweb_eval.is_some() {
            "enabled"
        } else {
            "disabled"
        }
    );
    if fineweb_eval.is_some() {
        println!("  fineweb_eval_stride: {}", args.fineweb_eval_stride);
    }
    println!(
        "  gpu_ffn: {}",
        if fwd_opts.use_gpu_ffn {
            "enabled"
        } else if requested_gpu_ffn {
            "requested but unavailable"
        } else {
            "disabled"
        }
    );
    println!(
        "  gpu_ffn_bwd_dx: {}",
        if bwd_opts.use_gpu_ffn_dx {
            "enabled"
        } else if requested_gpu_ffn_bwd_dx {
            "requested but unavailable"
        } else {
            "disabled"
        }
    );
    println!("  total steps: {}", tc.total_steps);
    println!();

    // Initial validation
    let (eval_weights, eval_label) =
        select_eval_weights(&weights, ema_weights.as_ref(), swa_weights.as_ref());
    if let Some((ref fineweb_data, ref bpb_lut, _, _)) = fineweb_eval {
        let (val_loss, val_bpb) = validate_fineweb(
            &cfg,
            &kernels,
            eval_weights,
            fineweb_data,
            bpb_lut,
            args.fineweb_eval_stride,
            tc.softcap,
            fwd_opts,
        );
        println!(
            "step 0: fineweb_val_loss = {val_loss:.4}, fineweb_val_bpb = {val_bpb:.4} ({eval_label})"
        );
    } else if let Some(ref vd) = val_data {
        let (val_loss, val_bpb) = validate(
            &cfg,
            &kernels,
            eval_weights,
            vd,
            token_bytes.as_ref(),
            args.val_steps,
            tc.softcap,
            fwd_opts,
        );
        println!("step 0: val_loss = {val_loss:.4}, val_bpb = {val_bpb:.4} ({eval_label})");
    }

    // Training loop
    let train_start = Instant::now();
    let mut best_bpb = f32::MAX;

    for step in 0..tc.total_steps {
        let step_t0 = Instant::now();

        // Train step (using mmap'd data via TokenData)
        grads.zero_out();

        let mut total_loss = 0.0f32;
        for micro in 0..tc.accum_steps {
            let (input_tokens, target_tokens) = train_data.next_training_pair(step, micro, cfg.seq);

            let loss = full_model::forward_ws_with_options(
                &cfg,
                &kernels,
                &weights,
                &input_tokens,
                &target_tokens,
                tc.softcap,
                &mut fwd_ws,
                fwd_opts,
            );
            total_loss += loss;
            full_model::backward_ws_with_options(
                &cfg,
                &kernels,
                &weights,
                &fwd_ws,
                &input_tokens,
                tc.softcap,
                tc.loss_scale,
                &mut grads,
                &mut bwd_ws,
                bwd_opts,
            );
        }

        // Grad norm + clip (descaling fused into Adam GPU kernel)
        let gsc = 1.0 / (tc.accum_steps as f32 * tc.loss_scale);
        let raw_norm = full_model::grad_norm(&grads);
        let gnorm = raw_norm * gsc;
        let combined_scale = if gnorm > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        let lr = full_model::learning_rate(step, &tc);
        if let Some(ref metal_muon) = metal_muon {
            full_model::update_weights_muon(
                &cfg,
                &mut weights,
                &grads,
                &mut opt,
                step + 1,
                lr,
                &tc,
                &muon_cfg,
                &metal_adam,
                metal_muon,
                combined_scale,
            );
        } else {
            full_model::update_weights(
                &cfg,
                &mut weights,
                &grads,
                &mut opt,
                step + 1,
                lr,
                &tc,
                &metal_adam,
                combined_scale,
            );
        }
        if let Some(ema_decay) = tc.ema_decay {
            if let Some(ema) = ema_weights.as_mut() {
                ema.blend_towards(&weights, 1.0 - ema_decay);
            }
        }
        if full_model::should_collect_swa(step + 1, &tc) {
            if let Some(swa) = swa_weights.as_mut() {
                let next_count = swa_count + 1;
                swa.blend_towards(&weights, 1.0 / next_count as f32);
                swa_count = next_count;
            } else {
                swa_weights = Some(weights.clone());
                swa_count = 1;
            }
        }

        let avg_loss = total_loss / tc.accum_steps as f32;
        let step_time = step_t0.elapsed().as_secs_f32();

        // Log every step
        let elapsed = train_start.elapsed().as_secs_f32();
        println!(
            "step {step:>5}: loss = {avg_loss:.4}, lr = {lr:.2e}, gnorm = {gnorm:.2}, time = {step_time:.2}s, elapsed = {elapsed:.0}s"
        );

        // Validation
        let (eval_weights, eval_label) =
            select_eval_weights(&weights, ema_weights.as_ref(), swa_weights.as_ref());
        if let Some((ref fineweb_data, ref bpb_lut, _, _)) = fineweb_eval {
            if (step + 1) % args.val_interval == 0 || step + 1 == tc.total_steps {
                let (val_loss, val_bpb) = validate_fineweb(
                    &cfg,
                    &kernels,
                    eval_weights,
                    fineweb_data,
                    bpb_lut,
                    args.fineweb_eval_stride,
                    tc.softcap,
                    fwd_opts,
                );
                println!(
                    "  >>> fineweb_val_loss = {val_loss:.4}, fineweb_val_bpb = {val_bpb:.4} [{}{}]",
                    eval_label,
                    if val_bpb < best_bpb {
                        best_bpb = val_bpb;
                        ", best"
                    } else {
                        ""
                    }
                );
            }
        } else if let Some(ref vd) = val_data {
            if (step + 1) % args.val_interval == 0 || step + 1 == tc.total_steps {
                let (val_loss, val_bpb) = validate(
                    &cfg,
                    &kernels,
                    eval_weights,
                    vd,
                    token_bytes.as_ref(),
                    args.val_steps,
                    tc.softcap,
                    fwd_opts,
                );
                println!(
                    "  >>> val_loss = {val_loss:.4}, val_bpb = {val_bpb:.4} [{}{}]",
                    eval_label,
                    if val_bpb < best_bpb {
                        best_bpb = val_bpb;
                        ", best"
                    } else {
                        ""
                    }
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
