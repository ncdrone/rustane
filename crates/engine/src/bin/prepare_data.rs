//! Prepare cached FineWeb shards via the parameter-golf downloader.

use std::path::PathBuf;
use std::process::Command;

struct Args {
    variant: String,
    train_shards: usize,
    with_docs: bool,
    parameter_golf_repo: PathBuf,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut variant = format!("sp{}", 8192);
    let mut train_shards = 80usize;
    let mut with_docs = false;
    let mut parameter_golf_repo = default_parameter_golf_repo().unwrap_or_else(|| {
        eprintln!("could not resolve parameter-golf repo; pass --parameter-golf-repo");
        std::process::exit(1);
    });

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--variant" => {
                variant = args[i + 1].clone();
                i += 2;
            }
            "--train-shards" => {
                train_shards = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--with-docs" => {
                with_docs = true;
                i += 1;
            }
            "--parameter-golf-repo" => {
                parameter_golf_repo = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    Args {
        variant,
        train_shards,
        with_docs,
        parameter_golf_repo,
    }
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

fn main() {
    let args = parse_args();
    let script = args
        .parameter_golf_repo
        .join("data")
        .join("cached_challenge_fineweb.py");
    if !script.exists() {
        eprintln!(
            "parameter-golf downloader not found at {}",
            script.display()
        );
        std::process::exit(1);
    }

    let mut cmd = Command::new("python3");
    cmd.arg(&script)
        .arg("--variant")
        .arg(&args.variant)
        .arg("--train-shards")
        .arg(args.train_shards.to_string())
        .current_dir(&args.parameter_golf_repo);
    if args.with_docs {
        cmd.arg("--with-docs");
    }

    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("failed to start {}: {e}", script.display());
        std::process::exit(1);
    });
    if !status.success() {
        eprintln!("downloader failed with status {status}");
        std::process::exit(1);
    }

    let dataset_name = dataset_dir_for_variant(&args.variant).unwrap_or_else(|| {
        eprintln!("unsupported variant {}", args.variant);
        std::process::exit(1);
    });
    let dataset_dir = args
        .parameter_golf_repo
        .join("data")
        .join("datasets")
        .join(dataset_name);
    println!("dataset_dir={}", dataset_dir.display());
    if let Some(vocab) = args.variant.strip_prefix("sp") {
        println!(
            "tokenizer_path={}",
            args.parameter_golf_repo
                .join("data")
                .join("tokenizers")
                .join(format!("fineweb_{vocab}_bpe.model"))
                .display()
        );
    }
}
