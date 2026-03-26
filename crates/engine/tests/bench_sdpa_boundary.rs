//! Split-SDPA compile smoke matrix.
//!
//! Run:
//!   cargo test -p engine --test bench_sdpa_boundary --release -- --ignored --nocapture

use engine::kernels::sdpa_fwd;
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn custom_config(
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
) -> ModelConfig {
    ModelConfig {
        dim,
        hidden,
        heads,
        kv_heads: heads,
        hd: 128,
        seq,
        nlayers,
        vocab: 8192,
        q_dim: heads * 128,
        kv_dim: heads * 128,
        gqa_ratio: 1,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SdpaCompileResult {
    name: String,
    dim: usize,
    hidden: usize,
    heads: usize,
    nlayers: usize,
    seq: usize,
    compile_ok: bool,
    compile_ms: f32,
    error: Option<String>,
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/sdpa_boundary")
}

fn json_path() -> PathBuf {
    results_dir().join("compile_matrix.json")
}

fn summary_path() -> PathBuf {
    results_dir().join("summary.md")
}

fn ensure_results_dir() {
    fs::create_dir_all(results_dir()).expect("create sdpa results dir");
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with non-string payload".to_string()
    }
}

fn run_case(name: &str, cfg: &ModelConfig) -> SdpaCompileResult {
    let run = std::panic::catch_unwind(|| {
        let t0 = Instant::now();
        let exe = sdpa_fwd::build(cfg).compile(NSQualityOfService::UserInteractive);
        let compile_ms = t0.elapsed().as_secs_f32() * 1000.0;
        (exe, compile_ms)
    });

    match run {
        Ok((Ok(_), compile_ms)) => SdpaCompileResult {
            name: name.to_string(),
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            nlayers: cfg.nlayers,
            seq: cfg.seq,
            compile_ok: true,
            compile_ms,
            error: None,
        },
        Ok((Err(err), compile_ms)) => SdpaCompileResult {
            name: name.to_string(),
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            nlayers: cfg.nlayers,
            seq: cfg.seq,
            compile_ok: false,
            compile_ms,
            error: Some(err.to_string()),
        },
        Err(payload) => SdpaCompileResult {
            name: name.to_string(),
            dim: cfg.dim,
            hidden: cfg.hidden,
            heads: cfg.heads,
            nlayers: cfg.nlayers,
            seq: cfg.seq,
            compile_ok: false,
            compile_ms: 0.0,
            error: Some(panic_payload_to_string(payload)),
        },
    }
}

fn write_results(results: &[SdpaCompileResult]) {
    ensure_results_dir();
    let json = serde_json::to_string_pretty(results).expect("serialize sdpa json");
    fs::write(json_path(), json).expect("write sdpa json");

    let mut summary = String::new();
    summary.push_str("# Split SDPA Compile Smoke Matrix\n\n");
    summary.push_str(
        "| name | dim | hidden | heads | layers | seq | compile | compile_ms | error |\n",
    );
    summary.push_str("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n");
    for r in results {
        let _ = writeln!(
            summary,
            "| {} | {} | {} | {} | {} | {} | {} | {:.1} | {} |",
            r.name,
            r.dim,
            r.hidden,
            r.heads,
            r.nlayers,
            r.seq,
            if r.compile_ok { "ok" } else { "fail" },
            r.compile_ms,
            r.error.as_deref().unwrap_or("")
        );
    }
    fs::write(summary_path(), summary).expect("write sdpa summary");
}

#[test]
#[ignore]
fn bench_sdpa_boundary() {
    let cases = vec![
        ("20B", custom_config(5120, 13824, 40, 64, 512)),
        ("30B", custom_config(6144, 16384, 48, 64, 512)),
        ("40B", custom_config(5120, 13824, 40, 128, 512)),
        ("50B", custom_config(6144, 16384, 48, 110, 512)),
        ("60B", custom_config(6144, 16384, 48, 132, 512)),
    ];

    let mut results = Vec::new();
    for (name, cfg) in cases {
        results.push(run_case(name, &cfg));
    }
    write_results(&results);
    assert!(
        results.iter().all(|r| r.compile_ok),
        "expected split sdpa matrix to compile cleanly"
    );
}
