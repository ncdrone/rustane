//! Benchmark result serialization + hardware collection for the leaderboard.
//!
//! After a benchmark run, call `write_result()` to queue a JSON file in
//! `target/bench-results/`. Users then run `make submit` to post pending results.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Serialize, Deserialize, Clone)]
pub struct BenchResult {
    pub schema_version: u32,
    pub rustane_version: String,
    pub git_sha: String,
    pub benchmark: String,
    pub config: ModelInfo,
    pub results: TimingResults,
    pub loss_trace: Vec<f32>,
    pub hardware: HardwareInfo,
    pub submitter: Submitter,
    pub timestamp_utc: String,
    pub fingerprint: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub nlayers: usize,
    pub seq: usize,
    pub params_m: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TimingResults {
    pub ms_per_step: f32,
    pub ms_fwd: f32,
    pub ms_bwd: f32,
    pub ms_upd: f32,
    pub tok_per_s: f32,
    pub loss_start: f32,
    pub loss_end: f32,
    pub loss_delta: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HardwareInfo {
    pub chip: String,
    pub cores_perf: u32,
    pub cores_eff: u32,
    pub ram_gb: u64,
    pub macos: String,
    pub kernel: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Submitter {
    pub name: String,
    pub x_handle: String,
}

/// Collect hardware info via sysctl (macOS only).
pub fn collect_hardware_info() -> HardwareInfo {
    HardwareInfo {
        chip: sysctl_string("machdep.cpu.brand_string"),
        cores_perf: sysctl_u32("hw.perflevel0.logicalcpu"),
        cores_eff: sysctl_u32("hw.perflevel1.logicalcpu"),
        ram_gb: sysctl_u64("hw.memsize") / (1 << 30),
        macos: run_cmd("sw_vers", &["-productVersion"]),
        kernel: sysctl_string("kern.osrelease"),
    }
}

/// Get the short git SHA of HEAD.
pub fn git_sha() -> String {
    run_cmd("git", &["rev-parse", "--short", "HEAD"])
}

/// Get a UTC timestamp without pulling in chrono.
pub fn utc_timestamp() -> String {
    run_cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
}

/// Compute SHA-256 fingerprint of the result (with fingerprint field empty).
pub fn compute_fingerprint(result: &BenchResult) -> String {
    use sha2::{Digest, Sha256};
    let mut tmp = result.clone();
    tmp.fingerprint = String::new();
    let json = serde_json::to_string(&tmp).unwrap_or_default();
    let hash = Sha256::digest(json.as_bytes());
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// Write a bench result to `target/bench-results/` at the workspace root.
pub fn write_result(result: &BenchResult) {
    let json = serde_json::to_string_pretty(result).unwrap();
    let ws_root = run_cmd("git", &["rev-parse", "--show-toplevel"]);
    let queue_dir = if ws_root.is_empty() {
        PathBuf::from("target/bench-results")
    } else {
        PathBuf::from(&ws_root).join("target/bench-results")
    };

    std::fs::create_dir_all(&queue_dir).expect("failed to create bench-results directory");

    let timestamp = queue_timestamp_component(&result.timestamp_utc)
        .or_else(|| queue_timestamp_component(&utc_timestamp()))
        .unwrap_or_else(|| "unknown_000000".to_string());
    let benchmark_slug = sanitize_benchmark_slug(&result.benchmark);
    let fingerprint_prefix = result
        .fingerprint
        .get(..8)
        .filter(|prefix| !prefix.is_empty())
        .unwrap_or("00000000");
    let filename = format!("{timestamp}_{benchmark_slug}_{fingerprint_prefix}.json");
    let final_path = queue_dir.join(&filename);
    let temp_path = queue_dir.join(format!("{filename}.{}.tmp", std::process::id()));

    std::fs::write(&temp_path, &json).expect("failed to write bench result temp file");
    std::fs::rename(&temp_path, &final_path).expect("failed to queue bench result");

    let pending = pending_queue_count(&queue_dir);
    println!(
        "\n  📊 Result queued in target/bench-results/  ({} pending)",
        pending
    );
    println!("  Submit to leaderboard: make submit");
}

fn sanitize_benchmark_slug(benchmark: &str) -> String {
    let sanitized: String = benchmark
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "benchmark".to_string()
    } else {
        sanitized
    }
}

fn queue_timestamp_component(timestamp: &str) -> Option<String> {
    let digits: String = timestamp.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.len() < 14 {
        return None;
    }

    Some(format!("{}_{}", &digits[0..8], &digits[8..14]))
}

fn pending_queue_count(queue_dir: &Path) -> usize {
    std::fs::read_dir(queue_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter(|entry| entry.path().is_file())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        .count()
}

fn sysctl_string(key: &str) -> String {
    run_cmd("sysctl", &["-n", key])
}

fn sysctl_u32(key: &str) -> u32 {
    sysctl_string(key).parse().unwrap_or(0)
}

fn sysctl_u64(key: &str) -> u64 {
    sysctl_string(key).parse().unwrap_or(0)
}

fn run_cmd(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{queue_timestamp_component, sanitize_benchmark_slug};

    #[test]
    fn queue_timestamp_uses_compact_sortable_format() {
        assert_eq!(
            queue_timestamp_component("2026-03-26T11:15:00Z").as_deref(),
            Some("20260326_111500")
        );
    }

    #[test]
    fn benchmark_slug_replaces_non_filename_characters() {
        assert_eq!(
            sanitize_benchmark_slug("train 600m / smoke+retry"),
            "train_600m___smoke_retry"
        );
    }
}
