//! Benchmark result serialization + hardware collection for the leaderboard.
//!
//! After a benchmark run, call `write_result()` to save a JSON file to
//! `target/bench-result.json`. Users then run `make submit` to post it.

use serde::{Serialize, Deserialize};
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
    use sha2::{Sha256, Digest};
    let mut tmp = result.clone();
    tmp.fingerprint = String::new();
    let json = serde_json::to_string(&tmp).unwrap_or_default();
    let hash = Sha256::digest(json.as_bytes());
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// Write a bench result to `target/bench-result.json`.
pub fn write_result(result: &BenchResult) {
    let path = "target/bench-result.json";
    let json = serde_json::to_string_pretty(result).unwrap();
    std::fs::write(path, &json).unwrap();
    println!("\n  📊 Result saved to {path}");
    println!("  Submit to leaderboard: make submit");
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
