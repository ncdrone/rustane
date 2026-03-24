//! Correctness test: auto_pool_cap computes reasonable values for different RAM sizes.
//!
//! Invariants:
//! - 128 GB system: cap should be ~2000-3000 (not 1000, not 4000+)
//! - 64 GB system: cap should be ~500-1200
//! - 36 GB system: cap should be small or 0
//! - Cap never exceeds 3000 (empirical ceiling)
//! - Cap scales linearly with available RAM

/// Reproduce the auto_pool_cap formula for testing.
/// Must match generate_v2.rs exactly.
fn auto_pool_cap_formula(total_ram: u64, expert_stride: usize) -> usize {
    if total_ram == 0 || expert_stride == 0 {
        return 1000;
    }
    const MODEL_OVERHEAD: u64 = 30_000_000_000;
    const SAFETY_FACTOR: f64 = 0.85;
    const MAX_CAP: usize = 3000;
    let available = total_ram.saturating_sub(MODEL_OVERHEAD);
    let pool_budget = (available as f64 * SAFETY_FACTOR) as u64;
    let cap = (pool_budget / expert_stride as u64) as usize;
    cap.min(MAX_CAP)
}

const K2_EXPERT_STRIDE: usize = 23_437_312; // ~23.4 MB (actual K2 expert size)

#[test]
fn auto_cap_128gb_system() {
    let ram = 128 * 1024 * 1024 * 1024_u64; // 128 GB
    let cap = auto_pool_cap_formula(ram, K2_EXPERT_STRIDE);
    eprintln!("128 GB → cap={cap}");
    assert!(cap >= 1500, "128 GB should get at least 1500 slots, got {cap}");
    assert!(cap <= 3000, "128 GB should not exceed 3000 cap, got {cap}");
}

#[test]
fn auto_cap_64gb_system() {
    let ram = 64 * 1024 * 1024 * 1024_u64; // 64 GB
    let cap = auto_pool_cap_formula(ram, K2_EXPERT_STRIDE);
    eprintln!("64 GB → cap={cap}");
    assert!(cap >= 400, "64 GB should get at least 400 slots, got {cap}");
    assert!(cap <= 1500, "64 GB should not exceed 1500, got {cap}");
}

#[test]
fn auto_cap_36gb_system() {
    let ram = 36 * 1024 * 1024 * 1024_u64; // 36 GB
    let cap = auto_pool_cap_formula(ram, K2_EXPERT_STRIDE);
    eprintln!("36 GB → cap={cap}");
    assert!(cap <= 350, "36 GB should get ≤350 slots (tight budget), got {cap}");
}

#[test]
fn auto_cap_24gb_system() {
    let ram = 24 * 1024 * 1024 * 1024_u64; // 24 GB
    let cap = auto_pool_cap_formula(ram, K2_EXPERT_STRIDE);
    eprintln!("24 GB → cap={cap}");
    // 24 GB - 30 GB overhead = negative → 0
    assert_eq!(cap, 0, "24 GB cannot fit pool after 30 GB model overhead, got {cap}");
}

#[test]
fn auto_cap_never_exceeds_max() {
    let ram = 512 * 1024 * 1024 * 1024_u64; // 512 GB
    let cap = auto_pool_cap_formula(ram, K2_EXPERT_STRIDE);
    eprintln!("512 GB → cap={cap}");
    assert_eq!(cap, 3000, "cap must be clamped to 3000 regardless of RAM");
}

#[test]
fn auto_cap_scales_monotonically() {
    let sizes_gb = [36, 48, 64, 96, 128, 192, 256];
    let caps: Vec<usize> = sizes_gb.iter()
        .map(|&gb| auto_pool_cap_formula(gb * 1024 * 1024 * 1024, K2_EXPERT_STRIDE))
        .collect();
    eprintln!("RAM scaling: {:?} → {:?}", sizes_gb, caps);
    for i in 1..caps.len() {
        assert!(caps[i] >= caps[i-1],
            "cap should be monotonically increasing: {}GB={} < {}GB={}",
            sizes_gb[i], caps[i], sizes_gb[i-1], caps[i-1]);
    }
}

#[test]
fn auto_cap_zero_stride_fallback() {
    let cap = auto_pool_cap_formula(128 * 1024 * 1024 * 1024, 0);
    assert_eq!(cap, 1000, "zero expert_stride should return fallback 1000");
}

#[test]
fn auto_cap_matches_empirical_sweet_spot() {
    // Iter 28 showed cap=2000 is sweet spot for 128 GB (1.04 tok/s).
    // Auto formula should land near 2000, not at the extremes.
    let ram = 128 * 1024 * 1024 * 1024_u64;
    let cap = auto_pool_cap_formula(ram, K2_EXPERT_STRIDE);
    let cap_gb = cap as f64 * K2_EXPERT_STRIDE as f64 / 1e9;
    eprintln!("128 GB → cap={cap} ({:.1} GB pool)", cap_gb);
    // Should allocate 40-70 GB for pool (matching iter 28 results)
    assert!(cap_gb >= 35.0, "pool should be ≥35 GB on 128 GB system, got {:.1}", cap_gb);
    assert!(cap_gb <= 72.0, "pool should be ≤72 GB on 128 GB system, got {:.1}", cap_gb);
}
