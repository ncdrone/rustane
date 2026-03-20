//! Integration test: 3-tier KV cache and sparse attention.
//!
//! Verifies tiered KV cache behavior: hot/warm/cold tier promotion,
//! memory budgets, and correctness of the Infini-attention cold tier.

use moe_kernels::sparse_attn::{SparseAttnConfig, TieredKvCache};

fn make_kv(dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.1
            })
            .collect()
    };
    let k = rand_vec(dim);
    let v = rand_vec(dim);
    (k, v)
}

/// Tiered cache fills hot tier first.
#[test]
fn hot_tier_fills_first() {
    let config = SparseAttnConfig {
        num_heads: 4,
        head_dim: 32,
        ..Default::default()
    };
    let kv_dim = config.num_heads * config.head_dim;
    let mut cache = TieredKvCache::new(config, 8, 16);

    for i in 0..8 {
        let (k, v) = make_kv(kv_dim, i);
        cache.append(&k, &v);
    }

    assert_eq!(cache.hot.current_len, 8);
    assert_eq!(cache.warm.current_len, 0);
}

/// Hot overflow promotes to warm tier.
#[test]
fn hot_overflow_promotes_to_warm() {
    let config = SparseAttnConfig {
        num_heads: 4,
        head_dim: 32,
        ..Default::default()
    };
    let kv_dim = config.num_heads * config.head_dim;
    let mut cache = TieredKvCache::new(config, 4, 8);

    for i in 0..6 {
        let (k, v) = make_kv(kv_dim, i);
        cache.append(&k, &v);
    }

    assert_eq!(cache.hot.current_len, 4);
    assert_eq!(cache.warm.current_len, 2);
}

/// Warm overflow absorbs into cold tier (Infini-attention).
#[test]
fn warm_overflow_absorbs_to_cold() {
    let config = SparseAttnConfig {
        num_heads: 2,
        head_dim: 16,
        ..Default::default()
    };
    let kv_dim = config.num_heads * config.head_dim;
    let mut cache = TieredKvCache::new(config, 2, 2);

    // Fill hot (2) + warm (2) + overflow (1 more triggers cold absorption)
    for i in 0..5 {
        let (k, v) = make_kv(kv_dim, i);
        cache.append(&k, &v);
    }

    assert_eq!(cache.hot.current_len, 2);
    assert_eq!(cache.warm.current_len, 2);
    // Cold should have absorbed 1 token
    assert!(cache.cold.norm[0] > 0.0, "cold tier should have absorbed tokens");
}

/// Memory budget: hot 64K + warm 1M should be manageable.
#[test]
fn memory_budget() {
    let config = SparseAttnConfig {
        num_heads: 56,
        head_dim: 128,
        ..Default::default()
    };

    // Hot: 64K tokens, Warm: 1M tokens
    let hot_tokens = 65536;
    let warm_tokens = 1_048_576;
    let kv_dim = config.num_heads * config.head_dim;

    // Estimate without allocating (too much memory for test)
    let hot_bytes = hot_tokens * kv_dim * 2 * 4; // K+V, f32
    let warm_bytes = warm_tokens * kv_dim * 2 * 4;

    let hot_gb = hot_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let warm_gb = warm_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("Hot tier (64K tokens, f32): {hot_gb:.2} GB");
    println!("Warm tier (1M tokens, f32): {warm_gb:.2} GB");
    println!("Warm tier (1M tokens, 3-bit): {:.2} GB", warm_gb * 3.0 / 32.0);

    // Hot at f32 should be < 8 GB
    assert!(hot_gb < 8.0, "hot tier too large: {hot_gb:.2} GB");
    // Warm at 3-bit should be < 8 GB (1M tokens is a lot)
    let warm_3bit_gb = warm_gb * 3.0 / 32.0;
    assert!(warm_3bit_gb < 8.0, "warm tier (3-bit) too large: {warm_3bit_gb:.2} GB");
}

/// Cold tier running mean converges.
#[test]
fn cold_tier_convergence() {
    let config = SparseAttnConfig {
        num_heads: 2,
        head_dim: 4,
        ..Default::default()
    };
    let kv_dim = config.num_heads * config.head_dim;
    let mut cache = TieredKvCache::new(config, 1, 1);

    // Force 100 tokens through to cold tier
    for i in 0..102 {
        // Constant value = 0.5 for all KV
        let k = vec![0.5f32; kv_dim];
        let v = vec![0.5f32; kv_dim];
        cache.append(&k, &v);
    }

    // Cold memory should converge to 0.5
    for &m in &cache.cold.memory {
        assert!(
            (m - 0.5).abs() < 0.01,
            "cold memory should converge to 0.5, got {m}"
        );
    }
}

/// Total tokens tracked correctly.
#[test]
fn total_tokens_tracked() {
    let config = SparseAttnConfig {
        num_heads: 2,
        head_dim: 8,
        ..Default::default()
    };
    let kv_dim = config.num_heads * config.head_dim;
    let mut cache = TieredKvCache::new(config, 4, 4);

    for i in 0..8 {
        let (k, v) = make_kv(kv_dim, i);
        cache.append(&k, &v);
    }

    assert_eq!(cache.total_tokens(), 8);
}
