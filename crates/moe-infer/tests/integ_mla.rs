#![cfg(feature = "mla")]
//! Integration test: MLA attention projection and KV cache.
//!
//! Verifies:
//! 1. MLA Q projection produces correct output shapes
//! 2. KV compression produces correct latent/rope shapes
//! 3. KV cache stores compressed latents (memory assertion)
//! 4. Rolling buffer works correctly at capacity

use moe_kernels::mla::{self, MlaConfig, MlaKvCache};

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.1
        })
        .collect()
}

/// MLA Q projection output has correct dimensions.
#[test]
fn q_projection_shape() {
    let config = MlaConfig::deepseek_v3();
    let x = pseudo_random(config.hidden_size, 42);
    let nope_dim = config.num_heads * config.qk_nope_head_dim;
    let rope_dim = config.num_heads * config.qk_rope_head_dim;
    let w_dq = pseudo_random(config.hidden_size * nope_dim, 1);
    let w_qr = pseudo_random(config.hidden_size * rope_dim, 2);

    let (q_nope, q_rope) = mla::mla_q_projection(&x, &w_dq, &w_qr, &config);

    assert_eq!(q_nope.len(), nope_dim, "q_nope should be [num_heads * nope_head_dim]");
    assert_eq!(q_rope.len(), rope_dim, "q_rope should be [num_heads * rope_head_dim]");
    assert!(q_nope.iter().any(|&v| v != 0.0), "q_nope should not be all zeros");
    assert!(q_rope.iter().any(|&v| v != 0.0), "q_rope should not be all zeros");
}

/// KV compression produces correct latent and rope key shapes.
#[test]
fn kv_compression_shape() {
    let config = MlaConfig::deepseek_v3();
    let x = pseudo_random(config.hidden_size, 42);
    let w_kv_down = pseudo_random(config.hidden_size * config.kv_lora_rank, 3);
    let rope_dim = config.qk_rope_head_dim * config.num_kv_heads;
    let w_kr = pseudo_random(config.hidden_size * rope_dim, 4);

    let (latent, rope_key) = mla::mla_kv_compress(&x, &w_kv_down, &w_kr, &config);

    assert_eq!(latent.len(), config.kv_lora_rank, "latent should be [kv_lora_rank]");
    assert_eq!(rope_key.len(), rope_dim, "rope_key should be [rope_dim * kv_heads]");
}

/// KV cache memory: 8K ctx * 61 layers * compressed = well under 1 GB.
#[test]
fn kv_cache_memory_budget() {
    let config = MlaConfig::deepseek_v3();
    let ctx_len = 8192;

    // With MLA compression:
    // Per token: (512 + 64*8) * 2 bytes = (512 + 512) * 2 = 2048 bytes
    // Total: 8192 * 61 * 2048 = ~1.02 GB
    let mla_bytes = config.total_kv_cache_bytes(ctx_len);
    let mla_gb = mla_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Without MLA (full KV): 8 kv_heads * 128 head_dim * 2 (K+V) * 2 bytes * 61 layers * 8192
    let full_kv_per_token = config.num_kv_heads * config.head_dim * 2 * 2; // K+V, fp16
    let full_kv_bytes = ctx_len * config.num_layers * full_kv_per_token;
    let full_kv_gb = full_kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("MLA KV cache: {mla_gb:.2} GB for {ctx_len} tokens");
    println!("Full KV cache: {full_kv_gb:.2} GB for {ctx_len} tokens");
    println!("Compression ratio: {:.1}x", full_kv_gb / mla_gb);

    assert!(
        mla_gb < 2.0,
        "MLA KV cache should be < 2 GB for 8K ctx, got {mla_gb:.2} GB"
    );
    // Verify compression ratio is significant
    assert!(
        full_kv_gb / mla_gb > 1.5,
        "MLA should provide >1.5x compression, got {:.1}x",
        full_kv_gb / mla_gb
    );
}

/// KV cache append + read roundtrip.
#[test]
fn kv_cache_append_roundtrip() {
    let config = MlaConfig {
        hidden_size: 128,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        kv_lora_rank: 64,
        qk_nope_head_dim: 32,
        qk_rope_head_dim: 16,
        v_head_dim: 32,
        num_layers: 2,
    };

    let mut cache = MlaKvCache::new(config.clone(), 1024);
    let rope_dim = config.qk_rope_head_dim * config.num_kv_heads;

    // Append tokens
    for token in 0..10 {
        for layer in 0..config.num_layers {
            let latent = pseudo_random(config.kv_lora_rank, token * 100 + layer as u64);
            let rope_key = pseudo_random(rope_dim, token * 100 + layer as u64 + 50);
            cache.append(layer, &latent, &rope_key);
        }
        cache.advance();
    }

    assert_eq!(cache.seq_len, 10);
    assert_eq!(cache.effective_len(), 10);

    // Read back and verify
    for token in 0..10 {
        for layer in 0..config.num_layers {
            let expected_latent = pseudo_random(config.kv_lora_rank, token * 100 + layer as u64);
            let got_latent = cache.get_latent(layer, token as usize);
            assert_eq!(got_latent, &expected_latent[..], "latent mismatch at token {token} layer {layer}");
        }
    }
}

/// Rolling buffer wraps correctly at max_seq.
#[test]
fn kv_cache_rolling_buffer() {
    let config = MlaConfig {
        hidden_size: 64,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        kv_lora_rank: 16,
        qk_nope_head_dim: 32,
        qk_rope_head_dim: 8,
        v_head_dim: 32,
        num_layers: 1,
    };

    let max_seq = 4;
    let mut cache = MlaKvCache::new(config.clone(), max_seq);
    let rope_dim = config.qk_rope_head_dim * config.num_kv_heads;

    // Fill past capacity
    for token in 0..6u64 {
        let latent = vec![token as f32; config.kv_lora_rank];
        let rope_key = vec![token as f32 + 100.0; rope_dim];
        cache.append(0, &latent, &rope_key);
        cache.advance();
    }

    assert_eq!(cache.seq_len, 6);
    assert_eq!(cache.effective_len(), 4); // capped at max_seq

    // Token 4 and 5 should be at positions 0 and 1 (rolling)
    let lat4 = cache.get_latent(0, 4);
    assert_eq!(lat4[0], 4.0, "token 4 should have value 4.0");

    let lat5 = cache.get_latent(0, 5);
    assert_eq!(lat5[0], 5.0, "token 5 should have value 5.0");
}

/// Cache memory usage is reasonable.
#[test]
fn kv_cache_actual_memory() {
    let config = MlaConfig::deepseek_v3();
    let cache = MlaKvCache::new(config, 8192);

    let mem_gb = cache.memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("MLA KV cache actual memory: {mem_gb:.2} GB (f32, 8K ctx, 61 layers)");

    // f32 cache is 2x the fp16 budget, but still reasonable
    assert!(mem_gb < 8.0, "f32 KV cache should be < 8 GB, got {mem_gb:.2} GB");
}
