//! Correctness test: IOSurface alignment edge cases (E10 / V-ANE-3).
//!
//! What: Validates that all ANE IOSurface spatial widths are multiples of 16
//! and channels meet minimum requirements. Tests boundary conditions.
//!
//! Invariant: Every IOSurface created for ANE MLA must have spatial width % 16 == 0
//! and channels >= 64. Violation causes SILENT data corruption (no error).
//!
//! Failure means: ANE will return garbage values that look plausible but are wrong.
//! This is the most dangerous failure mode — it passes all assertions except
//! numerical comparison against a reference.

use moe_kernels::mla::{conv1x1_spatial_width, absorption_spatial_width, pad16, MlaConfig};

/// Test pad16 at exact boundaries — off-by-one here means silent corruption.
/// Minimum is 64 (ANE requires spatial width ≥ 64).
#[test]
fn pad16_boundaries() {
    // Below 64 → all clamp to 64
    for v in [0, 1, 15, 16, 17, 32, 48, 63] {
        assert_eq!(pad16(v), 64, "pad16({v}) should be 64 (minimum)");
    }
    // Exact 64 and above: standard mod-16 padding
    for m in [64, 128, 256, 512, 1024, 4096, 8192, 16384] {
        assert_eq!(pad16(m), m, "pad16({m}) should be {m}");
    }
    // One over must round up
    for m in [64, 128, 256, 512, 1024, 4096] {
        assert_eq!(pad16(m + 1), m + 16, "pad16({}) should be {}", m + 1, m + 16);
    }
    // One under rounds up
    for m in [80, 128, 256] {
        assert_eq!(pad16(m - 1), m, "pad16({}) should be {}", m - 1, m);
    }
}

/// K2 projection spatial widths — the exact values that will be used in production.
#[test]
fn k2_production_spatial_widths() {
    let cfg = MlaConfig::kimi_k2();
    let q_lora_rank = 1536;
    let q_total = cfg.q_total_dim();   // 12288
    let v_total = cfg.v_total_dim();   // 8192
    let kv_out_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim; // 576

    // seq=1 for decode
    let checks = [
        ("q_a", 1, q_lora_rank, conv1x1_spatial_width(1, q_lora_rank)),
        ("q_b", 1, q_total, conv1x1_spatial_width(1, q_total)),
        ("kv_a", 1, kv_out_dim, conv1x1_spatial_width(1, kv_out_dim)),
        ("o_proj", 1, cfg.hidden_size, conv1x1_spatial_width(1, cfg.hidden_size)),
    ];

    for (name, seq, oc, sp) in checks {
        // Mod-16 check
        assert_eq!(sp % 16, 0, "[{name}] sp={sp} not mod 16");
        // Must fit seq + oc
        assert!(sp >= seq + oc, "[{name}] sp={sp} < seq({seq}) + oc({oc})");
        // Channel check: IC must be >= 64 (enforced at graph build time)
    }
}

/// V3 projection spatial widths — verify they also pass (larger dimensions).
#[test]
fn v3_production_spatial_widths() {
    let cfg = MlaConfig::deepseek_v3();
    let q_lora_rank = 1536;
    let q_total = cfg.q_total_dim();   // 24576
    let v_total = cfg.v_total_dim();   // 16384
    let kv_out_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

    let checks = [
        ("q_a", 1, q_lora_rank),
        ("q_b", 1, q_total),
        ("kv_a", 1, kv_out_dim),
        ("o_proj", 1, cfg.hidden_size),
    ];

    for (name, seq, oc) in checks {
        let sp = conv1x1_spatial_width(seq, oc);
        assert_eq!(sp % 16, 0, "[{name}] sp={sp} not mod 16");
        assert!(sp >= seq + oc, "[{name}] sp={sp} < seq({seq}) + oc({oc})");
    }
}

/// Absorption spatial widths for K2 and V3.
#[test]
fn absorption_alignment() {
    // K2: 64 heads × 128 nope = 8192 + 1 = 8193 → pad to 8208
    let sp_k2 = absorption_spatial_width(64, 128);
    assert_eq!(sp_k2 % 16, 0, "K2 absorption sp={sp_k2} not mod 16");
    assert!(sp_k2 >= 1 + 64 * 128);

    // V3: 128 heads × 128 nope = 16384 + 1 = 16385 → pad to 16400
    let sp_v3 = absorption_spatial_width(128, 128);
    assert_eq!(sp_v3 % 16, 0, "V3 absorption sp={sp_v3} not mod 16");
    assert!(sp_v3 >= 1 + 128 * 128);
}

/// Channel dimension minimums. ANE silently fails with small channel counts.
#[test]
fn channel_minimums_met() {
    let cfg = MlaConfig::kimi_k2();
    let q_lora_rank = 1536;
    let kv_out_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim; // 576

    // All input channel dimensions must be >= 64
    let channel_dims = [
        ("q_a IC (hidden)", cfg.hidden_size),    // 7168
        ("q_b IC (q_lora_rank)", q_lora_rank),   // 1536
        ("kv_a IC (hidden)", cfg.hidden_size),    // 7168
        ("o_proj IC (v_total)", cfg.v_total_dim()), // 8192
        ("absorption IC (kv_rank)", cfg.kv_lora_rank), // 512
    ];

    for (name, dim) in channel_dims {
        assert!(dim >= 64, "[{name}] channels={dim} < 64 — ANE will silently fail");
    }

    // Output channel dimensions
    let out_dims = [
        ("q_a OC (q_lora_rank)", q_lora_rank),  // 1536
        ("q_b OC (q_total)", cfg.q_total_dim()), // 12288
        ("kv_a OC (kv_out_dim)", kv_out_dim),    // 576
        ("o_proj OC (hidden)", cfg.hidden_size),  // 7168
    ];

    for (name, dim) in out_dims {
        assert!(dim >= 64, "[{name}] out_channels={dim} < 64 — ANE will silently fail");
    }
}

/// Verify that prefill seq_len values also produce valid spatial widths.
/// Power-of-2 buckets: 64, 128, 256, 512, 1024, 2048, 4096.
#[test]
fn prefill_bucket_alignment() {
    let oc = 1536; // q_lora_rank (largest projection input)
    for seq in [1, 64, 128, 256, 512, 1024, 2048, 4096] {
        let sp = conv1x1_spatial_width(seq, oc);
        assert_eq!(sp % 16, 0, "seq={seq} oc={oc} → sp={sp} not mod 16");
        assert!(sp >= seq + oc);
    }
}
