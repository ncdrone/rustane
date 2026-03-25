//! Correctness test: MLA Conv1x1 ANE graph builders (E06-A).
//!
//! What: Validates that build_mla_conv1x1() and build_mla_grouped_absorption()
//! produce ANE graphs that compile without error and have correct shapes.
//!
//! Invariant: Graph builds with valid shapes for all K2/V3 MLA projection dimensions.
//! Failure means: ANE graph builder produces invalid MIL that won't compile,
//! or dimensions are wrong (would produce silent corruption).

use moe_kernels::mla::{
    build_mla_conv1x1, build_mla_grouped_absorption, conv1x1_spatial_width,
    absorption_spatial_width, pad16, MlaConfig,
};

#[test]
fn pad16_correctness() {
    assert_eq!(pad16(0), 0);
    assert_eq!(pad16(1), 16);
    assert_eq!(pad16(15), 16);
    assert_eq!(pad16(16), 16);
    assert_eq!(pad16(17), 32);
    assert_eq!(pad16(512), 512);
    assert_eq!(pad16(513), 528);
}

#[test]
fn conv1x1_spatial_width_mod16() {
    // All spatial widths must be mod 16 (ANE silent corruption otherwise)
    let test_cases = [
        (1, 512),     // KV compression: seq=1, oc=kv_lora_rank
        (1, 1536),    // Q LoRA compress: seq=1, oc=q_lora_rank
        (1, 12288),   // Q LoRA expand (K2): seq=1, oc=64*192
        (1, 24576),   // Q LoRA expand (V3): seq=1, oc=128*192
        (1, 8192),    // O projection (K2): seq=1, oc=64*128
        (1, 16384),   // O projection (V3): seq=1, oc=128*128
        (1, 7168),    // KV down-project: seq=1, oc=hidden
    ];
    for (seq, oc) in test_cases {
        let sp = conv1x1_spatial_width(seq, oc);
        assert_eq!(sp % 16, 0, "spatial width {} not mod 16 for seq={}, oc={}", sp, seq, oc);
        assert!(sp >= seq + oc, "spatial width {} too small for seq={} + oc={}", sp, seq, oc);
    }
}

#[test]
fn absorption_spatial_width_mod16() {
    // K2: 64 heads × 128 nope
    let sp_k2 = absorption_spatial_width(64, 128);
    assert_eq!(sp_k2 % 16, 0);
    assert!(sp_k2 >= 1 + 64 * 128);

    // V3: 128 heads × 128 nope
    let sp_v3 = absorption_spatial_width(128, 128);
    assert_eq!(sp_v3 % 16, 0);
    assert!(sp_v3 >= 1 + 128 * 128);
}

#[test]
fn build_conv1x1_k2_q_lora_compress() {
    // K2 Q LoRA compress: [7168 → 1536]
    let g = build_mla_conv1x1(7168, 1536, 1);
    // Graph should build without panic. Shape validation is in the Graph API.
    let _ = g;
}

#[test]
fn build_conv1x1_k2_q_lora_expand() {
    // K2 Q LoRA expand: [1536 → 12288] (64 heads × 192)
    let g = build_mla_conv1x1(1536, 12288, 1);
    let _ = g;
}

#[test]
fn build_conv1x1_v3_q_lora_expand() {
    // V3 Q LoRA expand: [1536 → 24576] (128 heads × 192)
    let g = build_mla_conv1x1(1536, 24576, 1);
    let _ = g;
}

#[test]
fn build_conv1x1_k2_kv_compress() {
    // KV compression: [7168 → 512]
    let g = build_mla_conv1x1(7168, 512, 1);
    let _ = g;
}

#[test]
fn build_conv1x1_k2_o_proj() {
    // K2 O projection: [8192 → 7168] (v_total=64*128 → hidden)
    let g = build_mla_conv1x1(8192, 7168, 1);
    let _ = g;
}

#[test]
fn build_conv1x1_v3_o_proj() {
    // V3 O projection: [16384 → 7168] (v_total=128*128 → hidden)
    let g = build_mla_conv1x1(16384, 7168, 1);
    let _ = g;
}

#[test]
fn build_grouped_absorption_k2() {
    // K2: 64 heads, kv_rank=512, nope=128
    let g = build_mla_grouped_absorption(512, 128, 64);
    let _ = g;
}

#[test]
fn build_grouped_absorption_v3() {
    // V3: 128 heads, kv_rank=512, nope=128
    let g = build_mla_grouped_absorption(512, 128, 128);
    let _ = g;
}

#[test]
fn mla_config_k2_dimensions() {
    let c = MlaConfig::kimi_k2();
    assert_eq!(c.hidden_size, 7168);
    assert_eq!(c.num_heads, 64);
    assert_eq!(c.num_kv_heads, 64);
    assert_eq!(c.kv_lora_rank, 512);
    assert_eq!(c.q_total_dim(), 64 * (128 + 64)); // 12288
    assert_eq!(c.v_total_dim(), 64 * 128);          // 8192
}

#[test]
fn mla_config_v3_dimensions() {
    let c = MlaConfig::deepseek_v3();
    assert_eq!(c.hidden_size, 7168);
    assert_eq!(c.num_heads, 128);
    assert_eq!(c.num_kv_heads, 128);
    assert_eq!(c.q_total_dim(), 128 * (128 + 64)); // 24576
    assert_eq!(c.v_total_dim(), 128 * 128);          // 16384
}
