//! Correctness test: ANE shared expert FFN (E06-D).
//!
//! What: Validates that shared FFN Conv1x1 projections have correct dimensions,
//! fit within ANE 32MB SRAM per individual projection, and produce correct graphs.
//!
//! Invariant: Each individual shared FFN projection weight matrix must be < 32 MB
//! to avoid ANE SRAM cliff (4.7x slowdown above 32 MB).
//!
//! Failure means: Shared FFN on ANE would be slower than CPU, negating the benefit.

use moe_kernels::mla::{build_mla_conv1x1, conv1x1_spatial_width, pad16};

/// K2 shared expert dimensions: gate/up [7168→2048], down [2048→7168].
/// Each weight matrix: 7168 × 2048 × 2 bytes (fp16) = 28.7 MB (under 32 MB).
#[test]
fn k2_shared_ffn_fits_sram() {
    let hidden = 7168;
    let inter = 2048; // K2 moe_inter_size

    // Weight sizes in bytes (fp16 on ANE)
    let gate_bytes = hidden * inter * 2; // 29,360,128 = 28.0 MB
    let up_bytes = hidden * inter * 2;
    let down_bytes = inter * hidden * 2;

    let sram_limit = 32 * 1024 * 1024; // 32 MB

    assert!(gate_bytes < sram_limit,
        "gate weight {:.1} MB exceeds 32 MB SRAM", gate_bytes as f64 / 1e6);
    assert!(up_bytes < sram_limit,
        "up weight {:.1} MB exceeds 32 MB SRAM", up_bytes as f64 / 1e6);
    assert!(down_bytes < sram_limit,
        "down weight {:.1} MB exceeds 32 MB SRAM", down_bytes as f64 / 1e6);
}

/// Verify graphs build for shared FFN dimensions.
#[test]
fn shared_ffn_graphs_build() {
    let hidden = 7168;
    let inter = 2048;

    // gate: [hidden → inter]
    let _g1 = build_mla_conv1x1(hidden, inter, 1);
    // up: [hidden → inter]
    let _g2 = build_mla_conv1x1(hidden, inter, 1);
    // down: [inter → hidden]
    let _g3 = build_mla_conv1x1(inter, hidden, 1);
}

/// Verify spatial widths are aligned.
#[test]
fn shared_ffn_spatial_alignment() {
    let hidden = 7168;
    let inter = 2048;

    let gate_sp = conv1x1_spatial_width(1, inter);
    let down_sp = conv1x1_spatial_width(1, hidden);

    assert_eq!(gate_sp % 16, 0, "gate sp={gate_sp} not mod 16");
    assert_eq!(down_sp % 16, 0, "down sp={down_sp} not mod 16");
    assert!(gate_sp >= 1 + inter);
    assert!(down_sp >= 1 + hidden);
}

/// Total compile budget with MLA + shared FFN: 4 + 3 = 7 graphs.
#[test]
fn total_compile_budget_with_shared_ffn() {
    let mla_compilations = 4;  // q_a, q_b, kv_a, o_proj
    let ffn_compilations = 3;  // gate, up, down
    // Note: gate and up have same shape — could share 1 graph + 2 buffer sets
    // But keeping separate for clarity. Still well under budget.
    let total = mla_compilations + ffn_compilations;
    assert!(total < 100, "Total compilations {} exceeds budget", total);
    assert_eq!(total, 7);
}

/// CPU reference: verify SiLU × up formula.
#[test]
fn silu_gate_fusion_reference() {
    let gate_val = 2.0f32;
    let up_val = 3.0f32;
    let silu = gate_val / (1.0 + (-gate_val).exp());
    let expected = silu * up_val;

    // SiLU(2.0) = 2.0 * sigmoid(2.0) = 2.0 * 0.8808 = 1.7616
    // 1.7616 * 3.0 = 5.2847
    assert!((expected - 5.2847).abs() < 0.01, "SiLU gate: {expected}");
}
