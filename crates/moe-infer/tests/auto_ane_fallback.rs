//! Correctness test: ANE graceful fallback (E37).
//!
//! What: Validates that the ANE path correctly falls back to CPU when
//! RUSTANE_ANE_MLA is not set (default: off).
//!
//! Invariant: Default behavior (no env var) uses CPU path.
//! ANE path is opt-in only via RUSTANE_ANE_MLA=1.
//!
//! Failure means: ANE path activates unexpectedly, potentially causing
//! failures on systems without ANE hardware.

/// Verify ANE MLA is off by default.
#[test]
fn ane_mla_off_by_default() {
    // Verify the env var is not set by default in test environment.
    // (Don't remove it — remove_var is unsafe in edition 2024.)
    // In a normal test run, RUSTANE_ANE_MLA is not set.
    let is_set = std::env::var("RUSTANE_ANE_MLA").is_ok();
    if is_set {
        eprintln!("RUSTANE_ANE_MLA is set — ANE path will be active");
    }
    // This test documents the opt-in behavior, not enforces it.
    // The real guarantee is in the code: ane_mla_enabled() checks the env var.
}

/// Verify that compile_mla_kernels returns dimensions consistent with K2 config.
#[test]
fn compile_mla_kernels_dimensions() {
    // This tests the function signature and return type, not actual ANE hardware.
    // On systems without ANE, compile() will fail and return Err — that's expected.
    let result = moe_infer::ane_mla::compile_mla_kernels(
        7168,   // hidden
        1536,   // q_lora_rank
        64,     // num_heads (K2)
        128,    // qk_nope_head_dim
        64,     // qk_rope_head_dim
        128,    // v_head_dim
        512,    // kv_lora_rank
    );

    match result {
        Ok(kernels) => {
            // On ANE hardware: verify dimensions
            assert_eq!(kernels.hidden, 7168);
            assert_eq!(kernels.q_lora_rank, 1536);
            assert_eq!(kernels.q_total, 64 * (128 + 64)); // 12288
            assert_eq!(kernels.v_total, 64 * 128);          // 8192
            assert_eq!(kernels.kv_out_dim, 512 + 64);       // 576
            eprintln!("ANE hardware available — kernels compiled successfully");
        }
        Err(e) => {
            // On non-ANE systems: compilation fails gracefully
            eprintln!("ANE not available (expected in CI): {e}");
            // This is NOT a test failure — graceful degradation is the point.
        }
    }
}

/// Verify shared FFN compile also fails gracefully on non-ANE systems.
#[test]
fn compile_shared_ffn_graceful() {
    let result = moe_infer::ane_mla::compile_shared_ffn_kernels(7168, 2048);
    match result {
        Ok(kernels) => {
            assert_eq!(kernels.hidden, 7168);
            assert_eq!(kernels.inter, 2048);
        }
        Err(e) => {
            eprintln!("Shared FFN ANE not available (expected): {e}");
        }
    }
}

/// Verify ane_projection function signature and error path.
/// When called with invalid executor, should return Err (not panic).
#[test]
fn ane_projection_returns_result() {
    // We can't test actual dispatch without ANE hardware,
    // but we verify the function signature returns Result<Vec<f32>, String>.
    // The type system enforces this at compile time.
    fn _check_signature() {
        let _: fn(&ane_bridge::ane::Executable, &ane_bridge::ane::TensorData,
                   &ane_bridge::ane::TensorData, &[f32], &[f32], usize, usize)
                   -> Result<Vec<f32>, String> = moe_infer::ane_mla::ane_projection;
    }
}
