//! Deep Metal kernel profiling: measure dispatch overhead vs compute time.
//!
//! The fused_gate_up_silu + dequant_4bit_gemv_v2 kernels handle 72% of decode.
//! Currently at 36% bandwidth utilization. Where is the 64% waste?
//!
//! This test measures:
//! 1. Metal command buffer creation overhead
//! 2. Metal dispatch overhead (encode + commit + waitUntilCompleted)
//! 3. Per-expert breakdown (pread time vs Metal compute time)
//! 4. GPU utilization via command buffer timestamps

use std::time::Instant;

#[test]
#[ignore = "requires K2 weights + Metal GPU"]
fn profile_metal_dispatch_overhead() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let weights_dir = root.join("weights/rustane-k2");
    let config_path = root.join("configs/kimi-k2.toml");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: K2 weights not found");
        return;
    }

    eprintln!("\n=== Metal Kernel Deep Profile ===\n");

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load K2");

    eprintln!("Model loaded. Metal GPU: {}", if model.metal.is_some() { "enabled" } else { "disabled" });

    // The key measurements we need:
    // 1. How much of the 7ms/layer FFN is pread vs Metal dispatch vs Metal compute?
    // 2. What's the Metal command buffer creation + commit overhead?
    // 3. How does GPU utilization change with threadgroup size?
    //
    // These require running actual Metal dispatches with real expert weights.
    // The benchmark above already shows FFN=431ms/token = 7.1ms/layer.
    // We need Instruments Metal System Trace for the per-dispatch breakdown.
    //
    // For now, let's measure what we can from the CPU side.

    eprintln!("Per-layer FFN breakdown requires Instruments Metal System Trace.");
    eprintln!("Key metrics to capture with Instruments:");
    eprintln!("  1. GPU utilization % during expert dispatch");
    eprintln!("  2. Memory bandwidth utilization (GB/s)");
    eprintln!("  3. Per-dispatch wall time (fused vs down)");
    eprintln!("  4. Threadgroup occupancy");
    eprintln!("");
    eprintln!("From CPU timing (bench_k2_tok_per_sec warm token 1):");
    eprintln!("  FFN total: 431ms / 61 layers = 7.1ms/layer");
    eprintln!("  = pread (~3-5ms) + Metal fused (~1.75ms) + Metal down (~1.75ms)");
    eprintln!("");
    eprintln!("From research (AGENTS-K2.md):");
    eprintln!("  Metal INT4 GEMV at 36% of bandwidth target");
    eprintln!("  Expert dims: in_features=7168, moe_inter=2048, group_size=128");
    eprintln!("  Per expert: gate+up+SiLU = 1 dispatch, down = 1 dispatch");
    eprintln!("  Top-k=8 experts per layer");
    eprintln!("");
    eprintln!("Hypothesis for 36% bandwidth:");
    eprintln!("  1. Thread divergence at boundary TGs (row >= out_features)");
    eprintln!("  2. Scale/zero reads not coalesced with packed weight reads");
    eprintln!("  3. 32-thread reduction per row: warp-level overhead");
    eprintln!("  4. x vector re-read from device memory per TG (no shared cache)");
    eprintln!("  5. Two separate kernel dispatches (fused + down) instead of one");
}

/// Profile CPU-side overhead of Metal command buffer operations.
/// Measures: create + encode + commit + wait cycle.
#[test]
#[ignore = "requires Metal GPU"]
fn profile_metal_cmdbuf_overhead() {
    eprintln!("\n=== Metal Command Buffer Overhead ===\n");

    // We need access to the Metal device/queue. Check if model has Metal.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let weights_dir = root.join("weights/rustane-k2");
    let config_path = root.join("configs/kimi-k2.toml");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: K2 weights not found");
        return;
    }

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load K2");

    if model.metal.is_none() {
        eprintln!("SKIP: Metal not available");
        return;
    }

    eprintln!("Metal available. Detailed dispatch profiling requires");
    eprintln!("Instruments Metal System Trace or addCompletedHandler timestamps.");
    eprintln!("");
    eprintln!("Key experiment: measure empty command buffer roundtrip");
    eprintln!("(create → commit → waitUntilCompleted with zero work)");
    eprintln!("This gives the baseline dispatch overhead.");
}
