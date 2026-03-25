//! Experiment: ANE compile cache reuse (Hopper P0).
//!
//! Tests whether compiling the same MIL graph twice results in a cache hit
//! (instant second compile). If yes: per-layer baked weights become viable
//! because the 119 compile limit is per-session, not permanent.
//!
//! From whiskeypolitik's Hopper research: Apple has _ANEModelCacheManager
//! with compiledModelExistsMatchingHash. Rustane may be missing cache hits
//! by compiling from fresh temp paths every run.

use std::time::Instant;
use moe_kernels::mla::{build_mla_conv1x1, MlaConfig};
use objc2_foundation::NSQualityOfService;

#[test]
#[ignore = "requires ANE hardware"]
fn ane_compile_cache_test() {
    eprintln!("\n=== ANE Compile Cache Reuse Test ===\n");

    let qos = NSQualityOfService::UserInteractive;
    let cfg = MlaConfig::kimi_k2();

    // Test 1: Compile same graph shape twice
    eprintln!("Test 1: Same graph shape, same process");
    let g1 = build_mla_conv1x1(7168, 1536, 1);
    let t = Instant::now();
    let e1 = g1.compile(qos).expect("first compile");
    let first_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  First compile:  {:.1}ms", first_ms);

    let g2 = build_mla_conv1x1(7168, 1536, 1);
    let t = Instant::now();
    let e2 = g2.compile(qos).expect("second compile");
    let second_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Second compile: {:.1}ms", second_ms);

    let speedup = first_ms / second_ms;
    eprintln!("  Speedup: {:.1}x", speedup);
    if speedup > 3.0 {
        eprintln!("  → CACHE HIT: Second compile {:.0}x faster!", speedup);
    } else {
        eprintln!("  → NO CACHE: Both compiles take similar time");
    }

    // Test 2: Compile all 4 K2 projection graphs
    eprintln!("\nTest 2: All 4 K2 projection graphs");
    let shapes = [
        ("q_a [7168→1536]", 7168, 1536),
        ("q_b [1536→12288]", 1536, 12288),
        ("kv_a [7168→576]", 7168, 576),
        ("o_proj [8192→7168]", 8192, 7168),
    ];

    let mut total_first = 0.0;
    let mut total_second = 0.0;

    for (name, ic, oc) in shapes {
        let g = build_mla_conv1x1(ic, oc, 1);
        let t = Instant::now();
        let _ = g.compile(qos).expect("compile");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total_first += ms;
        eprintln!("  {name}: {:.1}ms", ms);
    }

    eprintln!("  Total first pass: {:.1}ms", total_first);

    // Second pass (same shapes)
    for (name, ic, oc) in shapes {
        let g = build_mla_conv1x1(ic, oc, 1);
        let t = Instant::now();
        let _ = g.compile(qos).expect("compile");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total_second += ms;
        eprintln!("  {name} (2nd): {:.1}ms", ms);
    }

    eprintln!("  Total second pass: {:.1}ms", total_second);
    eprintln!("  Overall speedup: {:.1}x", total_first / total_second);

    // Test 3: How many compiles before failure?
    eprintln!("\nTest 3: Compile count stress test (target: find the ceiling)");
    let mut success_count = 0;
    for i in 0..150 {
        let g = build_mla_conv1x1(128, 64, 1); // small graph
        match g.compile(qos) {
            Ok(_) => success_count += 1,
            Err(e) => {
                eprintln!("  COMPILE FAILED at attempt {}: {}", i + 1, e);
                break;
            }
        }
        if (i + 1) % 25 == 0 {
            eprintln!("  {} compiles OK...", i + 1);
        }
    }
    eprintln!("  Total successful compiles: {}/150", success_count);
    if success_count >= 119 {
        eprintln!("  → Passed 119 limit — cache may be clearing slots!");
    }
}
