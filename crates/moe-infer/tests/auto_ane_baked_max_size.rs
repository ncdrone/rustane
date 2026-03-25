//! Find maximum baked weight size that ANE can compile.

use std::time::Instant;
use moe_kernels::mla::{build_mla_conv1x1_baked, build_mla_matmul_baked, pad16};
use objc2_foundation::NSQualityOfService;

#[test]
#[ignore = "requires ANE hardware"]
fn find_max_baked_weight_size() {
    eprintln!("\n=== Maximum Baked Weight Size ===\n");
    let qos = NSQualityOfService::UserInteractive;

    // Test increasing sizes
    let sizes = [
        (64, 64),       // 8 KB
        (128, 128),     // 32 KB
        (256, 256),     // 128 KB
        (512, 512),     // 512 KB
        (1024, 512),    // 1 MB
        (1024, 1024),   // 2 MB
        (2048, 1024),   // 4 MB
        (2048, 2048),   // 8 MB
        (4096, 1024),   // 8 MB
        (4096, 2048),   // 16 MB
        (4096, 4096),   // 32 MB
        (7168, 512),    // 7 MB (KV compress size)
        (7168, 1024),   // 14 MB
        (7168, 1536),   // 22 MB (Q LoRA compress size)
        (7168, 2048),   // 29 MB (shared FFN gate size)
    ];

    eprintln!("{:<20} {:>10} {:>12} {:>10}", "Dims", "Weight MB", "Compile", "Time ms");
    eprintln!("{}", "-".repeat(55));

    for (ic, oc) in sizes {
        let weights: Vec<f32> = (0..oc * ic).map(|i| (i % 100) as f32 / 100.0).collect();
        let weight_mb = (ic * oc * 2) as f64 / 1e6; // fp16 bytes

        let graph = build_mla_conv1x1_baked(ic, oc, 1, &weights);
        let t = Instant::now();
        match graph.compile(qos) {
            Ok(_) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!("{:<20} {:>9.1}  {:>11} {:>9.1}", format!("[{ic}→{oc}]"), weight_mb, "OK", ms);
            }
            Err(_) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!("{:<20} {:>9.1}  {:>11} {:>9.1}", format!("[{ic}→{oc}]"), weight_mb, "FAILED", ms);
            }
        }
    }

    // Now test matmul path (may handle asymmetric dims)
    eprintln!("\n=== Matmul Baked Weights (asymmetric workaround) ===\n");
    eprintln!("{:<20} {:>10} {:>12} {:>10}", "Dims", "Weight MB", "Compile", "Time ms");
    eprintln!("{}", "-".repeat(55));

    let asym_sizes = [
        (1024, 512),    // 1 MB (conv FAILED)
        (2048, 1024),   // 4 MB (conv FAILED)
        (7168, 512),    // 7 MB (KV compress, conv FAILED)
        (7168, 1536),   // 22 MB (Q LoRA compress, conv FAILED)
        (1536, 12288),  // 37 MB (Q LoRA expand)
        (8192, 7168),   // 117 MB (O projection)
        (7168, 2048),   // 29 MB (shared gate)
    ];

    for (ic, oc) in asym_sizes {
        let weights: Vec<f32> = (0..oc * ic).map(|i| (i % 100) as f32 / 100.0).collect();
        let weight_mb = (ic * oc * 2) as f64 / 1e6;

        let graph = build_mla_matmul_baked(ic, oc, 1, &weights);
        let t = Instant::now();
        match graph.compile(qos) {
            Ok(_) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!("{:<20} {:>9.1}  {:>11} {:>9.1}", format!("[{ic}→{oc}]"), weight_mb, "OK", ms);
            }
            Err(_) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!("{:<20} {:>9.1}  {:>11} {:>9.1}", format!("[{ic}→{oc}]"), weight_mb, "FAILED", ms);
            }
        }
    }
}
