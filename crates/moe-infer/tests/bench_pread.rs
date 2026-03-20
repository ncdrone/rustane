//! Benchmark: SSD pread throughput at 1/4/8 threads.
//!
//! Target: >5 GB/s on M4 Max NVMe SSD.
//!
//! Run with: cargo test -p moe-infer --test bench_pread --release -- --ignored --nocapture

use expert_pager::loader::{ExpertFileLayout, ExpertLoader};
use std::io::Write;
use std::time::Instant;

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_pread_throughput() {
    let expert_size = 2 * 1024 * 1024; // 2 MB per expert (typical 4-bit expert)
    let num_experts = 128;
    let total_bytes = expert_size * num_experts;

    // Create test file (~256 MB)
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    let chunk = vec![0xABu8; expert_size];
    for _ in 0..num_experts {
        file.write_all(&chunk).expect("write");
    }
    file.flush().expect("flush");

    let layout = ExpertFileLayout { expert_size, num_experts };
    let loader = ExpertLoader::open(file.path().to_str().unwrap(), layout).unwrap();

    println!("\n{}", "=".repeat(60));
    println!("  SSD pread Throughput Benchmark");
    println!("  {} experts x {} MB = {} MB total",
        num_experts, expert_size / (1024 * 1024), total_bytes / (1024 * 1024));
    println!("{}", "=".repeat(60));
    println!("  {:>8} {:>10} {:>10}", "Threads", "Time (ms)", "GB/s");
    println!("{}", "-".repeat(60));

    for &num_threads in &[1, 2, 4, 8] {
        let expert_ids: Vec<u32> = (0..num_experts as u32).collect();
        let mut buffers: Vec<Vec<u8>> = vec![Vec::new(); num_experts];

        // Warmup
        loader.load_experts_parallel(&expert_ids, &mut buffers, num_threads).unwrap();

        // Timed run
        let t0 = Instant::now();
        loader.load_experts_parallel(&expert_ids, &mut buffers, num_threads).unwrap();
        let elapsed = t0.elapsed();

        let gb_per_sec = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
        println!(
            "  {:>8} {:>10.1} {:>10.2}",
            num_threads,
            elapsed.as_millis(),
            gb_per_sec,
        );
    }

    println!("{}", "-".repeat(60));
    println!("  Target: >5 GB/s");
    println!();
}
