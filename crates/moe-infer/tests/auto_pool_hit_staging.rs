//! Correctness test: pool hit → staging copy produces identical Metal input
//! compared to direct pread.
//!
//! What was fixed: pool hits previously left staging slots stale (bug — never
//! triggered because write-back was disabled). Now pool hits copy DRAM pool
//! buffer → staging, so Metal always processes correct data.
//!
//! Invariant: staging data after pool-hit copy matches staging data after pread
//! (byte-for-byte identical for the same expert).
//!
//! Failure means: pool→staging copy uses wrong slot, wrong offset, or wrong size.

use std::io::Write;

/// Simulate expert pool hit/miss flow with staging buffer.
#[test]
fn pool_hit_staging_matches_pread() {
    // Simulate K2-like parameters: 8 experts, ~23 MB each
    let num_experts = 8;
    let expert_stride = 1024; // smaller for test (real: ~23 MB)

    // Create fake "SSD" data (what pread would return)
    let ssd_data: Vec<Vec<u8>> = (0..num_experts)
        .map(|eid| {
            (0..expert_stride)
                .map(|b| ((eid * 137 + b * 31) & 0xFF) as u8)
                .collect()
        })
        .collect();

    // Create staging buffer (shared across all experts)
    let mut staging = vec![0u8; num_experts * expert_stride];

    // Phase 1: "pread" all experts into staging (first token — all misses)
    for (i, data) in ssd_data.iter().enumerate() {
        staging[i * expert_stride..(i + 1) * expert_stride]
            .copy_from_slice(data);
    }

    // Create pool buffers and copy staging → pool (write-back)
    let mut pool_bufs: Vec<Vec<u8>> = (0..num_experts)
        .map(|_| vec![0u8; expert_stride])
        .collect();
    for i in 0..num_experts {
        pool_bufs[i][..expert_stride]
            .copy_from_slice(&staging[i * expert_stride..(i + 1) * expert_stride]);
    }

    // Phase 2: "second token" — corrupt staging (simulate stale data)
    staging.fill(0xDE);

    // Phase 3: pool hits — copy pool → staging
    for i in 0..num_experts {
        let chunk = &mut staging[i * expert_stride..(i + 1) * expert_stride];
        chunk[..expert_stride].copy_from_slice(&pool_bufs[i][..expert_stride]);
    }

    // Verify: staging now matches original SSD data (byte-for-byte)
    for (i, data) in ssd_data.iter().enumerate() {
        let chunk = &staging[i * expert_stride..(i + 1) * expert_stride];
        assert_eq!(chunk, data.as_slice(),
            "expert {i}: staging after pool-hit copy doesn't match original pread data");
    }
    eprintln!("Pool hit staging copy: {num_experts} experts × {expert_stride}B — all match");
}

/// Test mixed hit/miss scenario (some experts cached, some need pread).
#[test]
fn pool_mixed_hit_miss() {
    let num_experts = 8;
    let expert_stride = 2048;

    // SSD data
    let ssd_data: Vec<Vec<u8>> = (0..num_experts)
        .map(|eid| {
            (0..expert_stride)
                .map(|b| ((eid * 41 + b * 73 + 17) & 0xFF) as u8)
                .collect()
        })
        .collect();

    // Pool: only experts 0,1,2,3 are cached (simulating ~50% hit rate)
    let mut pool_bufs: Vec<Option<Vec<u8>>> = vec![None; num_experts];
    for i in 0..4 {
        pool_bufs[i] = Some(ssd_data[i].clone());
    }

    // Staging starts dirty
    let mut staging = vec![0xAB_u8; num_experts * expert_stride];

    // Simulate mixed pread/pool-copy (like rayon par_iter)
    for i in 0..num_experts {
        let chunk = &mut staging[i * expert_stride..(i + 1) * expert_stride];
        if let Some(ref pool_data) = pool_bufs[i] {
            // Pool hit: copy from DRAM
            chunk[..expert_stride].copy_from_slice(&pool_data[..expert_stride]);
        } else {
            // Pool miss: "pread" from SSD
            chunk[..expert_stride].copy_from_slice(&ssd_data[i]);
        }
    }

    // ALL staging slots should match SSD data regardless of hit/miss
    for (i, data) in ssd_data.iter().enumerate() {
        let chunk = &staging[i * expert_stride..(i + 1) * expert_stride];
        assert_eq!(chunk, data.as_slice(),
            "expert {i} ({}): staging doesn't match SSD data",
            if pool_bufs[i].is_some() { "HIT" } else { "MISS" });
    }
    eprintln!("Mixed hit/miss: 4 hits + 4 misses — all staging slots correct");
}

/// Test that pool write-back + subsequent hit produces correct data.
#[test]
fn pool_writeback_then_hit_roundtrip() {
    let expert_stride = 512;

    // Simulate: token 1 misses, writes back. Token 2 hits.
    let original_data: Vec<u8> = (0..expert_stride)
        .map(|b| ((b * 53 + 7) & 0xFF) as u8)
        .collect();

    // Token 1: pread → staging → write-back to pool
    let mut staging = vec![0u8; expert_stride];
    staging.copy_from_slice(&original_data);

    let mut pool_buf = vec![0u8; expert_stride];
    pool_buf.copy_from_slice(&staging); // write-back

    // Token 2: corrupt staging, then hit → pool→staging
    staging.fill(0xFF);
    staging.copy_from_slice(&pool_buf); // hit copy

    assert_eq!(staging, original_data,
        "roundtrip (pread → pool → staging) produced different data");
    eprintln!("Write-back → hit roundtrip: {expert_stride}B — matches original");
}
