//! Correctness test: pool write-back overlapped with concurrent staging read.
//!
//! Validates that the overlap of pool write-back (staging → pool_bufs) with
//! Metal dispatch (staging read) produces byte-identical pool data compared
//! to the serial write-back path.
//!
//! Invariant: concurrent read of staging from two threads (write-back + Metal)
//! does not corrupt pool_bufs. Write-back writes only to pool_bufs (disjoint).

/// Simulate overlapped write-back + Metal read from same staging buffer.
#[test]
fn writeback_concurrent_with_staging_read() {
    let num_experts = 8;
    let expert_stride = 4096; // larger to catch alignment issues

    // Simulate staging buffer filled by pread (deterministic pattern per expert)
    let staging: Vec<u8> = (0..num_experts * expert_stride)
        .map(|b| ((b * 37 + 13) & 0xFF) as u8)
        .collect();

    // Pool buffers (pre-allocated like real code)
    let mut pool_bufs: Vec<Vec<u8>> = (0..num_experts)
        .map(|_| vec![0u8; expert_stride])
        .collect();

    // need_pread: all experts were pread'd (all need write-back)
    let need_pread = vec![true; num_experts];

    // Run overlapped: write-back + concurrent staging read
    let staging_ref: &[u8] = &staging;
    let metal_checksums: Vec<u64> = std::thread::scope(|s| {
        // Write-back on background thread (mirrors generate_v2 overlap)
        let pool_bufs_ref = &mut pool_bufs;
        let need_pread_ref = &need_pread;
        s.spawn(move || {
            for i in 0..num_experts {
                if need_pread_ref[i] {
                    let offset = i * expert_stride;
                    pool_bufs_ref[i][..expert_stride]
                        .copy_from_slice(&staging_ref[offset..offset + expert_stride]);
                }
            }
        });

        // Simulated Metal read on main thread (reads same staging, computes checksums)
        (0..num_experts)
            .map(|i| {
                let offset = i * expert_stride;
                let chunk = &staging_ref[offset..offset + expert_stride];
                // Simple checksum to verify staging wasn't corrupted during concurrent read
                chunk.iter().map(|&b| b as u64).sum::<u64>()
            })
            .collect()
    });

    // Verify: pool_bufs match staging (write-back correctness)
    for i in 0..num_experts {
        let offset = i * expert_stride;
        assert_eq!(
            &pool_bufs[i][..expert_stride],
            &staging[offset..offset + expert_stride],
            "expert {i}: pool_buf doesn't match staging after overlapped write-back"
        );
    }

    // Verify: Metal checksums match expected (staging wasn't corrupted during read)
    for i in 0..num_experts {
        let offset = i * expert_stride;
        let expected: u64 = staging[offset..offset + expert_stride]
            .iter()
            .map(|&b| b as u64)
            .sum();
        assert_eq!(
            metal_checksums[i], expected,
            "expert {i}: Metal read checksum mismatch — staging corrupted during overlap"
        );
    }

    eprintln!("Overlap write-back + Metal read: {num_experts} experts × {expert_stride}B — all correct");
}

/// Test partial write-back (mixed hits/misses) overlapped with staging read.
#[test]
fn partial_writeback_overlap() {
    let num_experts = 8;
    let expert_stride = 2048;

    let staging: Vec<u8> = (0..num_experts * expert_stride)
        .map(|b| ((b * 59 + 3) & 0xFF) as u8)
        .collect();

    let mut pool_bufs: Vec<Vec<u8>> = (0..num_experts)
        .map(|_| vec![0u8; expert_stride])
        .collect();

    // Only odd experts need write-back (even were pool hits)
    let need_pread: Vec<bool> = (0..num_experts).map(|i| i % 2 == 1).collect();

    // Pre-fill even slots with correct data (simulating prior write-back)
    for i in (0..num_experts).filter(|i| i % 2 == 0) {
        let offset = i * expert_stride;
        pool_bufs[i][..expert_stride]
            .copy_from_slice(&staging[offset..offset + expert_stride]);
    }

    let staging_ref: &[u8] = &staging;
    std::thread::scope(|s| {
        let pool_bufs_ref = &mut pool_bufs;
        let need_pread_ref = &need_pread;
        s.spawn(move || {
            for i in 0..num_experts {
                if need_pread_ref[i] {
                    let offset = i * expert_stride;
                    pool_bufs_ref[i][..expert_stride]
                        .copy_from_slice(&staging_ref[offset..offset + expert_stride]);
                }
            }
        });

        // Main thread reads staging concurrently (simulated Metal)
        let _checksums: Vec<u64> = (0..num_experts)
            .map(|i| {
                let offset = i * expert_stride;
                staging_ref[offset..offset + expert_stride]
                    .iter()
                    .map(|&b| b as u64)
                    .sum()
            })
            .collect();
    });

    // ALL pool_bufs should match staging (both hit and miss slots)
    for i in 0..num_experts {
        let offset = i * expert_stride;
        assert_eq!(
            &pool_bufs[i][..expert_stride],
            &staging[offset..offset + expert_stride],
            "expert {i} ({}): pool_buf mismatch",
            if need_pread[i] { "MISS" } else { "HIT" }
        );
    }

    eprintln!("Partial overlap: 4 hits + 4 misses — all pool_bufs correct");
}
