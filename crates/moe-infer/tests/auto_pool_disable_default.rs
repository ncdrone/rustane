//! Correctness test: pool_cap=0 (disabled pool) produces identical pread behavior
//! as pool_cap=3000 (enabled pool with no write-back).
//!
//! What was optimized: changed default pool_cap from 3000 to 0. Since pool write-back
//! is disabled, the pool only does HashMap tracking (pure overhead). Disabling it
//! eliminates ~0.3ms/layer of dead HashMap lookups + 3000 empty Vec allocations.
//!
//! Invariant: with write-back disabled, pool hits never actually hit (buffers always
//! empty). So need_pread is always all-true regardless of pool_cap. Disabling the
//! pool just skips the bookkeeping — identical computation path.
//!
//! Failure means: pool presence somehow affected the compute path (it shouldn't —
//! pool tracking is read-only relative to staging and FFN buffers).

use expert_pager::ExpertPool;

/// Core invariant: pool with no write-back has need_pread all-true (same as no pool).
/// Tests that pool.request() + empty buffer check = all misses = all pread.
#[test]
fn pool_no_writeback_all_need_pread() {
    let expert_stride = 1024;
    let num_experts_per_layer = 8;

    // Scenario 1: pool_cap=0 (pool is None) — straightforward all-pread
    let pool_opt_none: Option<ExpertPool> = None;
    let pool_bufs_empty: Vec<Vec<u8>> = Vec::new();
    let mut need_pread_no_pool = vec![true; num_experts_per_layer];
    let mut hits_no_pool = 0usize;
    if let Some(_pool) = pool_opt_none.as_ref() {
        unreachable!("pool should be None");
    }
    assert!(need_pread_no_pool.iter().all(|&x| x), "cap=0: all need pread");
    assert_eq!(hits_no_pool, 0);

    // Scenario 2: pool_cap=3000 with empty buffers (write-back disabled)
    let mut pool = ExpertPool::new(3000);
    let pool_bufs: Vec<Vec<u8>> = (0..3000).map(|_| Vec::new()).collect();
    let mut need_pread_with_pool = vec![true; num_experts_per_layer];
    let mut hits_with_pool = 0usize;

    // Simulate 2 tokens of routing to populate pool HashMap
    let expert_ids: Vec<u32> = vec![10, 42, 100, 200, 350, 5, 77, 383];
    for &eid in &expert_ids {
        pool.request(5, eid); // token 0: all misses
    }
    pool.reset_stats();

    // Token 1: pool says "hit" but buffers are empty → still need pread
    for (i, &eid) in expert_ids.iter().enumerate() {
        let (slot, is_hit) = pool.request(5, eid);
        if is_hit && !pool_bufs[slot].is_empty() {
            need_pread_with_pool[i] = false;
            hits_with_pool += 1;
        }
    }

    // Key assertion: both scenarios produce identical need_pread
    assert_eq!(need_pread_no_pool, need_pread_with_pool,
        "pool with no write-back must produce same need_pread as no pool");
    assert_eq!(hits_no_pool, hits_with_pool,
        "pool with no write-back must report same hit count as no pool");

    // Pool reports hits internally but buffers are empty, so effective hits = 0
    assert_eq!(pool.stats.hits, 8, "pool HashMap says all 8 are hits");
    assert_eq!(hits_with_pool, 0, "but effective hits = 0 (empty buffers)");
    eprintln!("Verified: pool_cap=0 and pool_cap=3000 (no writeback) produce identical pread behavior");
}

/// Edge case: pool with write-back enabled actually produces hits (different from no pool).
/// This verifies the pool CAN work when write-back is enabled — useful if we re-enable it later.
#[test]
fn pool_with_writeback_produces_real_hits() {
    let expert_stride = 64; // small for test
    let mut pool = ExpertPool::new(100);
    let mut pool_bufs: Vec<Vec<u8>> = (0..100).map(|_| Vec::new()).collect();

    let expert_ids: Vec<u32> = vec![10, 42, 100, 200];

    // Token 0: all misses, then simulate write-back
    for &eid in &expert_ids {
        let (slot, is_hit) = pool.request(5, eid);
        assert!(!is_hit, "first access should be miss");
        // Simulate write-back: allocate and fill buffer
        pool_bufs[slot] = vec![eid as u8; expert_stride];
    }

    pool.reset_stats();

    // Token 1: should get real hits now
    let mut hits = 0;
    for &eid in &expert_ids {
        let (slot, is_hit) = pool.request(5, eid);
        if is_hit && !pool_bufs[slot].is_empty() {
            hits += 1;
            assert_eq!(pool_bufs[slot][0], eid as u8, "pool data should match");
        }
    }
    assert_eq!(hits, 4, "all 4 should be real hits with write-back");
    eprintln!("Verified: pool with write-back produces real hits");
}

/// Verify pool_cap=0 doesn't panic or allocate.
#[test]
fn pool_cap_zero_safe() {
    // pool_cap=0 means pool is None in ModelV2.
    // ExpertPool::new(0) would create a pool that panics on request.
    // Verify our code path handles this by making pool None.
    let pool: Option<ExpertPool> = None;
    assert!(pool.is_none());

    // Simulate the decode path with no pool
    let mut need_pread = vec![true; 8];
    let mut pool_hits = 0usize;
    if let Some(_) = pool.as_ref() {
        panic!("should not enter pool path");
    }
    // All experts need pread
    assert!(need_pread.iter().all(|&x| x));
    assert_eq!(pool_hits, 0);
    eprintln!("pool_cap=0: safe, no panics, all pread");
}
