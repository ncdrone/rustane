//! Unit tests for ExpertPool Least-Stale cache.
//!
//! Invariants (all exact, tolerance = 0):
//! 1. Least-Stale eviction: lowest last_used_layer is evicted first
//! 2. Capacity tracking is accurate
//! 3. Hit/miss stats are accurate
//! 4. Requesting a resident expert is a hit

use expert_pager::ExpertPool;

#[test]
fn basic_insert_and_hit() {
    let mut pool = ExpertPool::new(4);

    let (slot0, hit0) = pool.request(5, 0);
    assert!(!hit0, "first request should be a miss");

    let (slot0b, hit0b) = pool.request(5, 0);
    assert!(hit0b, "second request should be a hit");
    assert_eq!(slot0, slot0b, "should return same slot");
}

#[test]
fn capacity_tracking() {
    let mut pool = ExpertPool::new(4);

    for i in 0..4 {
        pool.request(0, i);
    }
    assert_eq!(pool.resident_count(), 4);

    // Adding 5th should evict one
    pool.request(1, 99);
    assert_eq!(pool.resident_count(), 4, "should still be at capacity");
}

#[test]
fn least_stale_eviction_order() {
    let mut pool = ExpertPool::new(3);

    // Insert at different layers
    pool.request(3, 10);   // layer 3
    pool.request(59, 20);  // layer 59
    pool.request(30, 5);   // layer 30 — all slots full, no eviction yet

    // Insert a 4th — should evict layer 3 (lowest layer)
    pool.request(45, 7);
    assert!(!pool.is_resident(3, 10), "layer-3 expert should be evicted (lowest layer)");
    assert!(pool.is_resident(59, 20), "layer-59 expert should remain");
    assert!(pool.is_resident(30, 5), "layer-30 expert should remain");
    assert!(pool.is_resident(45, 7), "newly added should be resident");
}

#[test]
fn stats_accuracy() {
    let mut pool = ExpertPool::new(2);

    pool.request(0, 0); // miss
    pool.request(0, 1); // miss
    pool.request(0, 0); // hit
    pool.request(0, 1); // hit
    pool.request(1, 2); // miss + eviction

    assert_eq!(pool.stats.hits, 2);
    assert_eq!(pool.stats.misses, 3);
    assert_eq!(pool.stats.evictions, 1);

    let hit_rate = pool.stats.hit_rate();
    assert!((hit_rate - 0.4).abs() < 1e-6, "hit rate should be 2/5 = 0.4, got {hit_rate}");
}

#[test]
fn all_slots_unique() {
    let mut pool = ExpertPool::new(8);
    let mut slots = Vec::new();

    for i in 0..8 {
        let (slot, _) = pool.request(0, i);
        assert!(!slots.contains(&slot), "slot {slot} already used");
        slots.push(slot);
    }
}

#[test]
fn evicted_expert_returns_miss_on_reinsert() {
    let mut pool = ExpertPool::new(2);

    pool.request(3, 0);
    pool.request(10, 1);
    pool.request(20, 2); // evicts (3, 0) — lowest layer

    let (_, hit) = pool.request(3, 0); // re-insert
    assert!(!hit, "should be a miss after eviction");
    assert!(pool.is_resident(3, 0));
}

#[test]
fn reset_stats() {
    let mut pool = ExpertPool::new(4);
    pool.request(0, 0);
    pool.request(0, 1);
    pool.request(0, 0);

    pool.reset_stats();
    assert_eq!(pool.stats.hits, 0);
    assert_eq!(pool.stats.misses, 0);
    assert_eq!(pool.stats.evictions, 0);
}
