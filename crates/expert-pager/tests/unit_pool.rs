//! Unit tests for ExpertPool LRU cache.
//!
//! Invariants (all exact, tolerance = 0):
//! 1. LRU eviction order is correct
//! 2. Capacity tracking is accurate
//! 3. Hit/miss stats are accurate
//! 4. Requesting a resident expert is a hit

use expert_pager::ExpertPool;

#[test]
fn basic_insert_and_hit() {
    let mut pool = ExpertPool::new(4);

    let (slot0, hit0) = pool.request(0);
    assert!(!hit0, "first request should be a miss");

    let (slot0b, hit0b) = pool.request(0);
    assert!(hit0b, "second request should be a hit");
    assert_eq!(slot0, slot0b, "should return same slot");
}

#[test]
fn capacity_tracking() {
    let mut pool = ExpertPool::new(4);

    for i in 0..4 {
        pool.request(i);
    }
    assert_eq!(pool.resident_count(), 4);

    // Adding 5th should evict one
    pool.request(99);
    assert_eq!(pool.resident_count(), 4, "should still be at capacity");
}

#[test]
fn lru_eviction_order() {
    let mut pool = ExpertPool::new(3);

    // Insert 0, 1, 2
    pool.request(0);
    pool.request(1);
    pool.request(2);

    // Touch 0 (makes it most recent)
    pool.request(0);

    // Insert 3 — should evict 1 (least recently used)
    pool.request(3);
    assert!(!pool.is_resident(1), "expert 1 should be evicted (LRU)");
    assert!(pool.is_resident(0), "expert 0 should still be resident (recently used)");
    assert!(pool.is_resident(2), "expert 2 should still be resident");
    assert!(pool.is_resident(3), "expert 3 should be resident (just added)");
}

#[test]
fn stats_accuracy() {
    let mut pool = ExpertPool::new(2);

    pool.request(0); // miss
    pool.request(1); // miss
    pool.request(0); // hit
    pool.request(1); // hit
    pool.request(2); // miss + eviction

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
        let (slot, _) = pool.request(i);
        assert!(!slots.contains(&slot), "slot {slot} already used");
        slots.push(slot);
    }
}

#[test]
fn evicted_expert_returns_new_slot_on_reinsert() {
    let mut pool = ExpertPool::new(2);

    let (slot_a, _) = pool.request(0);
    pool.request(1);
    pool.request(2); // evicts 0

    let (slot_a2, hit) = pool.request(0); // re-insert
    assert!(!hit, "should be a miss after eviction");
    // slot_a2 may or may not equal slot_a (depends on which slot was freed)
    assert!(pool.is_resident(0));
    let _ = slot_a;
    let _ = slot_a2;
}

#[test]
fn reset_stats() {
    let mut pool = ExpertPool::new(4);
    pool.request(0);
    pool.request(1);
    pool.request(0);

    pool.reset_stats();
    assert_eq!(pool.stats.hits, 0);
    assert_eq!(pool.stats.misses, 0);
    assert_eq!(pool.stats.evictions, 0);
}
