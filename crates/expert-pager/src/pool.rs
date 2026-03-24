//! ExpertPool: Least-Stale cache for expert weight buffers.
//!
//! Evicts by minimum `last_used_layer` — experts used at low layers are evicted
//! first because they are furthest from reuse in the next decode step.
//! This is the Least-Stale policy (SpecMD, Apple 2026).

use std::collections::HashMap;

/// Statistics for cache performance monitoring.
#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl PoolStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// Least-Stale entry tracking.
struct Entry {
    /// Slot index in the buffer pool.
    slot: usize,
    /// Layer index where this expert was last used.
    last_used_layer: u32,
}

/// Least-Stale expert weight cache.
///
/// Manages a fixed number of buffer slots. When an expert is requested:
/// - If resident (hit): returns the slot index, updates last_used_layer.
/// - If not resident (miss): evicts the expert with lowest last_used_layer.
pub struct ExpertPool {
    /// Max number of experts that can be resident simultaneously.
    capacity: usize,
    /// Map from (layer_idx, expert_idx) to cache entry.
    entries: HashMap<(u32, u32), Entry>,
    /// Free slot stack.
    free_slots: Vec<usize>,
    /// Performance counters.
    pub stats: PoolStats,
}

impl ExpertPool {
    /// Create a pool with `capacity` slots.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            free_slots: (0..capacity).rev().collect(),
            stats: PoolStats::default(),
        }
    }

    /// Request an expert. Returns (slot_index, is_hit).
    /// If miss, the returned slot is free and ready for loading.
    pub fn request(&mut self, layer: u32, expert_id: u32) -> (usize, bool) {
        let key = (layer, expert_id);

        // Cache hit
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_layer = layer;
            self.stats.hits += 1;
            return (entry.slot, true);
        }

        // Cache miss
        self.stats.misses += 1;

        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            // Evict Least-Stale: minimum last_used_layer
            let (&victim_key, _) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used_layer)
                .expect("pool not empty but no entries");
            let evicted = self.entries.remove(&victim_key).unwrap();
            self.stats.evictions += 1;
            evicted.slot
        };

        self.entries.insert(key, Entry {
            slot,
            last_used_layer: layer,
        });

        (slot, false)
    }

    /// Check if an expert is resident without updating state.
    pub fn is_resident(&self, layer: u32, expert_id: u32) -> bool {
        self.entries.contains_key(&(layer, expert_id))
    }

    /// Maximum capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently resident experts.
    pub fn resident_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the slot index for a resident expert (None if not resident).
    pub fn entries_slot(&self, layer: u32, expert_id: u32) -> Option<usize> {
        self.entries.get(&(layer, expert_id)).map(|e| e.slot)
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_stale_evicts_lowest_layer() {
        let mut pool = ExpertPool::new(2);
        pool.request(3, 10);   // layer 3, expert 10
        pool.request(59, 20);  // layer 59, expert 20
        pool.request(30, 5);   // layer 30, expert 5 — triggers eviction
        // Layer 3 should be evicted (lowest layer = furthest from reuse)
        assert!(!pool.is_resident(3, 10));
        assert!(pool.is_resident(59, 20));  // kept (highest layer)
        assert!(pool.is_resident(30, 5));   // just added
    }

    #[test]
    fn cache_hit_returns_same_slot() {
        let mut pool = ExpertPool::new(4);
        let (slot1, hit1) = pool.request(5, 42);
        assert!(!hit1);
        let (slot2, hit2) = pool.request(5, 42);
        assert!(hit2);
        assert_eq!(slot1, slot2);
    }

    #[test]
    fn stats_tracking() {
        let mut pool = ExpertPool::new(2);
        pool.request(0, 1);  // miss
        pool.request(0, 2);  // miss
        pool.request(0, 1);  // hit
        pool.request(1, 3);  // miss, evicts
        assert_eq!(pool.stats.hits, 1);
        assert_eq!(pool.stats.misses, 3);
        assert_eq!(pool.stats.evictions, 1);
        assert!((pool.stats.hit_rate() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn fill_without_eviction() {
        let mut pool = ExpertPool::new(3);
        pool.request(0, 0);
        pool.request(0, 1);
        pool.request(0, 2);
        assert_eq!(pool.resident_count(), 3);
        assert_eq!(pool.stats.evictions, 0);
    }

    #[test]
    fn different_layers_same_expert_distinct() {
        let mut pool = ExpertPool::new(3);
        pool.request(0, 5);
        pool.request(1, 5);
        assert_eq!(pool.resident_count(), 2);
        assert!(pool.is_resident(0, 5));
        assert!(pool.is_resident(1, 5));
    }
}
