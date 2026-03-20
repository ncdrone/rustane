//! ExpertPool: LRU cache for expert weight buffers.
//!
//! Tracks which experts are resident in memory, evicts least-recently-used
//! when capacity is exceeded, and provides hit/miss statistics.

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

/// LRU entry tracking.
struct Entry {
    /// Slot index in the ring buffer.
    slot: usize,
    /// LRU timestamp (higher = more recent).
    last_used: u64,
}

/// LRU-based expert weight cache.
///
/// Manages a fixed number of buffer slots. When an expert is requested:
/// - If resident (hit): returns the slot index, updates LRU.
/// - If not resident (miss): evicts LRU entry, returns the freed slot for loading.
pub struct ExpertPool {
    /// Max number of experts that can be resident simultaneously.
    capacity: usize,
    /// Map from expert_id to cache entry.
    entries: HashMap<u32, Entry>,
    /// LRU clock (monotonically increasing).
    clock: u64,
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
            clock: 0,
            free_slots: (0..capacity).rev().collect(),
            stats: PoolStats::default(),
        }
    }

    /// Request an expert. Returns (slot_index, is_hit).
    /// If miss, the returned slot is free and ready for loading.
    pub fn request(&mut self, expert_id: u32) -> (usize, bool) {
        self.clock += 1;

        // Cache hit
        if let Some(entry) = self.entries.get_mut(&expert_id) {
            entry.last_used = self.clock;
            self.stats.hits += 1;
            return (entry.slot, true);
        }

        // Cache miss
        self.stats.misses += 1;

        let slot = if let Some(slot) = self.free_slots.pop() {
            // Free slot available
            slot
        } else {
            // Evict LRU
            let (&lru_id, _) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .expect("pool not empty but no entries");
            let evicted = self.entries.remove(&lru_id).unwrap();
            self.stats.evictions += 1;
            evicted.slot
        };

        self.entries.insert(expert_id, Entry {
            slot,
            last_used: self.clock,
        });

        (slot, false)
    }

    /// Check if an expert is resident without updating LRU.
    pub fn is_resident(&self, expert_id: u32) -> bool {
        self.entries.contains_key(&expert_id)
    }

    /// Number of currently resident experts.
    pub fn resident_count(&self) -> usize {
        self.entries.len()
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }
}
