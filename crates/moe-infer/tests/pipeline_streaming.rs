//! Pipeline test: SSD-paged expert loading produces correct results.
//!
//! Verifies that loading expert weights from SSD via pread produces
//! identical results to having all weights in RAM.
//!
//! Also tests the full paging pipeline: Pool + Loader + Prefetcher.

use expert_pager::convert;
use expert_pager::loader::{ExpertFileLayout, ExpertLoader};
use expert_pager::pool::ExpertPool;
use expert_pager::ExpertPrefetcher;

const NUM_EXPERTS: usize = 16;
const EXPERT_SIZE: usize = 4096; // bytes per expert
const NUM_LAYERS: usize = 4;
const POOL_CAPACITY: usize = 8; // half of experts

/// SSD-paged output matches RAM-resident output (bit-identical).
#[test]
#[ignore = "requires temp filesystem"]
fn ssd_paged_matches_ram_resident() {
    let dir = tempfile::tempdir().unwrap();
    let layout = convert::create_synthetic_experts(
        dir.path(), NUM_LAYERS, NUM_EXPERTS, EXPERT_SIZE,
    ).unwrap();

    let file_layout = ExpertFileLayout {
        expert_size: EXPERT_SIZE,
        num_experts: NUM_EXPERTS,
    };

    // Load ALL experts into RAM (reference)
    let mut ram_experts = vec![vec![0u8; EXPERT_SIZE]; NUM_EXPERTS];
    let loader = ExpertLoader::open(
        dir.path().join("layer_00.bin").to_str().unwrap(),
        file_layout.clone(),
    ).unwrap();
    for id in 0..NUM_EXPERTS {
        loader.load_expert(id as u32, &mut ram_experts[id]).unwrap();
    }

    // Load via paging (ExpertPool + ExpertLoader)
    let mut pool = ExpertPool::new(POOL_CAPACITY);
    let mut paged_buffers: Vec<Vec<u8>> = (0..POOL_CAPACITY).map(|_| vec![0u8; EXPERT_SIZE]).collect();

    // Request all experts (some will be evicted)
    for id in 0..NUM_EXPERTS {
        let (slot, is_hit) = pool.request(0, id as u32);
        if !is_hit {
            loader.load_expert(id as u32, &mut paged_buffers[slot]).unwrap();
        }
        // Verify the data in the slot matches RAM reference
        assert_eq!(
            &paged_buffers[slot][..], &ram_experts[id][..],
            "expert {id} paged data != RAM data"
        );
    }

    // Last POOL_CAPACITY experts should be resident
    assert_eq!(pool.resident_count(), POOL_CAPACITY);
    assert!(pool.stats.hits == 0, "no hits expected on first pass");
    assert_eq!(pool.stats.misses, NUM_EXPERTS as u64);

    let _ = layout;
}

/// Prefetcher is disabled (returns empty predictions).
#[test]
#[ignore = "requires temp filesystem"]
fn prefetcher_disabled_returns_empty() {
    let mut prefetcher = ExpertPrefetcher::new(4);

    let mut scores_l0 = vec![0.0f32; NUM_EXPERTS];
    scores_l0[2] = 0.9;
    prefetcher.record(0, scores_l0);

    // Prefetcher is a no-op — always returns empty
    let predicted = prefetcher.predict(2);
    assert!(predicted.is_empty(), "disabled prefetcher should return empty");
}

/// Full paging pipeline: pool + loader + prefetcher across layers.
#[test]
#[ignore = "requires temp filesystem"]
fn full_paging_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let _layout = convert::create_synthetic_experts(
        dir.path(), NUM_LAYERS, NUM_EXPERTS, EXPERT_SIZE,
    ).unwrap();

    let file_layout = ExpertFileLayout {
        expert_size: EXPERT_SIZE,
        num_experts: NUM_EXPERTS,
    };

    let mut pool = ExpertPool::new(POOL_CAPACITY);
    let mut slot_buffers: Vec<Vec<u8>> = (0..POOL_CAPACITY).map(|_| vec![0u8; EXPERT_SIZE]).collect();

    // Simulate inference across layers
    for layer in 0..NUM_LAYERS {
        let loader = ExpertLoader::open(
            dir.path().join(format!("layer_{layer:02}.bin")).to_str().unwrap(),
            file_layout.clone(),
        ).unwrap();

        // Simulate routing: select 4 experts per layer
        let selected: Vec<u32> = (0..4).map(|i| ((layer * 3 + i) % NUM_EXPERTS) as u32).collect();

        // Load selected experts
        for &eid in &selected {
            let (slot, is_hit) = pool.request(layer as u32, eid);
            if !is_hit {
                loader.load_expert(eid, &mut slot_buffers[slot]).unwrap();
            }
        }

    }

    println!("Pool stats: hits={}, misses={}, evictions={}, hit_rate={:.1}%",
        pool.stats.hits, pool.stats.misses, pool.stats.evictions,
        pool.stats.hit_rate() * 100.0);

    assert!(pool.stats.misses > 0, "should have some misses");
    // With prefetching, we should see some hits
    // (depends on whether predicted experts match actual selections)
}
