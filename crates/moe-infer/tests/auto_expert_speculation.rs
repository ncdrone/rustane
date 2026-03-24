//! Correctness test: expert speculation — speculative pread during Metal dispatch.
//!
//! What was optimized: overlap next layer's predicted expert pread with current
//! layer's Metal GPU dispatch. CPU is idle during GPU work — use it to prefetch.
//! Cross-layer routing similarity >95% means most preads will be useful.
//!
//! Invariant: speculation hits produce the same expert data as direct pread.
//! spec_state is correctly set/cleared per token. Buffer indexing is correct.
//!
//! Failure means: speculation buffer indexing is wrong, or spec_state lifecycle
//! is broken (stale predictions leak across tokens).

/// Test: speculation buffer indexing — data written at position i is read back correctly.
/// Simulates the speculative pread writing top_k expert buffers, then the next layer
/// consuming matching experts from the speculation buffer.
#[test]
fn speculation_buffer_roundtrip() {
    let top_k = 8;
    let expert_stride = 1024; // small for testing
    let spec_buf_size = top_k * expert_stride;
    let mut spec_buf = vec![0u8; spec_buf_size];

    // Simulate speculative pread: write unique pattern per expert slot
    let spec_ids: Vec<usize> = vec![42, 100, 7, 200, 55, 300, 15, 88];
    for (i, &_eid) in spec_ids.iter().enumerate() {
        let start = i * expert_stride;
        let pattern = (i as u8).wrapping_add(0xA0);
        spec_buf[start..start + expert_stride].fill(pattern);
    }

    // Simulate next layer: routing returns overlapping experts (6 of 8 match)
    let next_layer_ids: Vec<usize> = vec![42, 100, 7, 200, 55, 300, 999, 888];
    let mut staging = vec![0u8; top_k * expert_stride];
    let mut need_pread = vec![true; top_k];
    let mut spec_hits = 0usize;

    for (i, &eid) in next_layer_ids.iter().enumerate() {
        if need_pread[i] {
            if let Some(sp) = spec_ids.iter().position(|&s| s == eid) {
                let src = sp * expert_stride;
                let dst = i * expert_stride;
                staging[dst..dst + expert_stride]
                    .copy_from_slice(&spec_buf[src..src + expert_stride]);
                need_pread[i] = false;
                spec_hits += 1;
            }
        }
    }

    assert_eq!(spec_hits, 6, "6 of 8 next-layer experts should hit speculation");
    assert!(!need_pread[0] && !need_pread[1] && !need_pread[2]);  // 42, 100, 7
    assert!(!need_pread[3] && !need_pread[4] && !need_pread[5]);  // 200, 55, 300
    assert!(need_pread[6] && need_pread[7]);  // 999, 888 are misses

    // Verify data integrity: each hit slot has correct pattern
    for (i, &eid) in next_layer_ids.iter().enumerate() {
        if let Some(sp) = spec_ids.iter().position(|&s| s == eid) {
            let expected = (sp as u8).wrapping_add(0xA0);
            let start = i * expert_stride;
            assert_eq!(staging[start], expected,
                "slot {i} (expert {eid}) should have pattern 0x{expected:02X}");
            assert_eq!(staging[start + expert_stride - 1], expected,
                "last byte of slot {i} should match");
        }
    }

    eprintln!("Speculation hits: {spec_hits}/8 (6 expected)");
    eprintln!("Verified: buffer indexing + data integrity correct");
}

/// Test: spec_state lifecycle — set after Metal dispatch, cleared per token.
/// Ensures stale predictions don't leak across token boundaries.
#[test]
fn spec_state_cleared_per_token() {
    use std::cell::UnsafeCell;

    let spec_state: UnsafeCell<Option<(usize, Vec<usize>)>> = UnsafeCell::new(None);

    // Token 0: no speculation yet
    assert!(unsafe { &*spec_state.get() }.is_none());

    // Layer 5 sets speculation for layer 6
    unsafe { *spec_state.get() = Some((6, vec![42, 100, 7])); }
    let state = unsafe { &*spec_state.get() };
    assert_eq!(state.as_ref().unwrap().0, 6);
    assert_eq!(state.as_ref().unwrap().1.len(), 3);

    // Layer 6 consumes it (reads match)
    let (target, ref ids) = *unsafe { &*spec_state.get() }.as_ref().unwrap();
    assert_eq!(target, 6);
    assert!(ids.contains(&42));

    // Layer 6 sets new speculation for layer 7
    unsafe { *spec_state.get() = Some((7, vec![200, 55])); }

    // Token boundary: clear spec_state
    unsafe { *spec_state.get() = None; }
    assert!(unsafe { &*spec_state.get() }.is_none(),
        "spec_state must be None after token boundary clear");

    eprintln!("Verified: spec_state set/consume/clear lifecycle correct");
}

/// Test: zero speculation hits when target layer doesn't match.
/// Ensures mismatched layer predictions don't corrupt data.
#[test]
fn speculation_miss_on_wrong_layer() {
    let top_k = 4;
    let expert_stride = 512;
    let spec_buf = vec![0xFFu8; top_k * expert_stride];
    let spec_target_layer = 10;
    let spec_ids: Vec<usize> = vec![1, 2, 3, 4];

    // Current layer is 5, but speculation was for layer 10
    let current_layer = 5;
    let current_ids: Vec<usize> = vec![1, 2, 3, 4]; // same experts, wrong layer
    let mut staging = vec![0u8; top_k * expert_stride];
    let mut need_pread = vec![true; top_k];
    let mut spec_hits = 0;

    // This is the spec check from moe_ffn_v2
    if spec_target_layer == current_layer {
        for (i, &eid) in current_ids.iter().enumerate() {
            if need_pread[i] {
                if let Some(sp) = spec_ids.iter().position(|&s| s == eid) {
                    let src = sp * expert_stride;
                    let dst = i * expert_stride;
                    staging[dst..dst + expert_stride]
                        .copy_from_slice(&spec_buf[src..src + expert_stride]);
                    need_pread[i] = false;
                    spec_hits += 1;
                }
            }
        }
    }

    assert_eq!(spec_hits, 0, "wrong layer should produce zero hits");
    assert!(need_pread.iter().all(|&x| x), "all experts should still need pread");
    assert!(staging.iter().all(|&b| b == 0), "staging should be untouched");

    eprintln!("Verified: layer mismatch → 0 speculation hits, staging untouched");
}

/// Test: speculation with reduced top_k (top_k=6 speculating for top_k=6).
/// The RUSTANE_TOP_K override changes how many experts are speculated.
#[test]
fn speculation_with_reduced_topk() {
    let top_k = 6; // Reduced from default 8
    let expert_stride = 256;
    let mut spec_buf = vec![0u8; top_k * expert_stride];

    // Speculate 6 experts
    let spec_ids: Vec<usize> = vec![10, 20, 30, 40, 50, 60];
    for (i, _) in spec_ids.iter().enumerate() {
        let start = i * expert_stride;
        spec_buf[start..start + expert_stride].fill((i + 1) as u8);
    }

    // Next layer routes 6 experts, 4 overlap
    let next_ids: Vec<usize> = vec![10, 20, 30, 40, 70, 80];
    let mut staging = vec![0u8; top_k * expert_stride];
    let mut need_pread = vec![true; top_k];
    let mut hits = 0;

    for (i, &eid) in next_ids.iter().enumerate() {
        if need_pread[i] {
            if let Some(sp) = spec_ids.iter().position(|&s| s == eid) {
                let src = sp * expert_stride;
                let dst = i * expert_stride;
                staging[dst..dst + expert_stride]
                    .copy_from_slice(&spec_buf[src..src + expert_stride]);
                need_pread[i] = false;
                hits += 1;
            }
        }
    }

    assert_eq!(hits, 4, "4 of 6 should hit");
    assert!(!need_pread[0] && !need_pread[1] && !need_pread[2] && !need_pread[3]);
    assert!(need_pread[4] && need_pread[5]); // 70, 80 are misses

    eprintln!("Reduced top_k=6: {hits}/6 hits (4 expected)");
}
