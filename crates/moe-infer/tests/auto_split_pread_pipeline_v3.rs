//! Correctness test: split-pread pipeline produces identical staging buffer
//! contents as single-pread. Verifies that load_expert_partial(gate+up) +
//! load_expert_partial(down) == load_expert(full).
//!
//! What was optimized: moe_ffn_v2 now splits expert pread into two phases:
//! 1) gate+up pread (overlaps shared_ffn via thread::scope)
//! 2) down pread (overlaps GPU fused phase via dispatch_fused_phase non-blocking)
//!
//! Invariant: the staging buffer contents are byte-identical whether loaded as
//! one full pread or two partial preads. The Metal split dispatch (separate
//! command buffers for fused and down phases) is ordered by the serial queue.
//!
//! Failure means: load_expert_partial has an offset/length bug, or the staging
//! buffer layout doesn't match the two-phase split.

use std::io::Write;
use tempfile::NamedTempFile;

/// Create a temporary expert file with known data pattern.
/// Each expert has `expert_stride` bytes: [gate+up: gu_size][down: dn_size].
fn create_test_expert_file(num_experts: usize, expert_stride: usize) -> (NamedTempFile, Vec<u8>) {
    let mut data = vec![0u8; num_experts * expert_stride];
    // Fill with identifiable pattern: expert_id * 0x10 + byte_offset % 251
    for eid in 0..num_experts {
        let base = eid * expert_stride;
        for j in 0..expert_stride {
            data[base + j] = ((eid * 0x10 + j) % 251) as u8;
        }
    }
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(&data).unwrap();
    f.flush().unwrap();
    (f, data)
}

#[test]
fn split_pread_matches_full_pread() {
    use expert_pager::loader::{ExpertFileLayout, ExpertLoader};

    // Dimensions matching K2 structure (scaled down for test speed)
    let gu_size = 1024; // gate+up portion
    let dn_size = 512;  // down portion
    let expert_stride = gu_size + dn_size;
    let num_experts = 4;

    let (tmpfile, expected_data) = create_test_expert_file(num_experts, expert_stride);
    let path = tmpfile.path().to_str().unwrap();

    let layout = ExpertFileLayout {
        expert_size: expert_stride,
        num_experts,
    };
    let loader = ExpertLoader::open(path, layout).unwrap();

    for eid in 0..num_experts as u32 {
        // Method A: single full pread
        let mut full_buf = vec![0u8; expert_stride];
        loader.load_expert(eid, &mut full_buf).unwrap();

        // Method B: two partial preads (gate+up then down)
        let mut split_buf = vec![0u8; expert_stride];
        loader.load_expert_partial(eid, &mut split_buf[..gu_size], 0, gu_size).unwrap();
        loader.load_expert_partial(eid, &mut split_buf[gu_size..], gu_size, dn_size).unwrap();

        // Verify byte-identical
        assert_eq!(
            full_buf, split_buf,
            "expert {eid}: split pread differs from full pread"
        );

        // Verify against known data
        let base = eid as usize * expert_stride;
        assert_eq!(
            &full_buf[..],
            &expected_data[base..base + expert_stride],
            "expert {eid}: pread data doesn't match source file"
        );
    }

    eprintln!("split_pread: {num_experts} experts verified, split == full == source");
}

/// Verify that partial pread with offset correctly reads the down portion.
/// This catches off-by-one errors in the offset_within calculation.
#[test]
fn partial_pread_offset_boundary() {
    use expert_pager::loader::{ExpertFileLayout, ExpertLoader};

    // Use sizes that don't align to page boundaries (catch alignment bugs)
    let gu_size = 1337;
    let dn_size = 997;
    let expert_stride = gu_size + dn_size;
    let num_experts = 2;

    let (tmpfile, expected_data) = create_test_expert_file(num_experts, expert_stride);
    let path = tmpfile.path().to_str().unwrap();

    let layout = ExpertFileLayout {
        expert_size: expert_stride,
        num_experts,
    };
    let loader = ExpertLoader::open(path, layout).unwrap();

    // Read only the down portion of expert 1
    let eid = 1u32;
    let mut dn_buf = vec![0u8; dn_size];
    loader.load_expert_partial(eid, &mut dn_buf, gu_size, dn_size).unwrap();

    let expected_base = eid as usize * expert_stride + gu_size;
    assert_eq!(
        &dn_buf[..],
        &expected_data[expected_base..expected_base + dn_size],
        "down-only partial pread has wrong data"
    );

    eprintln!("partial_pread_offset: non-aligned sizes verified (gu={gu_size}, dn={dn_size})");
}
