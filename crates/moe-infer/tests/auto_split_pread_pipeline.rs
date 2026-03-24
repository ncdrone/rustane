//! Test: split-pread pipeline correctness.
//!
//! Verifies that dispatch_fused_phase + dispatch_down_phase produces identical
//! results to fused_and_down_single_cmdbuf (the original single-command-buffer path).
//! Also tests load_expert_partial for partial reads.

#[test]
fn partial_load_matches_full_load() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.bin");
    let expert_size = 4096;
    let num_experts = 4;

    // Write deterministic data
    {
        let mut f = std::fs::File::create(&path).unwrap();
        for eid in 0..num_experts {
            let data: Vec<u8> = (0..expert_size).map(|j| ((eid * 37 + j * 13) & 0xFF) as u8).collect();
            f.write_all(&data).unwrap();
        }
    }

    let layout = expert_pager::ExpertFileLayout { expert_size, num_experts };
    let loader = expert_pager::ExpertLoader::open(path.to_str().unwrap(), layout).unwrap();

    for eid in 0..num_experts as u32 {
        // Full load
        let mut full = vec![0u8; expert_size];
        loader.load_expert(eid, &mut full).unwrap();

        // Partial loads: first half + second half
        let half = expert_size / 2;
        let mut part1 = vec![0u8; half];
        let mut part2 = vec![0u8; half];
        loader.load_expert_partial(eid, &mut part1, 0, half).unwrap();
        loader.load_expert_partial(eid, &mut part2, half, half).unwrap();

        assert_eq!(&full[..half], &part1[..], "expert {eid} first half mismatch");
        assert_eq!(&full[half..], &part2[..], "expert {eid} second half mismatch");
    }
    eprintln!("partial_load_matches_full_load: OK");
}

#[test]
fn split_dispatch_matches_single_cmdbuf() {
    // This test requires Metal GPU — skip on CI or non-Mac.
    let mut metal = match moe_kernels::MetalDequantGemv::new() {
        Some(m) => m,
        None => { eprintln!("SKIP: no Metal device"); return; }
    };

    // K2 dims
    let in_features = 7168;
    let out_features = 2048;
    let group_size = 128;
    let top_k = 2; // use 2 experts for quick test

    metal.init_scratch(in_features, out_features, top_k, group_size);

    // Generate deterministic random weights
    let groups_per_row = in_features / group_size;
    let packed_per_row = in_features / 2;
    let total_packed = out_features * packed_per_row;
    let total_scales = out_features * groups_per_row;

    // We need: for each expert, gate(packed+scales+zeros) + up(packed+scales+zeros) + down(packed+scales+zeros)
    let gu_packed = out_features * (in_features / 2);
    let gu_scales = out_features * (in_features / group_size) * 2;
    let gu_total = gu_packed + gu_scales * 2;
    let dn_packed = in_features * (out_features / 2);
    let dn_groups = in_features * (out_features / group_size);
    let dn_scales = dn_groups * 2;
    let dn_total = dn_packed + dn_scales * 2;
    let expert_stride = 2 * gu_total + dn_total;

    let mut staging = vec![0x42u8; top_k * expert_stride];
    // Fill with pseudo-random data
    for i in 0..staging.len() {
        staging[i] = ((i * 31 + 17) & 0xFF) as u8;
    }

    let staging_metal = metal.wrap_mmap(&staging);

    // Build ops for top_k experts
    let mut fused_ops = Vec::new();
    let mut down_ops = Vec::new();
    for k in 0..top_k {
        let base = k * expert_stride;
        fused_ops.push(moe_kernels::FusedGateUpSiluOp {
            gate_packed_offset: base,
            gate_scales_offset: base + gu_packed,
            gate_zeros_offset: base + gu_packed + gu_scales,
            up_packed_offset: base + gu_total,
            up_scales_offset: base + gu_total + gu_packed,
            up_zeros_offset: base + gu_total + gu_packed + gu_scales,
            out_features, in_features, group_size,
        });
        down_ops.push(moe_kernels::ExpertGemvOp {
            packed_offset: base + 2 * gu_total,
            scales_offset: base + 2 * gu_total + dn_packed,
            zeros_offset: base + 2 * gu_total + dn_packed + dn_scales,
            out_features: in_features, in_features: out_features, group_size,
        });
    }

    // Input vector
    let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.001) - 3.5).collect();

    // Reference: single command buffer
    let ref_results = metal.fused_and_down_single_cmdbuf(&staging_metal, &fused_ops, &down_ops, &x);

    // Split: dispatch_fused_phase + dispatch_down_phase
    let fused_cmd = metal.dispatch_fused_phase(&staging_metal, &fused_ops, &x);
    fused_cmd.wait();
    let split_results = metal.dispatch_down_phase(&staging_metal, &down_ops);

    assert_eq!(ref_results.len(), split_results.len());
    for (k, (r, s)) in ref_results.iter().zip(split_results.iter()).enumerate() {
        assert_eq!(r.len(), s.len(), "expert {k} output length mismatch");
        let max_diff = r.iter().zip(s.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff == 0.0, "expert {k} max_diff={max_diff} — split != single");
    }
    eprintln!("split_dispatch_matches_single_cmdbuf: OK — {top_k} experts, max_diff=0.0");
}
