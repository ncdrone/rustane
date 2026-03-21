//! Metal dispatch overhead measurement (encode, commit, wait).
//! Tests 4 patterns to quantify cmd_buf overhead independent of compute.

use moe_kernels::MetalDequantGemv;

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

#[test]
fn metal_dispatch_latency_patterns() {
    let metal = MetalDequantGemv::new().expect("Metal GPU required");

    // Create tiny dummy weights [64, 64] — minimal compute to isolate overhead
    let out_features = 64usize;
    let in_features = 64usize;
    let group_size = 32usize;
    let packed_u32s = out_features * in_features / 8;
    let num_groups = out_features * (in_features / group_size);

    // Pack zeros (minimal compute, just measuring dispatch overhead)
    let packed_bytes: Vec<u8> = vec![0u8; packed_u32s * 4 + num_groups * 2 + num_groups * 2];

    let mmap_buf = metal.wrap_mmap(&packed_bytes);
    let x = vec![1.0f32; in_features];

    use moe_kernels::ExpertGemvOp;
    let gu_packed = packed_u32s * 4;
    let gu_scales_off = gu_packed;
    let gu_zeros_off = gu_packed + num_groups * 2;

    let op = ExpertGemvOp {
        packed_offset: 0,
        scales_offset: gu_scales_off,
        zeros_offset: gu_zeros_off,
        out_features,
        in_features,
        group_size,
    };

    let iters = 100;

    // Pattern 1: 1 dispatch, 1 commit
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let _r = metal.batch_gemv_mmap(&mmap_buf, &[ExpertGemvOp {
                packed_offset: op.packed_offset,
                scales_offset: op.scales_offset,
                zeros_offset: op.zeros_offset,
                out_features: op.out_features,
                in_features: op.in_features,
                group_size: op.group_size,
            }], &[&x]);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        eprintln!("Pattern 1 (1 dispatch, 1 commit):  median {:.0} µs", med);
    }

    // Pattern 2: 8 dispatches, 1 commit (existing batch pattern)
    {
        let ops: Vec<ExpertGemvOp> = (0..8).map(|_| ExpertGemvOp {
            packed_offset: op.packed_offset,
            scales_offset: op.scales_offset,
            zeros_offset: op.zeros_offset,
            out_features: op.out_features,
            in_features: op.in_features,
            group_size: op.group_size,
        }).collect();
        let x_slices: Vec<&[f32]> = vec![&x; 8];

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let _r = metal.batch_gemv_mmap(&mmap_buf, &ops, &x_slices);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        eprintln!("Pattern 2 (8 dispatches, 1 commit): median {:.0} µs", med);
    }

    // Pattern 3: 8 dispatches, 8 commits (naive per-dispatch)
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            for _ in 0..8 {
                let _r = metal.batch_gemv_mmap(&mmap_buf, &[ExpertGemvOp {
                    packed_offset: op.packed_offset,
                    scales_offset: op.scales_offset,
                    zeros_offset: op.zeros_offset,
                    out_features: op.out_features,
                    in_features: op.in_features,
                    group_size: op.group_size,
                }], &[&x]);
            }
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        eprintln!("Pattern 3 (8 dispatches, 8 commits): median {:.0} µs", med);
    }

    // Pattern 4: 384 dispatches, 1 commit (all-layers pattern: 8 experts × 48 layers)
    {
        let ops: Vec<ExpertGemvOp> = (0..384).map(|_| ExpertGemvOp {
            packed_offset: op.packed_offset,
            scales_offset: op.scales_offset,
            zeros_offset: op.zeros_offset,
            out_features: op.out_features,
            in_features: op.in_features,
            group_size: op.group_size,
        }).collect();
        let x_slices: Vec<&[f32]> = vec![&x; 384];

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let _r = metal.batch_gemv_mmap(&mmap_buf, &ops, &x_slices);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        eprintln!("Pattern 4 (384 dispatches, 1 commit): median {:.0} µs", med);
    }

    // Pattern 5: Real-size GEMV [768, 2048] — 1 dispatch (matches moe_inter × hidden)
    {
        let real_out = 768usize;
        let real_in = 2048usize;
        let real_packed_u32s = real_out * real_in / 8;
        let real_groups = real_out * (real_in / group_size);
        let real_bytes: Vec<u8> = vec![0u8; real_packed_u32s * 4 + real_groups * 2 + real_groups * 2];
        let real_buf = metal.wrap_mmap(&real_bytes);
        let real_x = vec![1.0f32; real_in];

        let real_op = ExpertGemvOp {
            packed_offset: 0,
            scales_offset: real_packed_u32s * 4,
            zeros_offset: real_packed_u32s * 4 + real_groups * 2,
            out_features: real_out,
            in_features: real_in,
            group_size,
        };

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let _r = metal.batch_gemv_mmap(&real_buf, &[ExpertGemvOp {
                packed_offset: real_op.packed_offset,
                scales_offset: real_op.scales_offset,
                zeros_offset: real_op.zeros_offset,
                out_features: real_op.out_features,
                in_features: real_op.in_features,
                group_size: real_op.group_size,
            }], &[&real_x]);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        eprintln!("Pattern 5 (1×[768,2048] real-size): median {:.0} µs", med);

        // Pattern 6: 16 real-size dispatches, 1 commit (gate+up for 8 experts)
        let ops16: Vec<ExpertGemvOp> = (0..16).map(|_| ExpertGemvOp {
            packed_offset: real_op.packed_offset,
            scales_offset: real_op.scales_offset,
            zeros_offset: real_op.zeros_offset,
            out_features: real_op.out_features,
            in_features: real_op.in_features,
            group_size: real_op.group_size,
        }).collect();
        let real_slices16: Vec<&[f32]> = vec![&real_x; 16];

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let _r = metal.batch_gemv_mmap(&real_buf, &ops16, &real_slices16);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let med = median(&mut times);
        eprintln!("Pattern 6 (16×[768,2048], 1 commit): median {:.0} µs", med);
    }

    eprintln!("\n=== ANALYSIS ===");
    eprintln!("Compare Pattern 2 vs 3 for per-commit overhead.");
    eprintln!("Compare Pattern 1 vs 5 for compute vs overhead ratio.");
    eprintln!("Compare Pattern 5 vs 6 for dispatch scaling at real sizes.");
}
