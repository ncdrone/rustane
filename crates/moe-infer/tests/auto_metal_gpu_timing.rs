//! Metal GPU-side timing: measure actual GPU compute time via MTLCommandBuffer timestamps.
//!
//! Uses MTLCommandBuffer.GPUStartTime / GPUEndTime for precise GPU-side timing,
//! not CPU-side Instant::now() which includes dispatch overhead + queue latency.
//!
//! This tells us:
//! 1. How much of the Metal dispatch is GPU compute vs CPU overhead
//! 2. What the actual GPU bandwidth utilization is
//! 3. Where the 64% bandwidth waste goes

use std::path::Path;

fn ws_root() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires K2 weights + Metal GPU"]
fn profile_metal_gpu_timestamps() {
    let root = ws_root();
    let weights_dir = root.join("weights/rustane-k2");
    let config_path = root.join("configs/kimi-k2.toml");

    if !weights_dir.join("backbone.bin").exists() {
        eprintln!("SKIP: K2 weights not found");
        return;
    }

    eprintln!("\n=== Metal GPU-Side Timing (GPUStartTime/GPUEndTime) ===\n");

    let model = moe_infer::generate_v2::ModelV2::load(&weights_dir, &config_path)
        .expect("load K2");

    let metal = model.metal.as_ref().expect("Metal GPU required");

    // Create a standalone INT4 GEMV dispatch and measure GPU time
    // We need access to an expert's packed weights. Let's use the benchmark approach:
    // load one expert, dispatch, measure.

    eprintln!("Testing Metal dispatch with GPU timestamps...");
    eprintln!("Expert dims: in_features=7168, moe_inter=2048, group_size=128");
    eprintln!("");

    // Measure empty command buffer roundtrip (baseline overhead)
    let queue = &metal.queue;
    let mut empty_times = Vec::new();
    for _ in 0..20 {
        let cmd = queue.commandBuffer().expect("cmd buf");
        let t = std::time::Instant::now();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted(); }
        let cpu_us = t.elapsed().as_micros();
        let gpu_start = unsafe { cmd.GPUStartTime() };
        let gpu_end = unsafe { cmd.GPUEndTime() };
        let gpu_us = ((gpu_end - gpu_start) * 1e6) as u64;
        empty_times.push((cpu_us, gpu_us));
    }
    // Skip first 5 (warmup)
    let avg_cpu: f64 = empty_times[5..].iter().map(|(c, _)| *c as f64).sum::<f64>() / 15.0;
    let avg_gpu: f64 = empty_times[5..].iter().map(|(_, g)| *g as f64).sum::<f64>() / 15.0;
    eprintln!("Empty command buffer roundtrip:");
    eprintln!("  CPU-side: {:.0}µs  GPU-side: {:.0}µs", avg_cpu, avg_gpu);
    eprintln!("  Overhead (CPU - GPU): {:.0}µs", avg_cpu - avg_gpu);
    eprintln!("");

    // Now measure a real compute dispatch
    // We need to set up a Metal compute pipeline and dispatch it.
    // Use the existing dequant_4bit_gemv_v2 shader.

    // Create dummy data for profiling
    let in_features: usize = 7168;
    let out_features: usize = 2048;
    let group_size: usize = 128;
    let packed_per_row = in_features / 8; // 896

    // Allocate Metal buffers
    let device = &metal.device;
    let packed_bytes = out_features * packed_per_row * 4; // uint32 per packed element
    let scales_bytes = out_features * (in_features / group_size) * 2; // fp16
    let zeros_bytes = scales_bytes;
    let x_bytes = in_features * 4; // f32
    let y_bytes = out_features * 4; // f32

    use objc2_metal::{MTLResourceOptions, MTLSize};
    let opts = MTLResourceOptions::StorageModeShared;
    let packed_buf = device.newBufferWithLength_options(packed_bytes, opts).unwrap();
    let scales_buf = device.newBufferWithLength_options(scales_bytes, opts).unwrap();
    let zeros_buf = device.newBufferWithLength_options(zeros_bytes, opts).unwrap();
    let x_buf = device.newBufferWithLength_options(x_bytes, opts).unwrap();
    let y_buf = device.newBufferWithLength_options(y_bytes, opts).unwrap();

    // Fill x with dummy data
    unsafe {
        let x_ptr = x_buf.contents().as_ptr() as *mut f32;
        for i in 0..in_features { std::ptr::write(x_ptr.add(i), 0.1); }
    }

    // Dispatch the v2 shader
    let tg_size: usize = 256;
    let rows_per_tg: usize = 8;
    let num_tgs = (out_features + rows_per_tg - 1) / rows_per_tg;

    let in_feat_buf = device.newBufferWithBytes_length_options(
        &(in_features as u32) as *const u32 as *const _, 4, opts).unwrap();
    let group_buf = device.newBufferWithBytes_length_options(
        &(group_size as u32) as *const u32 as *const _, 4, opts).unwrap();
    let out_feat_buf = device.newBufferWithBytes_length_options(
        &(out_features as u32) as *const u32 as *const _, 4, opts).unwrap();

    eprintln!("Dispatch config: TG={tg_size}, ROWS_PER_TG={rows_per_tg}, num_TGs={num_tgs}");
    eprintln!("Dimensions: in_features={in_features}, out_features={out_features}");
    eprintln!("");

    // Warmup
    for _ in 0..5 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        unsafe {
            enc.setComputePipelineState(&metal.pipeline_v2);
            enc.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&scales_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&zeros_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&y_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&in_feat_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&group_buf), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&out_feat_buf), 0, 7);
            let tgs = MTLSize { width: num_tgs as u64, height: 1, depth: 1 };
            let tpt = MTLSize { width: tg_size as u64, height: 1, depth: 1 };
            enc.dispatchThreadgroups_threadsPerThreadgroup(tgs, tpt);
        }
        enc.endEncoding();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted(); }
    }

    // Measured dispatches
    let mut cpu_times = Vec::new();
    let mut gpu_times = Vec::new();

    for _ in 0..30 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        unsafe {
            enc.setComputePipelineState(&metal.pipeline_v2);
            enc.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&scales_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&zeros_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&y_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&in_feat_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&group_buf), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&out_feat_buf), 0, 7);
            let tgs = MTLSize { width: num_tgs as u64, height: 1, depth: 1 };
            let tpt = MTLSize { width: tg_size as u64, height: 1, depth: 1 };
            enc.dispatchThreadgroups_threadsPerThreadgroup(tgs, tpt);
        }
        enc.endEncoding();

        let t = std::time::Instant::now();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted(); }
        let cpu_us = t.elapsed().as_micros() as f64;

        let gpu_start = unsafe { cmd.GPUStartTime() };
        let gpu_end = unsafe { cmd.GPUEndTime() };
        let gpu_us = (gpu_end - gpu_start) * 1e6;

        cpu_times.push(cpu_us);
        gpu_times.push(gpu_us);
    }

    let avg_cpu: f64 = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let avg_gpu: f64 = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
    let overhead = avg_cpu - avg_gpu;

    // Bandwidth calculation
    // Data read: packed weights (4 bits per element = 0.5 bytes)
    let weight_bytes = (out_features * in_features) as f64 / 2.0; // 4-bit packed
    let scale_bytes = (out_features * in_features / group_size) as f64 * 2.0; // fp16
    let zero_bytes = scale_bytes;
    let x_bytes_read = in_features as f64 * 4.0; // f32
    let total_read = weight_bytes + scale_bytes + zero_bytes + x_bytes_read;
    let total_read_mb = total_read / 1e6;
    let gpu_bw = total_read / (avg_gpu / 1e6) / 1e9; // GB/s

    eprintln!("dequant_4bit_gemv_v2 [{in_features}→{out_features}]:");
    eprintln!("  CPU-side (commit→wait): {:.0}µs", avg_cpu);
    eprintln!("  GPU-side (compute):     {:.0}µs", avg_gpu);
    eprintln!("  Overhead (CPU - GPU):   {:.0}µs ({:.0}%)", overhead, overhead / avg_cpu * 100.0);
    eprintln!("  Data read:              {:.1} MB", total_read_mb);
    eprintln!("  GPU bandwidth:          {:.1} GB/s", gpu_bw);
    eprintln!("  Peak bandwidth (M4 Max): ~400 GB/s");
    eprintln!("  Utilization:            {:.0}%", gpu_bw / 400.0 * 100.0);
}
