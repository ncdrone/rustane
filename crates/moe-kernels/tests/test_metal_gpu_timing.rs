//! Metal GPU-side timing for INT4 dequant GEMV kernel.

use moe_kernels::MetalDequantGemv;
use quantize::PackedWeights4Bit;
use half::f16;

#[test]
#[ignore = "requires Metal GPU"]
fn profile_metal_int4_gemv_gpu_timing() {
    eprintln!("\n=== Metal INT4 GEMV — GPU-Side Timing ===\n");

    let metal = MetalDequantGemv::new().expect("Metal GPU required");

    let dims = [
        ("Expert gate+up [7168→2048]", 7168usize, 2048usize, 128usize),
        ("Expert down [2048→7168]", 2048, 7168, 128),
    ];

    eprintln!("{:<35} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Kernel", "CPU µs", "GPU µs", "OH µs", "Data MB", "GPU GB/s");
    eprintln!("{}", "-".repeat(82));

    for (name, in_f, out_f, gs) in dims {
        let packed_per_row = in_f / 8;
        let num_groups = in_f / gs;

        let pw = PackedWeights4Bit {
            data: vec![0x88888888u32; out_f * packed_per_row],
            scales: vec![f16::from_f32(0.1); out_f * num_groups],
            zeros: vec![f16::from_f32(0.0); out_f * num_groups],
            in_features: in_f,
            out_features: out_f,
            group_size: gs,
        };

        let x = vec![0.1f32; in_f];

        let mut cpu_us_vec = Vec::new();
        let mut gpu_us_vec = Vec::new();

        for _ in 0..20 {
            let (_, cpu_s, gpu_s) = metal.gemv_gpu_timed(&pw, &x);
            cpu_us_vec.push(cpu_s * 1e6);
            gpu_us_vec.push(gpu_s * 1e6);
        }

        cpu_us_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gpu_us_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cpu_us = cpu_us_vec[cpu_us_vec.len() / 2];
        let gpu_us = gpu_us_vec[gpu_us_vec.len() / 2];
        let overhead = cpu_us - gpu_us;

        let data_bytes = (out_f * in_f) as f64 / 2.0
            + (out_f * num_groups * 2) as f64 * 2.0
            + (in_f * 4) as f64;
        let data_mb = data_bytes / 1e6;
        let gpu_bw = data_bytes / (gpu_us / 1e6) / 1e9;

        eprintln!("{:<35} {:>7.0} {:>7.0} {:>7.0} {:>7.1} {:>7.1}",
            name, cpu_us, gpu_us, overhead, data_mb, gpu_bw);
    }
}
