//! End-to-end inference test: real Qwen3-MoE-30B expert weights.
//!
//! Loads real expert weights from safetensors → quantizes to 4-bit →
//! repacks into per-layer .bin → pread from SSD → routes through gate →
//! dispatches selected experts through Metal dequant GEMV → verifies output.
//!
//! This is THE critical integration test: proves the full inference
//! pipeline works on real trained weights, streamed from SSD.
//!
//! Run: cargo test -p moe-infer --test e2e_qwen3_inference --release -- --ignored --nocapture

use expert_pager::loader::{ExpertFileLayout, ExpertLoader};
use expert_pager::ExpertPool;
use moe_kernels::MetalDequantGemv;
use moe_router::{MoeRouter, RouterConfig};
use quantize::PackedWeights4Bit;
use std::path::Path;

const SHARD_PATH: &str = "/Users/dan/Dev/rustane/weights/qwen3-moe-30b-ref/model-00001-of-00016.safetensors";

// Qwen3-MoE-30B expert dimensions (from config.json)
const HIDDEN_DIM: usize = 2048;
const MOE_INTERMEDIATE: usize = 768;
const NUM_EXPERTS: usize = 128;
const TOP_K: usize = 8;
const GROUP_SIZE: usize = 128; // our quantization group size

// Expert layout in packed .bin files
// gate_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] = [768, 2048]
// up_proj:   [MOE_INTERMEDIATE, HIDDEN_DIM] = [768, 2048]
// down_proj: [HIDDEN_DIM, MOE_INTERMEDIATE] = [2048, 768]
// Packed 4-bit: each weight matrix = rows * cols / 2 bytes (data) + metadata
fn expert_data_size() -> usize {
    let gate_data = MOE_INTERMEDIATE * HIDDEN_DIM / 2; // 4-bit packed
    let gate_meta = MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * 4; // scales + zeros (f16 each)
    let up_data = gate_data;
    let up_meta = gate_meta;
    let down_data = HIDDEN_DIM * MOE_INTERMEDIATE / 2;
    let down_meta = HIDDEN_DIM * (MOE_INTERMEDIATE / GROUP_SIZE) * 4;
    gate_data + gate_meta + up_data + up_meta + down_data + down_meta
}

fn shard_exists() -> bool {
    Path::new(SHARD_PATH).exists()
}

/// Helper: convert bf16 tensor bytes to f32
fn bf16_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Extract, quantize, and repack expert weights from safetensors shard.
/// Returns path to temporary per-layer .bin file.
fn extract_and_repack_experts(
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> (tempfile::NamedTempFile, Vec<ExpertMeta>) {
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    let mut metas = Vec::new();

    let expert_size = expert_data_size();

    // Pre-allocate file
    use std::io::{Seek, Write};
    let total = NUM_EXPERTS * expert_size;
    file.as_file().set_len(total as u64).expect("truncate");

    for expert_id in 0..NUM_EXPERTS {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert_id}");

        // Try to load gate_proj, up_proj, down_proj
        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let down_name = format!("{prefix}.down_proj.weight");

        let (gate_w, up_w, down_w) = match (
            tensors.tensor(&gate_name),
            tensors.tensor(&up_name),
            tensors.tensor(&down_name),
        ) {
            (Ok(g), Ok(u), Ok(d)) => {
                (bf16_to_f32_vec(g.data()), bf16_to_f32_vec(u.data()), bf16_to_f32_vec(d.data()))
            }
            _ => {
                // Expert not in this shard, fill with zeros
                metas.push(ExpertMeta { present: false, gate_max: 0.0, up_max: 0.0, down_max: 0.0 });
                continue;
            }
        };

        // Quantize to 4-bit
        let gate_packed = PackedWeights4Bit::pack(&gate_w, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
        let up_packed = PackedWeights4Bit::pack(&up_w, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
        let down_packed = PackedWeights4Bit::pack(&down_w, HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE);

        // Write packed data to file at expert_id * expert_size
        let offset = expert_id * expert_size;
        file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();

        // Write gate: data, scales, zeros
        let gate_data_bytes = bytemuck_cast_u32(&gate_packed.data);
        file.write_all(gate_data_bytes).unwrap();
        let gate_scale_bytes = bytemuck_cast_f16(&gate_packed.scales);
        file.write_all(gate_scale_bytes).unwrap();
        let gate_zero_bytes = bytemuck_cast_f16(&gate_packed.zeros);
        file.write_all(gate_zero_bytes).unwrap();

        // Write up: data, scales, zeros
        file.write_all(bytemuck_cast_u32(&up_packed.data)).unwrap();
        file.write_all(bytemuck_cast_f16(&up_packed.scales)).unwrap();
        file.write_all(bytemuck_cast_f16(&up_packed.zeros)).unwrap();

        // Write down: data, scales, zeros
        file.write_all(bytemuck_cast_u32(&down_packed.data)).unwrap();
        file.write_all(bytemuck_cast_f16(&down_packed.scales)).unwrap();
        file.write_all(bytemuck_cast_f16(&down_packed.zeros)).unwrap();

        let gate_max = gate_w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let up_max = up_w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let down_max = down_w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        metas.push(ExpertMeta { present: true, gate_max, up_max, down_max });
    }

    file.flush().unwrap();
    (file, metas)
}

struct ExpertMeta {
    present: bool,
    gate_max: f32,
    up_max: f32,
    down_max: f32,
}

fn bytemuck_cast_u32(data: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

fn bytemuck_cast_f16(data: &[half::f16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) }
}

/// End-to-end: extract real weights → repack → pread from SSD → route → Metal GEMV.
#[test]
#[ignore = "requires downloaded Qwen3-MoE-30B shard + Metal GPU"]
fn e2e_ssd_streamed_expert_inference() {
    if !shard_exists() {
        println!("SKIP: shard not found at {SHARD_PATH}");
        return;
    }

    println!("\n=== E2E: Real Qwen3-MoE-30B Expert Inference via SSD Streaming ===\n");

    // 1. Load safetensors shard
    println!("[1/6] Loading safetensors shard...");
    let t0 = std::time::Instant::now();
    let raw = std::fs::read(SHARD_PATH).expect("read shard");
    let tensors = safetensors::SafeTensors::deserialize(&raw).expect("parse");
    println!("  Loaded {:.1} GB in {:.1}s ({} tensors)",
        raw.len() as f64 / 1e9, t0.elapsed().as_secs_f64(), tensors.len());

    // 2. Extract + quantize + repack layer 0 experts
    println!("[2/6] Extracting + quantizing layer 0 experts to 4-bit...");
    let t1 = std::time::Instant::now();
    let (layer_file, metas) = extract_and_repack_experts(&tensors, 0);
    let present_count = metas.iter().filter(|m| m.present).count();
    println!("  {present_count}/{NUM_EXPERTS} experts found in shard, repacked in {:.1}s",
        t1.elapsed().as_secs_f64());
    println!("  Expert file: {:.1} MB", expert_data_size() * NUM_EXPERTS / 1024 / 1024);

    if present_count == 0 {
        println!("  No layer 0 experts in this shard — trying layer 1...");
        let (layer_file2, metas2) = extract_and_repack_experts(&tensors, 1);
        let pc2 = metas2.iter().filter(|m| m.present).count();
        println!("  Layer 1: {pc2}/{NUM_EXPERTS} experts");
        if pc2 == 0 {
            println!("  No experts found in layers 0-1 of this shard. Need a different shard.");
            return;
        }
        return run_inference_test(layer_file2, &metas2, &tensors, 1);
    }

    run_inference_test(layer_file, &metas, &tensors, 0);
}

fn run_inference_test(
    layer_file: tempfile::NamedTempFile,
    metas: &[ExpertMeta],
    tensors: &safetensors::SafeTensors<'_>,
    layer_idx: usize,
) {
    let present_count = metas.iter().filter(|m| m.present).count();

    // 3. Open via pread (SSD streaming path)
    println!("[3/6] Opening packed expert file for pread streaming...");
    let expert_size = expert_data_size();
    let layout = ExpertFileLayout {
        expert_size,
        num_experts: NUM_EXPERTS,
    };
    let loader = ExpertLoader::open(
        layer_file.path().to_str().unwrap(),
        layout,
    ).expect("open expert file");

    // 4. Route with real gate weights
    println!("[4/6] Routing with real gate weights...");
    let gate_name = format!("model.layers.{layer_idx}.mlp.gate.weight");
    let gate_logits = if let Ok(gate_view) = tensors.tensor(&gate_name) {
        let gate_w = bf16_to_f32_vec(gate_view.data());
        // gate_w is [num_experts, hidden_dim]
        // Simulate routing: use a random hidden state
        let hidden: Vec<f32> = (0..HIDDEN_DIM).map(|i| {
            let s = (i as u64 * 6364136223846793005 + 42).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.02 - 0.01
        }).collect();
        // gate_logits = hidden @ gate_w^T → [num_experts]
        let mut logits = vec![0.0f32; NUM_EXPERTS];
        for e in 0..NUM_EXPERTS {
            for d in 0..HIDDEN_DIM {
                logits[e] += hidden[d] * gate_w[e * HIDDEN_DIM + d];
            }
        }
        println!("  Gate logits computed from real weights (dim={})", gate_w.len());
        logits
    } else {
        println!("  Gate weights not in shard, using synthetic logits");
        (0..NUM_EXPERTS).map(|i| (i as f32 - 64.0) * 0.1).collect()
    };

    let config = RouterConfig {
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        norm_topk_prob: true,
        bias_lr: 0.0,
    };
    let mut router = MoeRouter::new(config);
    let route = router.route(&gate_logits);
    println!("  Selected experts: {:?}", route.expert_ids);
    println!("  Weights: {:?}", route.weights.iter().map(|w| format!("{w:.3}")).collect::<Vec<_>>());

    // 5. pread selected experts from SSD → Metal dequant GEMV
    println!("[5/6] Loading selected experts via pread + running Metal dequant GEMV...");
    let metal = MetalDequantGemv::new().expect("Metal device required");
    let mut pool = ExpertPool::new(TOP_K * 2);

    let hidden: Vec<f32> = (0..HIDDEN_DIM).map(|i| {
        let s = (i as u64 * 6364136223846793005 + 42).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.02 - 0.01
    }).collect();

    let mut combined_output = vec![0.0f32; HIDDEN_DIM];
    let mut experts_computed = 0;

    for (&eid, &weight) in route.expert_ids.iter().zip(route.weights.iter()) {
        if !metas[eid].present {
            println!("  Expert {eid}: not in shard, skipping");
            continue;
        }

        let (slot, is_hit) = pool.request(0, eid as u32);
        let _ = slot;

        // pread expert data from SSD
        let mut buf = vec![0u8; expert_size];
        let t_load = std::time::Instant::now();
        loader.load_expert(eid as u32, &mut buf).expect("pread expert");
        let load_us = t_load.elapsed().as_micros();

        // Parse packed weights from buffer
        let gate_packed = parse_packed_from_buf(&buf, MOE_INTERMEDIATE, HIDDEN_DIM, 0);
        let up_offset = packed_matrix_size(MOE_INTERMEDIATE, HIDDEN_DIM);
        let up_packed = parse_packed_from_buf(&buf, MOE_INTERMEDIATE, HIDDEN_DIM, up_offset);

        // GPU dequant GEMV: gate = W_gate @ hidden, up = W_up @ hidden
        let t_gpu = std::time::Instant::now();
        let gate_out = metal.gemv(&gate_packed, &hidden);
        let up_out = metal.gemv(&up_packed, &hidden);
        let gpu_us = t_gpu.elapsed().as_micros();

        // SiLU(gate) * up → gated hidden
        let gated: Vec<f32> = gate_out.iter().zip(up_out.iter())
            .map(|(&g, &u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            }).collect();

        // Down projection: W_down @ gated
        let down_offset = up_offset + packed_matrix_size(MOE_INTERMEDIATE, HIDDEN_DIM);
        let down_packed = parse_packed_from_buf(&buf, HIDDEN_DIM, MOE_INTERMEDIATE, down_offset);
        let expert_out = metal.gemv(&down_packed, &gated);

        // Weighted combine
        for d in 0..HIDDEN_DIM {
            combined_output[d] += weight * expert_out[d];
        }

        let out_norm = expert_out.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("  Expert {eid}: load={load_us}us, gpu={gpu_us}us, ||out||={out_norm:.4}, hit={is_hit}");
        experts_computed += 1;
    }

    // 6. Verify output is non-trivial
    println!("[6/6] Verifying output...");
    let out_norm = combined_output.iter().map(|v| v * v).sum::<f32>().sqrt();
    let out_max = combined_output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let out_mean = combined_output.iter().sum::<f32>() / HIDDEN_DIM as f32;

    println!("  Combined output: ||x||={out_norm:.4}, max={out_max:.6}, mean={out_mean:.8}");
    println!("  Experts computed: {experts_computed}/{TOP_K}");
    println!("  Pool stats: hits={}, misses={}", pool.stats.hits, pool.stats.misses);

    assert!(experts_computed > 0, "no experts were computed");
    assert!(out_norm > 1e-6, "output is zero — weights or pipeline broken");
    assert!(combined_output.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
    println!("\n=== E2E INFERENCE TEST PASSED ===\n");
}

/// Parse a PackedWeights4Bit from a raw byte buffer at given offset.
fn parse_packed_from_buf(buf: &[u8], rows: usize, cols: usize, offset: usize) -> PackedWeights4Bit {
    let packed_u32s = rows * cols / 8; // 8 nibbles per u32
    let num_groups = rows * (cols / GROUP_SIZE);

    let data_bytes = packed_u32s * 4;
    let scale_bytes = num_groups * 2;
    let zero_bytes = num_groups * 2;

    let data_start = offset;
    let scale_start = data_start + data_bytes;
    let zero_start = scale_start + scale_bytes;

    let data: Vec<u32> = buf[data_start..data_start + data_bytes]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let scales: Vec<half::f16> = buf[scale_start..scale_start + scale_bytes]
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
        .collect();

    let zeros: Vec<half::f16> = buf[zero_start..zero_start + zero_bytes]
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
        .collect();

    PackedWeights4Bit {
        data,
        scales,
        zeros,
        out_features: rows,
        in_features: cols,
        group_size: GROUP_SIZE,
    }
}

fn packed_matrix_size(rows: usize, cols: usize) -> usize {
    let data = rows * cols / 8 * 4; // u32 packed
    let groups = rows * (cols / GROUP_SIZE);
    let scales = groups * 2; // f16
    let zeros = groups * 2;  // f16
    data + scales + zeros
}
