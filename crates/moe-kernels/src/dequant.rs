//! Metal GPU fused 4-bit dequantization + GEMV kernel.
//!
//! Reads packed 4-bit weights, dequantizes on the fly using per-group
//! scale/zero, and computes matrix-vector product in a single pass.
//! Uses FMA for dequant (fma(nibble, scale, zero)) and accumulation.
//!
//! Threading model: one threadgroup (256 threads) per output row.
//! Threads divide columns among themselves for coalesced reads,
//! then reduce via SIMD + shared memory.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use quantize::PackedWeights4Bit;
use std::ffi::c_void;
use std::ptr::NonNull;

const TG_SIZE: usize = 256;

const DEQUANT_GEMV_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Fused 4-bit dequantization + GEMV: y = W_4bit @ x
// Pre-factored FMA: pre-compute sx = scale*x[col], bx = zero*x[col]
// per group, then use nibble as scalar: sum += nibble * sx + bx.
// This breaks the dependent FMA chain (flash-moe pattern).
//
// Threadgroup x-caching: load x into shared memory once, reuse across
// all threads computing different output rows.
//
// One threadgroup (256 threads) per output row.
// Partial sums reduced via SIMD + shared memory.
kernel void dequant_4bit_gemv(
    device const uint32_t* packed   [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const half* zeros        [[buffer(2)]],
    device const float* x           [[buffer(3)]],
    device float* y                 [[buffer(4)]],
    constant uint& in_features      [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    uint row    [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]]
) {
    const uint packed_per_row = in_features / 8;
    const uint num_groups = in_features / group_size;
    const uint TG = 256;
    const uint groups_per_8 = group_size / 8;

    // Cache x in threadgroup shared memory (max 4096 floats = 16KB)
    threadgroup float x_cache[4096];
    for (uint i = tid; i < in_features; i += TG) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0;

    device const uint32_t* row_packed = packed + row * packed_per_row;

    for (uint pi = tid; pi < packed_per_row; pi += TG) {
        uint col = pi * 8;
        uint group_idx = row * num_groups + pi / groups_per_8;
        float scale = float(scales[group_idx]);
        float zero = float(zeros[group_idx]);

        uint32_t pack = row_packed[pi];

        // Pre-factor: sx[i] = scale * x[col+i], bx[i] = zero * x[col+i]
        // Then: contribution = nibble * sx + bx = (nibble * scale + zero) * x
        // This is algebraically identical but breaks the dependent FMA chain.
        float sx0 = scale * x_cache[col    ]; float bx0 = zero * x_cache[col    ];
        float sx1 = scale * x_cache[col + 1]; float bx1 = zero * x_cache[col + 1];
        float sx2 = scale * x_cache[col + 2]; float bx2 = zero * x_cache[col + 2];
        float sx3 = scale * x_cache[col + 3]; float bx3 = zero * x_cache[col + 3];
        float sx4 = scale * x_cache[col + 4]; float bx4 = zero * x_cache[col + 4];
        float sx5 = scale * x_cache[col + 5]; float bx5 = zero * x_cache[col + 5];
        float sx6 = scale * x_cache[col + 6]; float bx6 = zero * x_cache[col + 6];
        float sx7 = scale * x_cache[col + 7]; float bx7 = zero * x_cache[col + 7];

        sum = fma(float((pack      ) & 0xF), sx0, sum) + bx0;
        sum = fma(float((pack >>  4) & 0xF), sx1, sum) + bx1;
        sum = fma(float((pack >>  8) & 0xF), sx2, sum) + bx2;
        sum = fma(float((pack >> 12) & 0xF), sx3, sum) + bx3;
        sum = fma(float((pack >> 16) & 0xF), sx4, sum) + bx4;
        sum = fma(float((pack >> 20) & 0xF), sx5, sum) + bx5;
        sum = fma(float((pack >> 24) & 0xF), sx6, sum) + bx6;
        sum = fma(float((pack >> 28) & 0xF), sx7, sum) + bx7;
    }

    // SIMD-group reduction
    sum = simd_sum(sum);

    // Cross-simdgroup reduction via shared memory
    threadgroup float partial[8];
    uint sg_id = tid / 32;

    if (tid % 32 == 0) {
        partial[sg_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0;
        uint num_sgs = min(TG / 32, 8u);
        for (uint i = 0; i < num_sgs; i++) {
            total += partial[i];
        }
        y[row] = total;
    }
}
"#;

/// V2 shader: ROWS_PER_TG=8 — 8 SIMD groups share x_cache, each handles one row.
/// Reduces threadgroup count 8x, amortizes x_cache load across 8 rows.
/// Needs out_features buffer for bounds checking last threadgroup.
const DEQUANT_GEMV_SHADER_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dequant_4bit_gemv_v2(
    device const uint32_t* packed   [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const half* zeros        [[buffer(2)]],
    device const float* x           [[buffer(3)]],
    device float* y                 [[buffer(4)]],
    constant uint& in_features      [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& out_features     [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint TG = 256;
    const uint THREADS_PER_ROW = TG / ROWS_PER_TG; // 32 = one SIMD group

    const uint packed_per_row = in_features / 8;
    const uint num_groups = in_features / group_size;
    const uint groups_per_8 = group_size / 8;

    // Read x directly from device memory (L2-cached after first TG reads it).
    // Removes 16 KB threadgroup memory → allows 4+ concurrent TGs per EU
    // instead of 2, doubling occupancy for better latency hiding.

    uint local_row = tid / THREADS_PER_ROW;  // 0..7
    uint lane = tid % THREADS_PER_ROW;       // 0..31 (SIMD lane)
    uint row = tgid * ROWS_PER_TG + local_row;

    if (row >= out_features) return;

    float sum = 0.0;
    device const uint32_t* row_packed = packed + row * packed_per_row;

    for (uint pi = lane; pi < packed_per_row; pi += THREADS_PER_ROW) {
        uint col = pi * 8;
        uint group_idx = row * num_groups + pi / groups_per_8;
        float scale = float(scales[group_idx]);
        float zero = float(zeros[group_idx]);

        uint32_t pack = row_packed[pi];

        float sx0 = scale * x[col    ]; float bx0 = zero * x[col    ];
        float sx1 = scale * x[col + 1]; float bx1 = zero * x[col + 1];
        float sx2 = scale * x[col + 2]; float bx2 = zero * x[col + 2];
        float sx3 = scale * x[col + 3]; float bx3 = zero * x[col + 3];
        float sx4 = scale * x[col + 4]; float bx4 = zero * x[col + 4];
        float sx5 = scale * x[col + 5]; float bx5 = zero * x[col + 5];
        float sx6 = scale * x[col + 6]; float bx6 = zero * x[col + 6];
        float sx7 = scale * x[col + 7]; float bx7 = zero * x[col + 7];

        sum = fma(float((pack      ) & 0xF), sx0, sum) + bx0;
        sum = fma(float((pack >>  4) & 0xF), sx1, sum) + bx1;
        sum = fma(float((pack >>  8) & 0xF), sx2, sum) + bx2;
        sum = fma(float((pack >> 12) & 0xF), sx3, sum) + bx3;
        sum = fma(float((pack >> 16) & 0xF), sx4, sum) + bx4;
        sum = fma(float((pack >> 20) & 0xF), sx5, sum) + bx5;
        sum = fma(float((pack >> 24) & 0xF), sx6, sum) + bx6;
        sum = fma(float((pack >> 28) & 0xF), sx7, sum) + bx7;
    }

    // SIMD reduction (within 32-thread SIMD group — no cross-SIMD needed)
    sum = simd_sum(sum);

    if (lane == 0) {
        y[row] = sum;
    }
}
"#;

/// Fused gate+up+SiLU kernel: computes SiLU(gate @ x) * (up @ x) in one pass.
/// Eliminates GPU→CPU→GPU roundtrip for SiLU. Uses ROWS_PER_TG=8.
/// gate and up weights are at separate offsets in the same mmap buffer.
const FUSED_GATE_UP_SILU_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_gate_up_silu(
    device const uint32_t* gate_packed [[buffer(0)]],
    device const half* gate_scales     [[buffer(1)]],
    device const half* gate_zeros      [[buffer(2)]],
    device const uint32_t* up_packed   [[buffer(3)]],
    device const half* up_scales       [[buffer(4)]],
    device const half* up_zeros        [[buffer(5)]],
    device const float* x              [[buffer(6)]],
    device float* y                    [[buffer(7)]],
    constant uint& in_features         [[buffer(8)]],
    constant uint& group_size          [[buffer(9)]],
    constant uint& out_features        [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint TG = 256;
    const uint THREADS_PER_ROW = TG / ROWS_PER_TG; // 32

    const uint packed_per_row = in_features / 8;
    const uint num_groups = in_features / group_size;
    const uint groups_per_8 = group_size / 8;

    // Read x directly from device memory (L2-cached after first TG reads it).
    // Removes 14 KB threadgroup memory → allows 4+ concurrent TGs per EU
    // instead of 2, doubling occupancy for better latency hiding.
    // Also eliminates f32→f16 precision loss from x_cache.

    uint local_row = tid / THREADS_PER_ROW;
    uint lane = tid % THREADS_PER_ROW;
    uint row = tgid * ROWS_PER_TG + local_row;

    if (row >= out_features) return;

    float gate_sum = 0.0;
    float up_sum = 0.0;

    device const uint32_t* gate_row_packed = gate_packed + row * packed_per_row;
    device const uint32_t* up_row_packed = up_packed + row * packed_per_row;

    for (uint pi = lane; pi < packed_per_row; pi += THREADS_PER_ROW) {
        uint col = pi * 8;
        uint group_idx = row * num_groups + pi / groups_per_8;

        float g_scale = float(gate_scales[group_idx]);
        float g_zero = float(gate_zeros[group_idx]);
        float u_scale = float(up_scales[group_idx]);
        float u_zero = float(up_zeros[group_idx]);

        uint32_t g_pack = gate_row_packed[pi];
        uint32_t u_pack = up_row_packed[pi];

        // Pre-factor x values (shared between gate and up)
        float xv0 = x[col    ]; float xv1 = x[col + 1];
        float xv2 = x[col + 2]; float xv3 = x[col + 3];
        float xv4 = x[col + 4]; float xv5 = x[col + 5];
        float xv6 = x[col + 6]; float xv7 = x[col + 7];

        // Gate GEMV
        float gsx0 = g_scale * xv0; float gbx0 = g_zero * xv0;
        float gsx1 = g_scale * xv1; float gbx1 = g_zero * xv1;
        float gsx2 = g_scale * xv2; float gbx2 = g_zero * xv2;
        float gsx3 = g_scale * xv3; float gbx3 = g_zero * xv3;
        float gsx4 = g_scale * xv4; float gbx4 = g_zero * xv4;
        float gsx5 = g_scale * xv5; float gbx5 = g_zero * xv5;
        float gsx6 = g_scale * xv6; float gbx6 = g_zero * xv6;
        float gsx7 = g_scale * xv7; float gbx7 = g_zero * xv7;

        gate_sum = fma(float((g_pack      ) & 0xF), gsx0, gate_sum) + gbx0;
        gate_sum = fma(float((g_pack >>  4) & 0xF), gsx1, gate_sum) + gbx1;
        gate_sum = fma(float((g_pack >>  8) & 0xF), gsx2, gate_sum) + gbx2;
        gate_sum = fma(float((g_pack >> 12) & 0xF), gsx3, gate_sum) + gbx3;
        gate_sum = fma(float((g_pack >> 16) & 0xF), gsx4, gate_sum) + gbx4;
        gate_sum = fma(float((g_pack >> 20) & 0xF), gsx5, gate_sum) + gbx5;
        gate_sum = fma(float((g_pack >> 24) & 0xF), gsx6, gate_sum) + gbx6;
        gate_sum = fma(float((g_pack >> 28) & 0xF), gsx7, gate_sum) + gbx7;

        // Up GEMV
        float usx0 = u_scale * xv0; float ubx0 = u_zero * xv0;
        float usx1 = u_scale * xv1; float ubx1 = u_zero * xv1;
        float usx2 = u_scale * xv2; float ubx2 = u_zero * xv2;
        float usx3 = u_scale * xv3; float ubx3 = u_zero * xv3;
        float usx4 = u_scale * xv4; float ubx4 = u_zero * xv4;
        float usx5 = u_scale * xv5; float ubx5 = u_zero * xv5;
        float usx6 = u_scale * xv6; float ubx6 = u_zero * xv6;
        float usx7 = u_scale * xv7; float ubx7 = u_zero * xv7;

        up_sum = fma(float((u_pack      ) & 0xF), usx0, up_sum) + ubx0;
        up_sum = fma(float((u_pack >>  4) & 0xF), usx1, up_sum) + ubx1;
        up_sum = fma(float((u_pack >>  8) & 0xF), usx2, up_sum) + ubx2;
        up_sum = fma(float((u_pack >> 12) & 0xF), usx3, up_sum) + ubx3;
        up_sum = fma(float((u_pack >> 16) & 0xF), usx4, up_sum) + ubx4;
        up_sum = fma(float((u_pack >> 20) & 0xF), usx5, up_sum) + ubx5;
        up_sum = fma(float((u_pack >> 24) & 0xF), usx6, up_sum) + ubx6;
        up_sum = fma(float((u_pack >> 28) & 0xF), usx7, up_sum) + ubx7;
    }

    // SIMD reduction
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);

    // SiLU(gate) * up — applied inline on lane 0
    if (lane == 0) {
        float silu_gate = gate_sum / (1.0 + exp(-gate_sum));
        y[row] = silu_gate * up_sum;
    }
}
"#;

/// Simple f32 GEMV shader: y = W @ x where W is f32 row-major.
/// Uses ROWS_PER_TG=8 pattern with threadgroup x_cache for bandwidth efficiency.
/// For MLA o_proj dispatch on GPU (replacing CPU sgemv during GPU-idle MLA phase).
const F32_GEMV_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void f32_gemv(
    device const float* W              [[buffer(0)]],
    device const float* x              [[buffer(1)]],
    device float* y                    [[buffer(2)]],
    constant uint& in_features         [[buffer(3)]],
    constant uint& out_features        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint TG = 256;
    const uint THREADS_PER_ROW = TG / ROWS_PER_TG; // 32

    // Cache x in shared memory
    threadgroup float x_cache[8192]; // max in_features = 8192 for o_proj
    for (uint i = tid; i < in_features; i += TG) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_row = tid / THREADS_PER_ROW;
    uint lane = tid % THREADS_PER_ROW;
    uint row = tgid * ROWS_PER_TG + local_row;

    if (row >= out_features) return;

    device const float* row_w = W + row * in_features;
    float sum = 0.0;

    for (uint col = lane; col < in_features; col += THREADS_PER_ROW) {
        sum = fma(row_w[col], x_cache[col], sum);
    }

    // SIMD reduction (THREADS_PER_ROW=32 = one SIMD group)
    sum = simd_sum(sum);

    if (lane == 0) {
        y[row] = sum;
    }
}
"#;

/// Expert gate+up+SiLU fused operation layout.
#[derive(Clone)]
pub struct FusedGateUpSiluOp {
    pub gate_packed_offset: usize,
    pub gate_scales_offset: usize,
    pub gate_zeros_offset: usize,
    pub up_packed_offset: usize,
    pub up_scales_offset: usize,
    pub up_zeros_offset: usize,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

/// Expert GEMV layout info for zero-copy offset-based dispatch.
#[derive(Clone)]
pub struct ExpertGemvOp {
    /// Byte offset into mmap buffer for packed data
    pub packed_offset: usize,
    /// Byte offset into mmap buffer for scales
    pub scales_offset: usize,
    /// Byte offset into mmap buffer for zeros
    pub zeros_offset: usize,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

/// Metal-accelerated fused 4-bit dequant + GEMV.
/// Compile once, reuse for all inference steps.
pub struct MetalDequantGemv {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// V2: ROWS_PER_TG=8 shader (8 SIMD groups share x_cache).
    pipeline_v2: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused gate+up+SiLU shader.
    pipeline_fused: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// f32 GEMV shader (for MLA o_proj on GPU).
    pipeline_f32_gemv: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Pre-allocated scratch: x input buffer (max of hidden, moe_inter).
    scratch_x: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Pre-allocated scratch: fused output / down input (one per top-k expert).
    scratch_activated: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Pre-allocated scratch: down output (one per top-k expert).
    scratch_down_y: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Cached constant u32 Metal buffers (key = value, e.g. 7168, 2048, 128).
    /// Pre-populated in init_scratch to avoid per-dispatch Metal buffer allocation.
    const_u32_bufs: std::collections::HashMap<u32, Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl MetalDequantGemv {
    /// Create and compile the dequant GEMV pipelines (v1 + v2).
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;

        // V1 pipeline (1 row per threadgroup)
        let source = objc2_foundation::NSString::from_str(DEQUANT_GEMV_SHADER);
        let library = device.newLibraryWithSource_options_error(&source, None).ok()?;
        let fn_name = objc2_foundation::NSString::from_str("dequant_4bit_gemv");
        let function = library.newFunctionWithName(&fn_name)?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;

        // V2 pipeline (8 rows per threadgroup)
        let source_v2 = objc2_foundation::NSString::from_str(DEQUANT_GEMV_SHADER_V2);
        let library_v2 = device.newLibraryWithSource_options_error(&source_v2, None).ok()?;
        let fn_name_v2 = objc2_foundation::NSString::from_str("dequant_4bit_gemv_v2");
        let function_v2 = library_v2.newFunctionWithName(&fn_name_v2)?;
        let pipeline_v2 = device
            .newComputePipelineStateWithFunction_error(&function_v2)
            .ok()?;

        // Fused gate+up+SiLU pipeline
        let source_fused = objc2_foundation::NSString::from_str(FUSED_GATE_UP_SILU_SHADER);
        let library_fused = device.newLibraryWithSource_options_error(&source_fused, None).ok()?;
        let fn_name_fused = objc2_foundation::NSString::from_str("fused_gate_up_silu");
        let function_fused = library_fused.newFunctionWithName(&fn_name_fused)?;
        let pipeline_fused = device
            .newComputePipelineStateWithFunction_error(&function_fused)
            .ok()?;

        // f32 GEMV pipeline (MLA o_proj)
        let source_f32 = objc2_foundation::NSString::from_str(F32_GEMV_SHADER);
        let library_f32 = device.newLibraryWithSource_options_error(&source_f32, None).ok()?;
        let fn_name_f32 = objc2_foundation::NSString::from_str("f32_gemv");
        let function_f32 = library_f32.newFunctionWithName(&fn_name_f32)?;
        let pipeline_f32_gemv = device
            .newComputePipelineStateWithFunction_error(&function_f32)
            .ok()?;

        Some(Self {
            device, queue, pipeline, pipeline_v2, pipeline_fused, pipeline_f32_gemv,
            scratch_x: None,
            scratch_activated: Vec::new(),
            scratch_down_y: Vec::new(),
            const_u32_bufs: std::collections::HashMap::new(),
        })
    }

    /// Pre-allocate scratch buffers for zero-allocation inference hot path.
    /// Call once at model load after knowing hidden/moe_inter/top_k.
    pub fn init_scratch(&mut self, hidden: usize, moe_inter: usize, top_k: usize, group_size: usize) {
        let x_size = hidden.max(moe_inter) * 4; // f32
        self.scratch_x = Some(
            self.device.newBufferWithLength_options(x_size, MTLResourceOptions::StorageModeShared)
                .expect("scratch_x")
        );
        self.scratch_activated = (0..top_k).map(|_| {
            self.device.newBufferWithLength_options(moe_inter * 4, MTLResourceOptions::StorageModeShared)
                .expect("scratch_activated")
        }).collect();
        self.scratch_down_y = (0..top_k).map(|_| {
            self.device.newBufferWithLength_options(hidden * 4, MTLResourceOptions::StorageModeShared)
                .expect("scratch_down_y")
        }).collect();

        // Pre-cache constant u32 Metal buffers for dispatch.
        // Eliminates ~360 Metal buffer allocations per token (6 per layer × 60 layers).
        self.const_u32_bufs.clear();
        for val in [hidden as u32, moe_inter as u32, group_size as u32] {
            if !self.const_u32_bufs.contains_key(&val) {
                let buf = self.u32_buffer(val);
                self.const_u32_bufs.insert(val, buf);
            }
        }
    }

    /// Look up a cached constant u32 buffer. Panics if not pre-cached in init_scratch.
    fn get_const_buf(&self, val: u32) -> &ProtocolObject<dyn MTLBuffer> {
        self.const_u32_bufs.get(&val)
            .map(|b| &**b)
            .unwrap_or_else(|| panic!("uncached u32 constant {val} — call init_scratch with correct params"))
    }

    /// Combined fused+down dispatch using scratch buffers, single command buffer.
    /// Fused gate+up+SiLU writes to scratch_activated, down reads from it.
    /// Returns down outputs (one Vec<f32> per expert).
    pub fn fused_and_down_single_cmdbuf(
        &self,
        mmap_buf: &ProtocolObject<dyn MTLBuffer>,
        fused_ops: &[FusedGateUpSiluOp],
        down_ops: &[ExpertGemvOp],
        x: &[f32],
    ) -> Vec<Vec<f32>> {
        let n = fused_ops.len();
        assert_eq!(n, down_ops.len());
        if n == 0 { return vec![]; }

        let scratch_x = self.scratch_x.as_ref().expect("call init_scratch first");

        // Copy x into scratch_x
        unsafe {
            let dst = scratch_x.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), dst, x.len());
        }

        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");

        // Phase 1: Fused gate+up+SiLU → scratch_activated[i]
        enc.setComputePipelineState(&self.pipeline_fused);
        for (i, op) in fused_ops.iter().enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);

            unsafe {
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_packed_offset, 3);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_scales_offset, 4);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_zeros_offset, 5);
                enc.setBuffer_offset_atIndex(Some(scratch_x), 0, 6);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_activated[i]), 0, 7);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 8);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 9);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 10);

                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
                let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
        }

        // Phase 2: Down GEMV — reads from scratch_activated[i], writes to scratch_down_y[i]
        enc.setComputePipelineState(&self.pipeline_v2);
        for (i, op) in down_ops.iter().enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);

            unsafe {
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_activated[i]), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_down_y[i]), 0, 4);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 5);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 6);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 7);

                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
                let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read results from scratch_down_y
        down_ops.iter().enumerate().map(|(i, op)| {
            let mut out = vec![0.0f32; op.out_features];
            unsafe {
                let src = self.scratch_down_y[i].contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), op.out_features);
            }
            out
        }).collect()
    }

    /// Dispatch fused phase only — commits to GPU, returns immediately.
    /// Call dispatch_down_phase_and_wait() after to complete the pipeline.
    pub fn dispatch_fused_phase(
        &self,
        mmap_buf: &ProtocolObject<dyn MTLBuffer>,
        fused_ops: &[FusedGateUpSiluOp],
        x: &[f32],
    ) {
        if fused_ops.is_empty() { return; }
        let scratch_x = self.scratch_x.as_ref().expect("call init_scratch first");
        unsafe {
            let dst = scratch_x.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), dst, x.len());
        }
        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.pipeline_fused);
        for (i, op) in fused_ops.iter().enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_packed_offset, 3);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_scales_offset, 4);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_zeros_offset, 5);
                enc.setBuffer_offset_atIndex(Some(scratch_x), 0, 6);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_activated[i]), 0, 7);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 8);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 9);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 10);
                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: num_tgs, height: 1, depth: 1 },
                    MTLSize { width: TG_SIZE, height: 1, depth: 1 },
                );
            }
        }
        enc.endEncoding();
        cmd.commit(); // non-blocking — GPU starts immediately
    }

    /// Dispatch down phase and wait for completion. Returns down outputs.
    /// Must be called after dispatch_fused_phase on the same queue (ordering guaranteed).
    pub fn dispatch_down_phase_and_wait(
        &self,
        mmap_buf: &ProtocolObject<dyn MTLBuffer>,
        down_ops: &[ExpertGemvOp],
    ) -> Vec<Vec<f32>> {
        if down_ops.is_empty() { return vec![]; }
        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.pipeline_v2);
        for (i, op) in down_ops.iter().enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_activated[i]), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_down_y[i]), 0, 4);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 5);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 6);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 7);
                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: num_tgs, height: 1, depth: 1 },
                    MTLSize { width: TG_SIZE, height: 1, depth: 1 },
                );
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        down_ops.iter().enumerate().map(|(i, op)| {
            let mut out = vec![0.0f32; op.out_features];
            unsafe {
                let src = self.scratch_down_y[i].contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), op.out_features);
            }
            out
        }).collect()
    }

    /// Compute y = packed_weights @ x on the GPU.
    pub fn gemv(&self, weights: &PackedWeights4Bit, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), weights.in_features);

        let out_n = weights.out_features;
        let mut y = vec![0.0f32; out_n];

        let (packed_buf, scales_buf, zeros_buf, x_buf, y_buf, in_feat_buf, group_buf) =
            self.create_buffers(weights, x, &y);

        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");

        self.encode_dispatch(
            &enc, &packed_buf, &scales_buf, &zeros_buf,
            &x_buf, &y_buf, &in_feat_buf, &group_buf, out_n,
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back output
        unsafe {
            let src = y_buf.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, y.as_mut_ptr(), out_n);
        }

        y
    }

    /// Compute y = packed_weights @ x, returning elapsed time in seconds.
    pub fn gemv_timed(&self, weights: &PackedWeights4Bit, x: &[f32]) -> (Vec<f32>, f64) {
        assert_eq!(x.len(), weights.in_features);

        let out_n = weights.out_features;
        let mut y = vec![0.0f32; out_n];

        let (packed_buf, scales_buf, zeros_buf, x_buf, y_buf, in_feat_buf, group_buf) =
            self.create_buffers(weights, x, &y);

        // Warmup
        {
            let cmd = self.queue.commandBuffer().expect("command buffer");
            let enc = cmd.computeCommandEncoder().expect("compute encoder");
            self.encode_dispatch(
                &enc, &packed_buf, &scales_buf, &zeros_buf,
                &x_buf, &y_buf, &in_feat_buf, &group_buf, out_n,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        // Timed dispatch
        let t0 = std::time::Instant::now();
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");

        self.encode_dispatch(
            &enc, &packed_buf, &scales_buf, &zeros_buf,
            &x_buf, &y_buf, &in_feat_buf, &group_buf, out_n,
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = t0.elapsed().as_secs_f64();

        unsafe {
            let src = y_buf.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, y.as_mut_ptr(), out_n);
        }

        (y, elapsed)
    }

    /// GPU f32 GEMV: y = W @ x where W is f32 [out_features, in_features].
    /// Uses Metal GPU bandwidth (~400 GB/s) instead of CPU (~200 GB/s).
    /// `w_buf` is a pre-populated Metal shared buffer containing the f32 weights.
    pub fn gemv_f32_gpu(
        &self,
        w_buf: &ProtocolObject<dyn MTLBuffer>,
        x: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        let scratch_x = self.scratch_x.as_ref().expect("call init_scratch first");

        // Copy x into scratch_x (reuse existing scratch buffer)
        unsafe {
            let dst = scratch_x.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), dst, x.len());
        }

        // Output buffer (reuse scratch_down_y[0] if big enough, else allocate)
        let y_buf = if !self.scratch_down_y.is_empty()
            && self.scratch_down_y[0].length() >= out_features * 4
        {
            &self.scratch_down_y[0]
        } else {
            // Fallback: this shouldn't happen in normal inference
            panic!("scratch_down_y not large enough for f32 GEMV output");
        };

        let in_buf = self.get_const_buf(in_features as u32);
        let out_buf = self.get_const_buf(out_features as u32);

        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.pipeline_f32_gemv);

        unsafe {
            enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(scratch_x), 0, 1);
            enc.setBuffer_offset_atIndex(Some(y_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(in_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 4);

            let num_tgs = (out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
            let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
            let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
            enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let mut result = vec![0.0f32; out_features];
        unsafe {
            let src = y_buf.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, result.as_mut_ptr(), out_features);
        }
        result
    }

    /// Encode a GEMV dispatch into an existing command encoder (no commit).
    /// Caller manages command buffer lifecycle for batching.
    /// Returns the output buffer for reading results after commit.
    pub fn encode_into(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &PackedWeights4Bit,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let packed_buf = self.make_buffer(
            weights.data.as_ptr() as *const u8,
            weights.data.len() * 4,
        );
        let scales_buf = self.make_buffer(
            weights.scales.as_ptr() as *const u8,
            weights.scales.len() * 2,
        );
        let zeros_buf = self.make_buffer(
            weights.zeros.as_ptr() as *const u8,
            weights.zeros.len() * 2,
        );
        let in_feat_buf = self.u32_buffer(weights.in_features as u32);
        let group_buf = self.u32_buffer(weights.group_size as u32);

        self.encode_dispatch(
            enc, &packed_buf, &scales_buf, &zeros_buf,
            x_buf, y_buf, &in_feat_buf, &group_buf,
            weights.out_features,
        );
    }

    /// Wrap an mmap'd byte slice as a zero-copy Metal buffer.
    /// Falls back to copy-based buffer if alignment requirements aren't met.
    pub fn wrap_mmap(&self, data: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        const PAGE_SIZE: usize = 16384; // Apple Silicon uses 16KB pages
        let ptr = data.as_ptr() as usize;
        let len = data.len();
        // Round up length to page boundary for Metal
        let padded_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        if ptr % PAGE_SIZE == 0 && padded_len >= PAGE_SIZE {
            // Zero-copy: wrap existing memory
            if let Some(buf) = unsafe {
                self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(data.as_ptr() as *mut c_void).expect("non-null"),
                    padded_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            } {
                return buf;
            }
        }
        // Fallback: copy-based
        self.make_buffer(data.as_ptr() as *const u8, len)
    }

    /// Like fused_and_down_single_cmdbuf but each expert reads from its OWN Metal buffer.
    /// Used for zero-copy ExpertPool: each pool slot IS a Metal buffer, no staging copy needed.
    pub fn fused_and_down_per_expert_bufs(
        &self,
        expert_bufs: &[&ProtocolObject<dyn MTLBuffer>],
        fused_ops: &[FusedGateUpSiluOp],
        down_ops: &[ExpertGemvOp],
        x: &[f32],
    ) -> Vec<Vec<f32>> {
        let n = fused_ops.len();
        assert_eq!(n, down_ops.len());
        assert_eq!(n, expert_bufs.len());
        if n == 0 { return vec![]; }

        let scratch_x = self.scratch_x.as_ref().expect("call init_scratch first");
        unsafe {
            let dst = scratch_x.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), dst, x.len());
        }

        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");

        // Phase 1: Fused gate+up+SiLU — each expert from its own buffer
        enc.setComputePipelineState(&self.pipeline_fused);
        for (i, (op, buf)) in fused_ops.iter().zip(expert_bufs.iter()).enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);

            unsafe {
                // Offsets are relative to THIS expert's buffer (start at 0 for pool slots)
                enc.setBuffer_offset_atIndex(Some(*buf), op.gate_packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(*buf), op.gate_scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(*buf), op.gate_zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(*buf), op.up_packed_offset, 3);
                enc.setBuffer_offset_atIndex(Some(*buf), op.up_scales_offset, 4);
                enc.setBuffer_offset_atIndex(Some(*buf), op.up_zeros_offset, 5);
                enc.setBuffer_offset_atIndex(Some(scratch_x), 0, 6);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_activated[i]), 0, 7);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 8);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 9);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 10);

                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
                let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
        }

        // Phase 2: Down GEMV
        enc.setComputePipelineState(&self.pipeline_v2);
        for (i, (op, buf)) in down_ops.iter().zip(expert_bufs.iter()).enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);

            unsafe {
                enc.setBuffer_offset_atIndex(Some(*buf), op.packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(*buf), op.scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(*buf), op.zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_activated[i]), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.scratch_down_y[i]), 0, 4);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 5);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 6);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 7);

                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
                let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        down_ops.iter().enumerate().map(|(i, op)| {
            let mut out = vec![0.0f32; op.out_features];
            unsafe {
                let src = self.scratch_down_y[i].contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), op.out_features);
            }
            out
        }).collect()
    }

    /// Batched GEMV using pre-wrapped mmap buffer (zero-copy weights).
    /// All expert weights come from `mmap_buf` at different offsets.
    /// Deduplicates x_bufs and u32 constant buffers for minimal allocation overhead.
    pub fn batch_gemv_mmap(
        &self,
        mmap_buf: &ProtocolObject<dyn MTLBuffer>,
        ops: &[ExpertGemvOp],
        x_slices: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        if ops.is_empty() { return vec![]; }
        assert_eq!(ops.len(), x_slices.len());

        // Deduplicate x_bufs by pointer identity (gate+up share same x)
        let mut x_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
        let mut x_buf_map: Vec<(*const f32, usize)> = Vec::new(); // (ptr, buf_index)
        let mut x_buf_indices: Vec<usize> = Vec::with_capacity(ops.len());

        for x in x_slices {
            let ptr = x.as_ptr();
            if let Some(&(_, idx)) = x_buf_map.iter().find(|&&(p, _)| p == ptr) {
                x_buf_indices.push(idx);
            } else {
                let idx = x_bufs.len();
                x_bufs.push(self.make_buffer(ptr as *const u8, x.len() * 4));
                x_buf_map.push((ptr, idx));
                x_buf_indices.push(idx);
            }
        }

        // Create y buffers (one per op, small)
        let y_bufs: Vec<_> = ops.iter().map(|op| {
            self.device
                .newBufferWithLength_options(
                    op.out_features * 4,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("y_buf")
        }).collect();

        // Encode all dispatches using V2 shader (ROWS_PER_TG=8)
        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");

        enc.setComputePipelineState(&self.pipeline_v2);

        for (i, op) in ops.iter().enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);

            unsafe {
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(&x_bufs[x_buf_indices[i]]), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&y_bufs[i]), 0, 4);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 5);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 6);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 7);

                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
                let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back results
        ops.iter().enumerate().map(|(i, op)| {
            let mut out = vec![0.0f32; op.out_features];
            unsafe {
                let src = y_bufs[i].contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), op.out_features);
            }
            out
        }).collect()
    }

    /// Batched fused gate+up+SiLU: computes SiLU(gate@x) * (up@x) for each expert.
    /// Returns one activated vector per expert (ready for down GEMV).
    /// Uses single cmd_buf for all experts.
    pub fn batch_fused_gate_up_silu_mmap(
        &self,
        mmap_buf: &ProtocolObject<dyn MTLBuffer>,
        ops: &[FusedGateUpSiluOp],
        x_slices: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        if ops.is_empty() { return vec![]; }
        assert_eq!(ops.len(), x_slices.len());

        // Deduplicate x_bufs
        let mut x_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
        let mut x_buf_map: Vec<(*const f32, usize)> = Vec::new();
        let mut x_buf_indices: Vec<usize> = Vec::with_capacity(ops.len());

        for x in x_slices {
            let ptr = x.as_ptr();
            if let Some(&(_, idx)) = x_buf_map.iter().find(|&&(p, _)| p == ptr) {
                x_buf_indices.push(idx);
            } else {
                let idx = x_bufs.len();
                x_bufs.push(self.make_buffer(ptr as *const u8, x.len() * 4));
                x_buf_map.push((ptr, idx));
                x_buf_indices.push(idx);
            }
        }

        // Create y buffers
        let y_bufs: Vec<_> = ops.iter().map(|op| {
            self.device
                .newBufferWithLength_options(
                    op.out_features * 4,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("y_buf")
        }).collect();

        const ROWS_PER_TG: usize = 8;
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");

        enc.setComputePipelineState(&self.pipeline_fused);

        for (i, op) in ops.iter().enumerate() {
            let ifb = self.get_const_buf(op.in_features as u32);
            let gb = self.get_const_buf(op.group_size as u32);
            let ofb = self.get_const_buf(op.out_features as u32);

            unsafe {
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_packed_offset, 0);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_scales_offset, 1);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.gate_zeros_offset, 2);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_packed_offset, 3);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_scales_offset, 4);
                enc.setBuffer_offset_atIndex(Some(mmap_buf), op.up_zeros_offset, 5);
                enc.setBuffer_offset_atIndex(Some(&x_bufs[x_buf_indices[i]]), 0, 6);
                enc.setBuffer_offset_atIndex(Some(&y_bufs[i]), 0, 7);
                enc.setBuffer_offset_atIndex(Some(ifb), 0, 8);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 9);
                enc.setBuffer_offset_atIndex(Some(ofb), 0, 10);

                let num_tgs = (op.out_features + ROWS_PER_TG - 1) / ROWS_PER_TG;
                let threadgroups = MTLSize { width: num_tgs, height: 1, depth: 1 };
                let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        ops.iter().enumerate().map(|(i, op)| {
            let mut out = vec![0.0f32; op.out_features];
            unsafe {
                let src = y_bufs[i].contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), op.out_features);
            }
            out
        }).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dispatch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        packed_buf: &ProtocolObject<dyn MTLBuffer>,
        scales_buf: &ProtocolObject<dyn MTLBuffer>,
        zeros_buf: &ProtocolObject<dyn MTLBuffer>,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        in_feat_buf: &ProtocolObject<dyn MTLBuffer>,
        group_buf: &ProtocolObject<dyn MTLBuffer>,
        out_features: usize,
    ) {
        unsafe {
            enc.setComputePipelineState(&self.pipeline);
            enc.setBuffer_offset_atIndex(Some(packed_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(scales_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(zeros_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(y_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(in_feat_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(group_buf), 0, 6);

            // One threadgroup per output row, TG_SIZE threads per group
            let threadgroups = MTLSize { width: out_features, height: 1, depth: 1 };
            let threads_per_tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };
            enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        }
    }

    fn create_buffers(
        &self,
        weights: &PackedWeights4Bit,
        x: &[f32],
        y: &[f32],
    ) -> (
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
    ) {
        let packed_buf = self.make_buffer(
            weights.data.as_ptr() as *const u8,
            weights.data.len() * 4,
        );
        let scales_buf = self.make_buffer(
            weights.scales.as_ptr() as *const u8,
            weights.scales.len() * 2,
        );
        let zeros_buf = self.make_buffer(
            weights.zeros.as_ptr() as *const u8,
            weights.zeros.len() * 2,
        );
        let x_buf = self.make_buffer(x.as_ptr() as *const u8, x.len() * 4);
        let y_buf = self.make_buffer(y.as_ptr() as *const u8, y.len() * 4);

        let in_feat_buf = self.u32_buffer(weights.in_features as u32);
        let group_buf = self.u32_buffer(weights.group_size as u32);

        (packed_buf, scales_buf, zeros_buf, x_buf, y_buf, in_feat_buf, group_buf)
    }

    fn make_buffer(
        &self,
        ptr: *const u8,
        byte_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    NonNull::new(ptr as *mut c_void).expect("non-null"),
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("make_buffer")
        }
    }

    fn u32_buffer(&self, val: u32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    NonNull::new(&val as *const u32 as *mut c_void).expect("non-null"),
                    4,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("u32_buffer")
        }
    }
}
