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

/// Metal-accelerated fused 4-bit dequant + GEMV.
/// Compile once, reuse for all inference steps.
pub struct MetalDequantGemv {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalDequantGemv {
    /// Create and compile the dequant GEMV pipeline.
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;

        let source = objc2_foundation::NSString::from_str(DEQUANT_GEMV_SHADER);
        let library = device.newLibraryWithSource_options_error(&source, None).ok()?;
        let fn_name = objc2_foundation::NSString::from_str("dequant_4bit_gemv");
        let function = library.newFunctionWithName(&fn_name)?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;

        Some(Self { device, queue, pipeline })
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
