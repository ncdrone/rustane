//! Metal GPU-accelerated AdamW optimizer.
//!
//! Batches all parameter updates into a single command buffer dispatch.
//! Uses zero-copy buffers for page-aligned data (all weight matrices).
//! ~6ms for 48.8M params vs ~53ms on CPU.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::ffi::c_void;

const ADAM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void adam_step(
    device float* param     [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m          [[buffer(2)]],
    device float* v          [[buffer(3)]],
    constant float& lr       [[buffer(4)]],
    constant float& beta1    [[buffer(5)]],
    constant float& beta2    [[buffer(6)]],
    constant float& eps      [[buffer(7)]],
    constant float& wd       [[buffer(8)]],
    constant float& bc1      [[buffer(9)]],
    constant float& bc2      [[buffer(10)]],
    constant float& gscale   [[buffer(11)]],
    uint id [[thread_position_in_grid]]
) {
    float g = grad[id] * gscale;
    float mi = beta1 * m[id] + (1.0 - beta1) * g;
    float vi = beta2 * v[id] + (1.0 - beta2) * g * g;
    m[id] = mi;
    v[id] = vi;
    float m_hat = mi * bc1;
    float v_hat = vi * bc2;
    float update = m_hat / (sqrt(v_hat) + eps);
    param[id] -= lr * (update + wd * param[id]);
}
"#;

const PAGE_SIZE: usize = 16384;

/// Metal-accelerated AdamW optimizer.
/// Compile once, reuse for all training steps.
pub struct MetalAdam {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Tracks a copy-based Metal buffer that needs readback after GPU execution.
struct Readback {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dst: *mut f32,
    len: usize,
}

/// A batch of Adam updates encoded into a single command buffer.
/// Shared scalar buffers (beta1, beta2, eps, bc1, bc2, grad_scale) are created once
/// per batch in `begin_batch()` and reused for every `add()` call.
/// Per-call buffers (lr, weight_decay) are still created per-tensor since they vary.
pub struct AdamBatch<'a> {
    adam: &'a MetalAdam,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    readbacks: Vec<Readback>,
    // Hold zero-copy buffers alive until commit+wait completes
    _held: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Shared scalar buffers: same for every tensor in this batch
    b1_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    b2_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    bc1_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    bc2_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gs_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalAdam {
    /// Create a new Metal Adam optimizer, compiling the shader.
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;

        let source = objc2_foundation::NSString::from_str(ADAM_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .ok()?;
        let fn_name = objc2_foundation::NSString::from_str("adam_step");
        let function = library.newFunctionWithName(&fn_name)?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;

        Some(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Start a batch of Adam updates. Encode all dispatches, then call `execute()`.
    /// Pass hyperparams that are constant across all tensors in the batch.
    /// Creates 6 shared scalar buffers once; `add()` only creates lr+wd per call.
    pub fn begin_batch(
        &self,
        t: u32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        grad_scale: f32,
    ) -> AdamBatch<'_> {
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        let bc1 = 1.0f32 / (1.0 - beta1.powi(t as i32));
        let bc2 = 1.0f32 / (1.0 - beta2.powi(t as i32));
        AdamBatch {
            adam: self,
            cmd,
            enc,
            readbacks: Vec::new(),
            _held: Vec::new(),
            b1_buf: self.scalar_buffer(beta1),
            b2_buf: self.scalar_buffer(beta2),
            eps_buf: self.scalar_buffer(eps),
            bc1_buf: self.scalar_buffer(bc1),
            bc2_buf: self.scalar_buffer(bc2),
            gs_buf: self.scalar_buffer(grad_scale),
        }
    }

    /// Single-dispatch convenience (for tests). Encodes one step and executes immediately.
    pub fn step(
        &self,
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        let mut batch = self.begin_batch(t, beta1, beta2, eps, 1.0);
        batch.add(param, grad, m, v, lr, weight_decay);
        batch.execute();
    }

    /// Try zero-copy buffer (requires page-aligned pointer + page-multiple length).
    /// Returns None if alignment requirements aren't met.
    fn try_buffer_no_copy(
        &self,
        ptr: *mut c_void,
        byte_len: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        if byte_len < PAGE_SIZE || byte_len % PAGE_SIZE != 0 || (ptr as usize) % PAGE_SIZE != 0 {
            return None;
        }
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(ptr)?,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
    }

    /// Create a copy-based Metal buffer (works with any alignment).
    fn make_buffer(
        &self,
        ptr: *const f32,
        byte_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(ptr as *mut c_void).expect("non-null"),
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("make_buffer")
        }
    }

    fn scalar_buffer(&self, val: f32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(&val as *const f32 as *mut c_void).expect("non-null"),
                    4,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("scalar_buffer")
        }
    }
}

impl<'a> AdamBatch<'a> {
    /// Encode one Adam dispatch into this batch.
    /// Shared scalars (beta1, beta2, eps, bc1, bc2, grad_scale) come from `begin_batch()`.
    /// Only `lr` and `weight_decay` are per-tensor (differ across embedding/matrix/norm groups).
    pub fn add(
        &mut self,
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        weight_decay: f32,
    ) {
        let n = param.len();
        assert_eq!(grad.len(), n);
        assert_eq!(m.len(), n);
        assert_eq!(v.len(), n);
        let byte_len = n * 4;

        // Create buffers — zero-copy for large page-aligned data, copy for small
        let (param_buf, param_needs_rb) = self.smart_buffer_mut(param.as_mut_ptr(), byte_len);
        let grad_buf = self.smart_buffer_ro(grad.as_ptr(), byte_len);
        let (m_buf, m_needs_rb) = self.smart_buffer_mut(m.as_mut_ptr(), byte_len);
        let (v_buf, v_needs_rb) = self.smart_buffer_mut(v.as_mut_ptr(), byte_len);

        // Per-tensor scalars (lr and wd vary across embed/matrix/norm groups)
        let lr_buf = self.adam.scalar_buffer(lr);
        let wd_buf = self.adam.scalar_buffer(weight_decay);

        unsafe {
            self.enc.setComputePipelineState(&self.adam.pipeline);
            self.enc.setBuffer_offset_atIndex(Some(&param_buf), 0, 0);
            self.enc.setBuffer_offset_atIndex(Some(&grad_buf), 0, 1);
            self.enc.setBuffer_offset_atIndex(Some(&m_buf), 0, 2);
            self.enc.setBuffer_offset_atIndex(Some(&v_buf), 0, 3);
            self.enc.setBuffer_offset_atIndex(Some(&lr_buf), 0, 4);
            self.enc.setBuffer_offset_atIndex(Some(&self.b1_buf), 0, 5);
            self.enc.setBuffer_offset_atIndex(Some(&self.b2_buf), 0, 6);
            self.enc.setBuffer_offset_atIndex(Some(&self.eps_buf), 0, 7);
            self.enc.setBuffer_offset_atIndex(Some(&wd_buf), 0, 8);
            self.enc.setBuffer_offset_atIndex(Some(&self.bc1_buf), 0, 9);
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bc2_buf), 0, 10);
            self.enc.setBuffer_offset_atIndex(Some(&self.gs_buf), 0, 11);

            let tg = self.adam.pipeline.maxTotalThreadsPerThreadgroup().min(256);
            let grid = MTLSize {
                width: n,
                height: 1,
                depth: 1,
            };
            let group = MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            };
            self.enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        }

        // Track readbacks for copy-based buffers
        if param_needs_rb {
            self.readbacks.push(Readback {
                buf: param_buf,
                dst: param.as_mut_ptr(),
                len: n,
            });
        } else {
            self._held.push(param_buf);
        }
        if m_needs_rb {
            self.readbacks.push(Readback {
                buf: m_buf,
                dst: m.as_mut_ptr(),
                len: n,
            });
        } else {
            self._held.push(m_buf);
        }
        if v_needs_rb {
            self.readbacks.push(Readback {
                buf: v_buf,
                dst: v.as_mut_ptr(),
                len: n,
            });
        } else {
            self._held.push(v_buf);
        }
        self._held.push(grad_buf);
    }

    /// Commit the command buffer, wait for GPU, copy back any copy-based buffers.
    pub fn execute(self) {
        self.enc.endEncoding();
        self.cmd.commit();
        self.cmd.waitUntilCompleted();

        // Copy back from any copy-based (non-zero-copy) buffers
        for rb in &self.readbacks {
            unsafe {
                let src = rb.buf.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, rb.dst, rb.len);
            }
        }
    }

    /// Create a buffer for mutable data. Returns (buffer, needs_readback).
    /// Zero-copy if page-aligned; copy-based otherwise.
    fn smart_buffer_mut(
        &self,
        ptr: *mut f32,
        byte_len: usize,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, bool) {
        if let Some(buf) = self.adam.try_buffer_no_copy(ptr as *mut c_void, byte_len) {
            (buf, false) // zero-copy, GPU writes directly to Rust memory
        } else {
            (self.adam.make_buffer(ptr as *const f32, byte_len), true)
        }
    }

    /// Create a buffer for read-only data. Zero-copy if possible.
    fn smart_buffer_ro(
        &self,
        ptr: *const f32,
        byte_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        if let Some(buf) = self
            .adam
            .try_buffer_no_copy(ptr as *const f32 as *mut c_void, byte_len)
        {
            buf
        } else {
            self.adam.make_buffer(ptr, byte_len)
        }
    }
}
