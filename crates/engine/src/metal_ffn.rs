//! Metal GPU FFN path using MPS GEMMs plus a small configurable gate kernel.

use crate::model::{FfnActivation, ModelConfig};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::ffi::c_void;

const PAGE_SIZE: usize = 16384;

const GATE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_gate(
    device const float* h1 [[buffer(0)]],
    device const float* h3 [[buffer(1)]],
    device float* gate [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = h1[id];
    float sig = 1.0f / (1.0f + exp(-x));
    gate[id] = x * sig * h3[id];
}

kernel void leaky_relu_sq_gate(
    device const float* h1 [[buffer(0)]],
    device const float* h3 [[buffer(1)]],
    device float* gate [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = h1[id];
    float l = max(x, 0.5f * x);
    gate[id] = l * l * h3[id];
}
"#;

struct Readback {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dst: *mut f32,
    len: usize,
}

/// Metal-backed SwiGLU FFN for a fixed model shape.
pub struct MetalFFN {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    gate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x_desc: Retained<MPSMatrixDescriptor>,
    hidden_desc: Retained<MPSMatrixDescriptor>,
    weight_desc: Retained<MPSMatrixDescriptor>,
    w1_matmul: Retained<MPSMatrixMultiplication>,
    w3_matmul: Retained<MPSMatrixMultiplication>,
    w2_matmul: Retained<MPSMatrixMultiplication>,
    dx_w1_matmul: Retained<MPSMatrixMultiplication>,
    dx_w3_matmul: Retained<MPSMatrixMultiplication>,
    hidden_elems: usize,
}

impl MetalFFN {
    pub fn new(cfg: &ModelConfig) -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;

        let source = NSString::from_str(GATE_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .ok()?;
        let fn_name = NSString::from_str(match cfg.ffn_activation {
            FfnActivation::SwiGlu => "silu_gate",
            FfnActivation::LeakyReluSq => "leaky_relu_sq_gate",
        });
        let function = library.newFunctionWithName(&fn_name)?;
        let gate_pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;

        let x_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                cfg.dim,
                cfg.seq,
                cfg.seq * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };
        let hidden_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                cfg.hidden,
                cfg.seq,
                cfg.seq * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };
        let weight_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                cfg.dim,
                cfg.hidden,
                cfg.hidden * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };

        let w1_matmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &device,
                true,
                false,
                cfg.hidden,
                cfg.seq,
                cfg.dim,
                1.0,
                0.0,
            )
        };
        let w3_matmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &device,
                true,
                false,
                cfg.hidden,
                cfg.seq,
                cfg.dim,
                1.0,
                0.0,
            )
        };
        let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();
        let w2_matmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &device,
                false,
                false,
                cfg.dim,
                cfg.seq,
                cfg.hidden,
                alpha as f64,
                1.0,
            )
        };
        let dx_w1_matmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &device,
                false,
                false,
                cfg.dim,
                cfg.seq,
                cfg.hidden,
                1.0,
                0.0,
            )
        };
        let dx_w3_matmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &device,
                false,
                false,
                cfg.dim,
                cfg.seq,
                cfg.hidden,
                1.0,
                1.0,
            )
        };

        Some(Self {
            device,
            queue,
            gate_pipeline,
            x_desc,
            hidden_desc,
            weight_desc,
            w1_matmul,
            w3_matmul,
            w2_matmul,
            dx_w1_matmul,
            dx_w3_matmul,
            hidden_elems: cfg.hidden * cfg.seq,
        })
    }

    /// Compute:
    /// - `h1 = W1^T @ x2norm`
    /// - `h3 = W3^T @ x2norm`
    /// - `gate = silu(h1) * h3`
    /// - `x_next = x2 + alpha * (W2 @ gate)`
    pub fn forward_into(
        &self,
        cfg: &ModelConfig,
        x2norm: &[f32],
        w1: &[f32],
        w3: &[f32],
        w2: &[f32],
        x2: &[f32],
        h1: &mut [f32],
        h3: &mut [f32],
        gate: &mut [f32],
        x_next: &mut [f32],
    ) {
        let x_elems = cfg.dim * cfg.seq;
        let hidden_elems = cfg.hidden * cfg.seq;
        let weight_elems = cfg.dim * cfg.hidden;

        assert_eq!(x2norm.len(), x_elems);
        assert_eq!(x2.len(), x_elems);
        assert_eq!(w1.len(), weight_elems);
        assert_eq!(w3.len(), weight_elems);
        assert_eq!(w2.len(), weight_elems);
        assert_eq!(h1.len(), hidden_elems);
        assert_eq!(h3.len(), hidden_elems);
        assert_eq!(gate.len(), hidden_elems);
        assert_eq!(x_next.len(), x_elems);

        x_next.copy_from_slice(x2);

        let x_bytes = x_elems * std::mem::size_of::<f32>();
        let hidden_bytes = hidden_elems * std::mem::size_of::<f32>();
        let weight_bytes = weight_elems * std::mem::size_of::<f32>();

        let x2norm_buf = self.smart_buffer_ro(x2norm.as_ptr(), x_bytes);
        let w1_buf = self.smart_buffer_ro(w1.as_ptr(), weight_bytes);
        let w3_buf = self.smart_buffer_ro(w3.as_ptr(), weight_bytes);
        let w2_buf = self.smart_buffer_ro(w2.as_ptr(), weight_bytes);
        let (h1_buf, h1_needs_rb) = self.smart_buffer_mut(h1.as_mut_ptr(), hidden_bytes);
        let (h3_buf, h3_needs_rb) = self.smart_buffer_mut(h3.as_mut_ptr(), hidden_bytes);
        let (gate_buf, gate_needs_rb) = self.smart_buffer_mut(gate.as_mut_ptr(), hidden_bytes);
        let (x_next_buf, x_next_needs_rb) = self.smart_buffer_mut(x_next.as_mut_ptr(), x_bytes);

        let mut held = vec![x2norm_buf, w1_buf, w3_buf, w2_buf];
        let mut readbacks = Vec::new();

        let x2norm_matrix = self.make_matrix(&held[0], &self.x_desc);
        let w1_matrix = self.make_matrix(&held[1], &self.weight_desc);
        let w3_matrix = self.make_matrix(&held[2], &self.weight_desc);
        let w2_matrix = self.make_matrix(&held[3], &self.weight_desc);
        let h1_matrix = self.make_matrix(&h1_buf, &self.hidden_desc);
        let h3_matrix = self.make_matrix(&h3_buf, &self.hidden_desc);
        let gate_matrix = self.make_matrix(&gate_buf, &self.hidden_desc);
        let x_next_matrix = self.make_matrix(&x_next_buf, &self.x_desc);

        let cmd = self.queue.commandBuffer().expect("command buffer");

        unsafe {
            self.w1_matmul
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    &w1_matrix,
                    &x2norm_matrix,
                    &h1_matrix,
                );
            self.w3_matmul
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    &w3_matrix,
                    &x2norm_matrix,
                    &h3_matrix,
                );
        }

        {
            let enc = cmd.computeCommandEncoder().expect("compute encoder");
            unsafe {
                enc.setComputePipelineState(&self.gate_pipeline);
                enc.setBuffer_offset_atIndex(Some(&h1_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&h3_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&gate_buf), 0, 2);
                let tg = self.gate_pipeline.maxTotalThreadsPerThreadgroup().min(256);
                let grid = MTLSize {
                    width: self.hidden_elems,
                    height: 1,
                    depth: 1,
                };
                let group = MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            }
            enc.endEncoding();
        }

        unsafe {
            self.w2_matmul
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    &w2_matrix,
                    &gate_matrix,
                    &x_next_matrix,
                );
        }

        if h1_needs_rb {
            readbacks.push(Readback {
                buf: h1_buf,
                dst: h1.as_mut_ptr(),
                len: hidden_elems,
            });
        } else {
            held.push(h1_buf);
        }
        if h3_needs_rb {
            readbacks.push(Readback {
                buf: h3_buf,
                dst: h3.as_mut_ptr(),
                len: hidden_elems,
            });
        } else {
            held.push(h3_buf);
        }
        if gate_needs_rb {
            readbacks.push(Readback {
                buf: gate_buf,
                dst: gate.as_mut_ptr(),
                len: hidden_elems,
            });
        } else {
            held.push(gate_buf);
        }
        if x_next_needs_rb {
            readbacks.push(Readback {
                buf: x_next_buf,
                dst: x_next.as_mut_ptr(),
                len: x_elems,
            });
        } else {
            held.push(x_next_buf);
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        for rb in readbacks {
            unsafe {
                let src = rb.buf.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, rb.dst, rb.len);
            }
        }
    }

    /// Compute `dx_ffn = W1 @ dh1 + W3 @ dh3`.
    pub fn backward_dx_into(
        &self,
        cfg: &ModelConfig,
        dh1: &[f32],
        dh3: &[f32],
        w1: &[f32],
        w3: &[f32],
        dx_ffn: &mut [f32],
    ) {
        let x_elems = cfg.dim * cfg.seq;
        let hidden_elems = cfg.hidden * cfg.seq;
        let weight_elems = cfg.dim * cfg.hidden;

        assert_eq!(dh1.len(), hidden_elems);
        assert_eq!(dh3.len(), hidden_elems);
        assert_eq!(w1.len(), weight_elems);
        assert_eq!(w3.len(), weight_elems);
        assert_eq!(dx_ffn.len(), x_elems);

        dx_ffn.fill(0.0);

        let x_bytes = x_elems * std::mem::size_of::<f32>();
        let hidden_bytes = hidden_elems * std::mem::size_of::<f32>();
        let weight_bytes = weight_elems * std::mem::size_of::<f32>();

        let dh1_buf = self.smart_buffer_ro(dh1.as_ptr(), hidden_bytes);
        let dh3_buf = self.smart_buffer_ro(dh3.as_ptr(), hidden_bytes);
        let w1_buf = self.smart_buffer_ro(w1.as_ptr(), weight_bytes);
        let w3_buf = self.smart_buffer_ro(w3.as_ptr(), weight_bytes);
        let (dx_buf, dx_needs_rb) = self.smart_buffer_mut(dx_ffn.as_mut_ptr(), x_bytes);

        let mut held = vec![dh1_buf, dh3_buf, w1_buf, w3_buf];
        let mut readbacks = Vec::new();

        let dh1_matrix = self.make_matrix(&held[0], &self.hidden_desc);
        let dh3_matrix = self.make_matrix(&held[1], &self.hidden_desc);
        let w1_matrix = self.make_matrix(&held[2], &self.weight_desc);
        let w3_matrix = self.make_matrix(&held[3], &self.weight_desc);
        let dx_matrix = self.make_matrix(&dx_buf, &self.x_desc);

        let cmd = self.queue.commandBuffer().expect("command buffer");
        unsafe {
            self.dx_w1_matmul
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    &w1_matrix,
                    &dh1_matrix,
                    &dx_matrix,
                );
            self.dx_w3_matmul
                .encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    &w3_matrix,
                    &dh3_matrix,
                    &dx_matrix,
                );
        }

        if dx_needs_rb {
            readbacks.push(Readback {
                buf: dx_buf,
                dst: dx_ffn.as_mut_ptr(),
                len: x_elems,
            });
        } else {
            held.push(dx_buf);
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        for rb in readbacks {
            unsafe {
                let src = rb.buf.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, rb.dst, rb.len);
            }
        }
    }

    fn make_matrix(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        descriptor: &MPSMatrixDescriptor,
    ) -> Retained<MPSMatrix> {
        unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buffer, descriptor) }
    }

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

    fn smart_buffer_mut(
        &self,
        ptr: *mut f32,
        byte_len: usize,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, bool) {
        if let Some(buf) = self.try_buffer_no_copy(ptr as *mut c_void, byte_len) {
            (buf, false)
        } else {
            (self.make_buffer(ptr as *const f32, byte_len), true)
        }
    }

    fn smart_buffer_ro(
        &self,
        ptr: *const f32,
        byte_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        if let Some(buf) = self.try_buffer_no_copy(ptr as *const f32 as *mut c_void, byte_len) {
            buf
        } else {
            self.make_buffer(ptr, byte_len)
        }
    }
}
