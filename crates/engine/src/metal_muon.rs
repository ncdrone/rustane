//! Metal GPU helper for Muon's Newton-Schulz orthogonalization step.

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::ffi::c_void;
use std::ptr::NonNull;

const MUON_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void combine_poly(
    device const float* a [[buffer(0)]],
    device const float* a2 [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& bcoef [[buffer(3)]],
    constant float& ccoef [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = bcoef * a[id] + ccoef * a2[id];
}

kernel void combine_x(
    device const float* x [[buffer(0)]],
    device const float* bx [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& acoef [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = acoef * x[id] + bx[id];
}
"#;

const A_COEF: f32 = 3.4445;
const B_COEF: f32 = -4.7750;
const C_COEF: f32 = 2.0315;
const EPS: f32 = 1e-7;

pub struct MetalMuon {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    combine_poly_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    combine_x_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalMuon {
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;

        let source = NSString::from_str(MUON_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .ok()?;
        let poly_name = NSString::from_str("combine_poly");
        let x_name = NSString::from_str("combine_x");
        let poly_fn = library.newFunctionWithName(&poly_name)?;
        let x_fn = library.newFunctionWithName(&x_name)?;
        let combine_poly_pipeline = device
            .newComputePipelineStateWithFunction_error(&poly_fn)
            .ok()?;
        let combine_x_pipeline = device
            .newComputePipelineStateWithFunction_error(&x_fn)
            .ok()?;

        Some(Self {
            device,
            queue,
            combine_poly_pipeline,
            combine_x_pipeline,
        })
    }

    pub fn orthogonalize_inplace(&self, rows: usize, cols: usize, matrix: &mut [f32], steps: u32) {
        assert_eq!(matrix.len(), rows * cols);
        if matrix.is_empty() {
            return;
        }

        let transposed = rows > cols;
        let x_rows = rows.min(cols);
        let x_cols = rows.max(cols);
        let mut host_x = if transposed {
            transpose_row_major(matrix, rows, cols)
        } else {
            matrix.to_vec()
        };

        let norm = host_x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= EPS {
            matrix.fill(0.0);
            return;
        }
        let inv_norm = 1.0 / (norm + EPS);
        for v in &mut host_x {
            *v *= inv_norm;
        }

        let x_bytes = host_x.len() * std::mem::size_of::<f32>();
        let a_elems = x_rows * x_rows;
        let a_bytes = a_elems * std::mem::size_of::<f32>();

        let x0 = buffer_from_slice(&self.device, &host_x);
        let x1 = zero_buffer(&self.device, x_bytes);
        let a = zero_buffer(&self.device, a_bytes);
        let a2 = zero_buffer(&self.device, a_bytes);
        let b = zero_buffer(&self.device, a_bytes);
        let bx = zero_buffer(&self.device, x_bytes);

        let x_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                x_rows,
                x_cols,
                x_cols * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };
        let a_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                x_rows,
                x_rows,
                x_rows * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };

        let x0_mat = make_matrix(&x0, &x_desc);
        let x1_mat = make_matrix(&x1, &x_desc);
        let a_mat = make_matrix(&a, &a_desc);
        let a2_mat = make_matrix(&a2, &a_desc);
        let b_mat = make_matrix(&b, &a_desc);
        let bx_mat = make_matrix(&bx, &x_desc);

        let xxt = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &self.device,
                false,
                true,
                x_rows,
                x_rows,
                x_cols,
                1.0,
                0.0,
            )
        };
        let aa = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &self.device,
                false,
                false,
                x_rows,
                x_rows,
                x_rows,
                1.0,
                0.0,
            )
        };
        let bxmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &self.device,
                false,
                false,
                x_rows,
                x_cols,
                x_rows,
                1.0,
                0.0,
            )
        };

        let bcoef_buf = scalar_buffer(&self.device, B_COEF);
        let ccoef_buf = scalar_buffer(&self.device, C_COEF);
        let acoef_buf = scalar_buffer(&self.device, A_COEF);
        let cmd = self.queue.commandBuffer().expect("command buffer");

        let mut current_buf = &x0;
        let mut current_mat = &x0_mat;
        let mut next_buf = &x1;
        let mut next_mat = &x1_mat;
        for _ in 0..steps {
            unsafe {
                xxt.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    current_mat,
                    current_mat,
                    &a_mat,
                );
                aa.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd, &a_mat, &a_mat, &a2_mat,
                );
            }
            encode_combine_poly(
                &cmd,
                &self.combine_poly_pipeline,
                &a,
                &a2,
                &b,
                &bcoef_buf,
                &ccoef_buf,
                a_elems,
            );
            unsafe {
                bxmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd,
                    &b_mat,
                    current_mat,
                    &bx_mat,
                );
            }
            encode_combine_x(
                &cmd,
                &self.combine_x_pipeline,
                current_buf,
                &bx,
                next_buf,
                &acoef_buf,
                host_x.len(),
            );
            std::mem::swap(&mut current_buf, &mut next_buf);
            std::mem::swap(&mut current_mat, &mut next_mat);
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        download_f32(current_buf, &mut host_x);
        if transposed {
            let out = transpose_row_major(&host_x, x_rows, x_cols);
            matrix.copy_from_slice(&out);
        } else {
            matrix.copy_from_slice(&host_x);
        }
    }
}

fn buffer_from_slice(
    device: &ProtocolObject<dyn MTLDevice>,
    xs: &[f32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(xs.as_ptr() as *mut c_void).expect("non-null"),
                std::mem::size_of_val(xs),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("buffer from slice")
    }
}

fn zero_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    byte_len: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let buf = device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
        .expect("zero buffer");
    unsafe {
        std::ptr::write_bytes(buf.contents().as_ptr(), 0, byte_len);
    }
    buf
}

fn scalar_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    val: f32,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(&val as *const f32 as *mut c_void).expect("non-null"),
                std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("scalar buffer")
    }
}

fn download_f32(buf: &ProtocolObject<dyn MTLBuffer>, xs: &mut [f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr() as *const f32,
            xs.as_mut_ptr(),
            xs.len(),
        );
    }
}

fn make_matrix(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    descriptor: &MPSMatrixDescriptor,
) -> Retained<MPSMatrix> {
    unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buffer, descriptor) }
}

fn encode_combine_poly(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &ProtocolObject<dyn MTLBuffer>,
    a2: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    bcoef: &ProtocolObject<dyn MTLBuffer>,
    ccoef: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    unsafe {
        enc.setComputePipelineState(pipeline);
        enc.setBuffer_offset_atIndex(Some(a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(a2), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(bcoef), 0, 3);
        enc.setBuffer_offset_atIndex(Some(ccoef), 0, 4);
        dispatch_1d(enc.as_ref(), pipeline, len);
        enc.endEncoding();
    }
}

fn encode_combine_x(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    x: &ProtocolObject<dyn MTLBuffer>,
    bx: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    acoef: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    unsafe {
        enc.setComputePipelineState(pipeline);
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(bx), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(acoef), 0, 3);
        dispatch_1d(enc.as_ref(), pipeline, len);
        enc.endEncoding();
    }
}

unsafe fn dispatch_1d(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    len: usize,
) {
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let grid = MTLSize {
        width: len,
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

fn transpose_row_major(xs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; xs.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = xs[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::MetalMuon;

    fn zeropower_cpu(g: &[f32], rows: usize, cols: usize, steps: u32) -> Vec<f32> {
        let transposed = rows > cols;
        let x_rows = rows.min(cols);
        let x_cols = rows.max(cols);
        let mut x = if transposed {
            transpose_cpu(g, rows, cols)
        } else {
            g.to_vec()
        };
        let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm <= 1e-7 {
            return vec![0.0; g.len()];
        }
        for v in &mut x {
            *v /= norm + 1e-7;
        }

        let mut a = vec![0.0; x_rows * x_rows];
        let mut a2 = vec![0.0; x_rows * x_rows];
        let mut b = vec![0.0; x_rows * x_rows];
        let mut bx = vec![0.0; x_rows * x_cols];
        for _ in 0..steps {
            matmul_nt(&x, x_rows, x_cols, &x, x_rows, x_cols, &mut a);
            matmul_nn(&a, x_rows, x_rows, &a, x_rows, x_rows, &mut a2);
            for i in 0..b.len() {
                b[i] = super::B_COEF * a[i] + super::C_COEF * a2[i];
            }
            matmul_nn(&b, x_rows, x_rows, &x, x_rows, x_cols, &mut bx);
            for i in 0..x.len() {
                x[i] = super::A_COEF * x[i] + bx[i];
            }
        }
        if transposed {
            transpose_cpu(&x, x_rows, x_cols)
        } else {
            x
        }
    }

    fn matmul_nn(
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        _b_rows: usize,
        b_cols: usize,
        out: &mut [f32],
    ) {
        out.fill(0.0);
        for i in 0..a_rows {
            for k in 0..a_cols {
                let aik = a[i * a_cols + k];
                for j in 0..b_cols {
                    out[i * b_cols + j] += aik * b[k * b_cols + j];
                }
            }
        }
    }

    fn matmul_nt(
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
        out: &mut [f32],
    ) {
        out.fill(0.0);
        for i in 0..a_rows {
            for j in 0..b_rows {
                let mut acc = 0.0;
                for k in 0..a_cols {
                    acc += a[i * a_cols + k] * b[j * b_cols + k];
                }
                out[i * b_rows + j] = acc;
            }
        }
    }

    fn transpose_cpu(xs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; xs.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = xs[r * cols + c];
            }
        }
        out
    }

    #[test]
    fn metal_muon_matches_cpu_reference_small_matrix() {
        let Some(muon) = MetalMuon::new() else {
            return;
        };
        let rows = 4;
        let cols = 3;
        let mut gpu = vec![
            0.3, -0.2, 0.1, 0.7, 0.4, -0.5, -0.6, 0.2, 0.9, 0.1, -0.3, 0.8,
        ];
        let cpu = zeropower_cpu(&gpu, rows, cols, 5);
        muon.orthogonalize_inplace(rows, cols, &mut gpu, 5);
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert!((g - c).abs() < 5e-3, "gpu={g} cpu={c}");
        }
    }
}
