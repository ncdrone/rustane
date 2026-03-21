//! Accelerate BLAS bindings for matrix-vector multiply.
//!
//! Uses cblas_sgemv (AMX hardware on Apple Silicon) — ~40x faster than scalar loops.
//! No Cargo.toml changes needed: `#[link(name = "Accelerate", kind = "framework")]`.

use half::f16;

// CBLAS enums
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

/// y = W @ x using Accelerate cblas_sgemv.
/// W is [rows, cols] row-major f32, x is [cols], y is [rows].
pub fn sgemv_f32(w: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);

    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            rows as i32,
            cols as i32,
            1.0,          // alpha
            w.as_ptr(),
            cols as i32,  // lda
            x.as_ptr(),
            1,            // incx
            0.0,          // beta
            y.as_mut_ptr(),
            1,            // incy
        );
    }
}

/// y = W @ x where W is f16 (converted to f32 on the fly).
/// W is [out_dim, in_dim] row-major f16, x is [in_dim] f32, y is [out_dim] f32.
pub fn sgemv_f16(w: &[f16], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    // Convert f16 weights to f32 for BLAS
    let w_f32: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();

    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            out_dim as i32,
            in_dim as i32,
            1.0,
            w_f32.as_ptr(),
            in_dim as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemv_matches_naive() {
        let rows = 4096;
        let cols = 2048;
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.001).sin()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut blas_out = vec![0.0f32; rows];
        sgemv_f32(&w, &x, &mut blas_out, rows, cols);

        // Naive reference
        let mut naive_out = vec![0.0f32; rows];
        for i in 0..rows {
            let mut sum = 0.0f64;
            for j in 0..cols {
                sum += w[i * cols + j] as f64 * x[j] as f64;
            }
            naive_out[i] = sum as f32;
        }

        let max_diff = blas_out
            .iter()
            .zip(naive_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-3, "BLAS vs naive max_diff={max_diff}");
    }

    #[test]
    fn sgemv_f16_matches_naive() {
        let rows = 128;
        let cols = 64;
        let w: Vec<f16> = (0..rows * cols)
            .map(|i| f16::from_f32((i as f32 * 0.01).sin()))
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).cos()).collect();
        let mut blas_out = vec![0.0f32; rows];
        sgemv_f16(&w, &x, &mut blas_out, rows, cols);

        // Naive reference
        let mut naive_out = vec![0.0f32; rows];
        for i in 0..rows {
            let mut sum = 0.0f64;
            for j in 0..cols {
                sum += w[i * cols + j].to_f32() as f64 * x[j] as f64;
            }
            naive_out[i] = sum as f32;
        }

        let max_diff = blas_out
            .iter()
            .zip(naive_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-2, "BLAS f16 vs naive max_diff={max_diff}");
    }
}
