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

    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
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

/// y = W @ x where W is f16 — chunked convert+sgemv for minimal memory traffic.
///
/// Converts 64 rows at a time into an L2-sized f32 buffer, then uses AMX-optimized
/// cblas_sgemv on the chunk. Main memory reads only f16 weights (half of f32).
/// The f32 chunk lives in L2 cache — sgemv reads it at ~1 TB/s, not ~200 GB/s.
pub fn sgemv_f16(w: &[f16], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use half::slice::HalfFloatSliceExt;
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    const CHUNK_ROWS: usize = 64; // 64 rows × 16384 cols × 4B = 4 MB — fits in L2
    let mut chunk_buf = vec![0.0f32; CHUNK_ROWS * in_dim];

    for chunk_start in (0..out_dim).step_by(CHUNK_ROWS) {
        let chunk_end = (chunk_start + CHUNK_ROWS).min(out_dim);
        let chunk_rows = chunk_end - chunk_start;
        let w_chunk = &w[chunk_start * in_dim..chunk_end * in_dim];
        let y_chunk = &mut y[chunk_start..chunk_end];

        // SIMD bulk convert f16→f32 (FCVTL: 4 f16→4 f32 per instruction)
        let buf = &mut chunk_buf[..chunk_rows * in_dim];
        w_chunk.convert_to_f32_slice(buf);

        // AMX-optimized sgemv on the L2-resident chunk
        sgemv_f32(buf, x, y_chunk, chunk_rows, in_dim);
    }
}

/// y = W @ x where W is f16 — multi-threaded chunked convert+sgemv.
///
/// Each rayon thread converts its chunk of rows from f16→f32 into an L2-resident
/// buffer, then runs AMX-optimized sgemv on the chunk. Reads half the DRAM bytes
/// vs pre-converted f32 path, with full multi-core parallelism.
/// Use for large matrices (>100 MB) where DRAM bandwidth is the bottleneck.
pub fn sgemv_f16_par(w: &[f16], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use rayon::prelude::*;
    use half::slice::HalfFloatSliceExt;
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    const CHUNK_ROWS: usize = 64; // 64 × 16384 × 4B = 4 MB — fits in L2

    y.par_chunks_mut(CHUNK_ROWS).enumerate().for_each(|(ci, y_chunk)| {
        let chunk_start = ci * CHUNK_ROWS;
        let chunk_rows = y_chunk.len();
        let mut buf = vec![0.0f32; chunk_rows * in_dim];
        let w_start = chunk_start * in_dim;
        w[w_start..w_start + chunk_rows * in_dim].convert_to_f32_slice(&mut buf);
        sgemv_f32(&buf, x, y_chunk, chunk_rows, in_dim);
    });
}

/// y = W^T @ x where W is f16 — chunked convert+sgemv_trans.
pub fn sgemv_f16_trans(w: &[f16], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use half::slice::HalfFloatSliceExt;
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    // For transposed sgemv, we need the full matrix converted since each output
    // depends on ALL rows. Use chunked conversion for cache efficiency.
    // W is stored [in_dim, out_dim]. We compute y = W^T @ x.
    const CHUNK_ROWS: usize = 64;
    let mut chunk_buf = vec![0.0f32; CHUNK_ROWS * out_dim];

    // Zero output first (we accumulate)
    for d in 0..out_dim { y[d] = 0.0; }

    for chunk_start in (0..in_dim).step_by(CHUNK_ROWS) {
        let chunk_end = (chunk_start + CHUNK_ROWS).min(in_dim);
        let chunk_rows = chunk_end - chunk_start;
        let w_chunk = &w[chunk_start * out_dim..chunk_end * out_dim];
        let x_chunk = &x[chunk_start..chunk_end];

        // SIMD bulk convert f16→f32
        let buf = &mut chunk_buf[..chunk_rows * out_dim];
        w_chunk.convert_to_f32_slice(buf);

        // Accumulate: y += chunk^T @ x_chunk
        // Using sgemv with beta=1.0 to accumulate
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                112, // CblasTrans
                chunk_rows as i32,
                out_dim as i32,
                1.0,
                buf.as_ptr(),
                out_dim as i32,
                x_chunk.as_ptr(),
                1,
                1.0, // beta=1.0 to accumulate
                y.as_mut_ptr(),
                1,
            );
        }
    }
}

/// y = W^T @ x using Accelerate cblas_sgemv (transposed).
/// W is [cols, rows] row-major f32 (rows/cols from the TRANSPOSED perspective).
/// Computes y[i] = sum_j W[j][i] * x[j], i.e., y = W^T @ x.
pub fn sgemv_f32_trans(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // W is stored as [in_dim, out_dim] row-major
    // We want y[out_dim] = W^T[out_dim, in_dim] @ x[in_dim]
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    const CBLAS_TRANS: i32 = 112;
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            in_dim as i32,    // m (rows of W)
            out_dim as i32,   // n (cols of W)
            1.0,
            w.as_ptr(),
            out_dim as i32,   // lda = out_dim (cols of W in row-major)
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

/// C = A @ B using Accelerate cblas_sgemm.
/// A is [m, k] row-major, B is [k, n] row-major, C is [m, n] row-major.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,          // alpha
            a.as_ptr(),
            k as i32,     // lda
            b.as_ptr(),
            n as i32,     // ldb
            0.0,          // beta
            c.as_mut_ptr(),
            n as i32,     // ldc
        );
    }
}

/// C = A @ B^T using Accelerate cblas_sgemm (B transposed).
/// A is [m, k] row-major, B is [n, k] row-major, C is [m, n] row-major.
pub fn sgemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    const CBLAS_TRANS: i32 = 112;
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,     // lda
            b.as_ptr(),
            k as i32,     // ldb (B is [n, k], stored row-major)
            0.0,
            c.as_mut_ptr(),
            n as i32,     // ldc
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
    fn sgemm_matches_naive() {
        let m = 2048;
        let n = 13;
        let k = 4096;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();
        let mut blas_out = vec![0.0f32; m * n];
        sgemm(&a, &b, &mut blas_out, m, n, k);

        // Naive reference
        let mut naive_out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a[i * k + p] as f64 * b[p * n + j] as f64;
                }
                naive_out[i * n + j] = sum as f32;
            }
        }

        let max_diff = blas_out.iter().zip(naive_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-2, "BLAS sgemm vs naive max_diff={max_diff}");
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
