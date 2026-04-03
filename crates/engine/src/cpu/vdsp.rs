//! Raw FFI bindings to Apple's Accelerate framework (vDSP + vecLib).
//!
//! These are the building blocks for RMSNorm, cross-entropy, Adam, and SiLU.
//! All operate on f32 buffers — ANE handles fp16 conversion internally.

use std::os::raw::c_int;

// vDSP stride type
type Stride = isize;
type Length = u64;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    // ── Element-wise arithmetic ──────────────────────────────────

    /// C = A * B (element-wise)
    pub fn vDSP_vmul(
        a: *const f32,
        ia: Stride,
        b: *const f32,
        ib: Stride,
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    /// C = A + B (element-wise)
    pub fn vDSP_vadd(
        a: *const f32,
        ia: Stride,
        b: *const f32,
        ib: Stride,
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    /// C = A - B (element-wise)
    pub fn vDSP_vsub(
        a: *const f32,
        ia: Stride, // subtracted FROM result
        b: *const f32,
        ib: Stride, // subtracted from THIS
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    // ── Scalar-vector operations ─────────────────────────────────

    /// C = A * scalar (element-wise)
    pub fn vDSP_vsmul(
        a: *const f32,
        ia: Stride,
        scalar: *const f32,
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    /// C = A + scalar (element-wise)
    pub fn vDSP_vsadd(
        a: *const f32,
        ia: Stride,
        scalar: *const f32,
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    /// C = A * scalar + B (fused multiply-add, scalar)
    pub fn vDSP_vsma(
        a: *const f32,
        ia: Stride,
        scalar: *const f32,
        b: *const f32,
        ib: Stride,
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    /// C = A * scalar_mul + scalar_add (fused scale + offset)
    pub fn vDSP_vsmsa(
        a: *const f32,
        ia: Stride,
        scalar_mul: *const f32,
        scalar_add: *const f32,
        c: *mut f32,
        ic: Stride,
        n: Length,
    );

    // ── Reductions ───────────────────────────────────────────────

    /// result = max(A)
    pub fn vDSP_maxv(a: *const f32, ia: Stride, result: *mut f32, n: Length);

    /// result = sum(A)
    pub fn vDSP_sve(a: *const f32, ia: Stride, result: *mut f32, n: Length);

    // ── vecLib math functions ────────────────────────────────────

    /// y[i] = exp(x[i]) for n elements
    pub fn vvexpf(y: *mut f32, x: *const f32, n: *const c_int);

    /// y[i] = 1/sqrt(x[i]) for n elements
    pub fn vvrsqrtf(y: *mut f32, x: *const f32, n: *const c_int);

    /// y[i] = tanh(x[i]) for n elements
    pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const c_int);

    /// y[i] = 1/x[i] for n elements
    pub fn vvrecf(y: *mut f32, x: *const f32, n: *const c_int);

    // ── Matrix operations ──────────────────────────────────────────

    /// Transpose an M×N matrix A into N×M matrix C.
    pub fn vDSP_mtrans(a: *const f32, ia: Stride, c: *mut f32, ic: Stride, m: Length, n: Length);

    /// Sum of squares: result = sum(A[i]^2)
    pub fn vDSP_svesq(a: *const f32, ia: Stride, result: *mut f32, n: Length);

    /// In-place scale: X *= alpha
    pub fn cblas_sscal(n: c_int, alpha: f32, x: *mut f32, incx: c_int);

    // ── BLAS ───────────────────────────────────────────────────────

    /// C = alpha * op(A) @ op(B) + beta * C
    pub fn cblas_sgemm(
        order: c_int,
        transA: c_int,
        transB: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

// ── Safe wrappers ────────────────────────────────────────────────

/// Element-wise multiply: out = a * b
pub fn vmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    unsafe {
        vDSP_vmul(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            n as Length,
        )
    }
}

/// Element-wise add: out = a + b
pub fn vadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    unsafe {
        vDSP_vadd(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            n as Length,
        )
    }
}

/// Element-wise subtract: out = b - a  (note vDSP_vsub order)
pub fn vsub(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    unsafe {
        vDSP_vsub(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            n as Length,
        )
    }
}

/// Scalar multiply: out = a * scalar
pub fn vsmul(a: &[f32], scalar: f32, out: &mut [f32]) {
    let n = a.len().min(out.len());
    unsafe { vDSP_vsmul(a.as_ptr(), 1, &scalar, out.as_mut_ptr(), 1, n as Length) }
}

/// Scalar add: out = a + scalar
pub fn vsadd(a: &[f32], scalar: f32, out: &mut [f32]) {
    let n = a.len().min(out.len());
    unsafe { vDSP_vsadd(a.as_ptr(), 1, &scalar, out.as_mut_ptr(), 1, n as Length) }
}

/// In-place scalar add: v[i] += scalar
pub fn vsadd_inplace(v: &mut [f32], scalar: f32) {
    let n = v.len();
    unsafe { vDSP_vsadd(v.as_ptr(), 1, &scalar, v.as_mut_ptr(), 1, n as Length) }
}

/// Fused multiply-add: out = a * scalar + b
pub fn vsma(a: &[f32], scalar: f32, b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    unsafe {
        vDSP_vsma(
            a.as_ptr(),
            1,
            &scalar,
            b.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            n as Length,
        )
    }
}

/// Max element
pub fn maxv(a: &[f32]) -> f32 {
    let mut result: f32 = f32::NEG_INFINITY;
    if !a.is_empty() {
        unsafe { vDSP_maxv(a.as_ptr(), 1, &mut result, a.len() as Length) }
    }
    result
}

/// Sum of all elements
pub fn sve(a: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    if !a.is_empty() {
        unsafe { vDSP_sve(a.as_ptr(), 1, &mut result, a.len() as Length) }
    }
    result
}

/// Element-wise exp: out[i] = exp(a[i])
pub fn expf(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len()) as c_int;
    unsafe { vvexpf(out.as_mut_ptr(), a.as_ptr(), &n) }
}

/// In-place exp: v[i] = exp(v[i])
pub fn expf_inplace(v: &mut [f32]) {
    let n = v.len() as c_int;
    unsafe { vvexpf(v.as_mut_ptr(), v.as_ptr(), &n) }
}

/// Element-wise reciprocal sqrt: out[i] = 1/sqrt(a[i])
pub fn rsqrtf(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len()) as c_int;
    unsafe { vvrsqrtf(out.as_mut_ptr(), a.as_ptr(), &n) }
}

/// Element-wise tanh: out[i] = tanh(a[i])
pub fn tanhf(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len()) as c_int;
    unsafe { vvtanhf(out.as_mut_ptr(), a.as_ptr(), &n) }
}

/// In-place tanh: v[i] = tanh(v[i])
pub fn tanhf_inplace(v: &mut [f32]) {
    let n = v.len() as c_int;
    unsafe { vvtanhf(v.as_mut_ptr(), v.as_ptr(), &n) }
}

/// Element-wise reciprocal: out[i] = 1/a[i]
pub fn recf(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len()) as c_int;
    unsafe { vvrecf(out.as_mut_ptr(), a.as_ptr(), &n) }
}

/// In-place reciprocal: v[i] = 1/v[i]
pub fn recf_inplace(v: &mut [f32]) {
    let n = v.len() as c_int;
    unsafe { vvrecf(v.as_mut_ptr(), v.as_ptr(), &n) }
}

/// Matrix transpose: C[n,m] = A[m,n]^T
/// A is m rows × n cols, C is n rows × m cols.
pub fn mtrans(a: &[f32], a_cols: usize, c: &mut [f32], c_cols: usize, m: usize, n: usize) {
    assert!(
        a.len() >= m * n,
        "mtrans: a too small ({} < {})",
        a.len(),
        m * n
    );
    assert!(
        c.len() >= n * m,
        "mtrans: c too small ({} < {})",
        c.len(),
        n * m
    );
    assert_eq!(a_cols, n, "mtrans: a_cols must equal n");
    assert_eq!(c_cols, m, "mtrans: c_cols must equal m");
    unsafe { vDSP_mtrans(a.as_ptr(), 1, c.as_mut_ptr(), 1, n as Length, m as Length) }
}

/// Sum of squares with stride: result = sum(a[i*stride]^2 for i in 0..n)
pub fn svesq_strided(a: &[f32], offset: usize, stride: usize, n: usize) -> f32 {
    let mut result: f32 = 0.0;
    if n > 0 {
        unsafe {
            vDSP_svesq(
                a.as_ptr().add(offset),
                stride as Stride,
                &mut result,
                n as Length,
            )
        }
    }
    result
}

/// Sum of squares (contiguous): result = sum(a[i]^2)
pub fn svesq(a: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    if !a.is_empty() {
        unsafe { vDSP_svesq(a.as_ptr(), 1, &mut result, a.len() as Length) }
    }
    result
}

/// In-place scale: v *= scalar (uses cblas_sscal)
pub fn sscal(v: &mut [f32], scalar: f32) {
    let n = v.len();
    if n > 0 {
        unsafe { cblas_sscal(n as c_int, scalar, v.as_mut_ptr(), 1) }
    }
}

// CBLAS constants
const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

/// Matrix multiply: C += alpha * A @ B^T
/// A is [m, k] row-major, B is [n, k] row-major (transposed), C is [m, n] row-major.
/// Beta=1.0 for accumulation into C.
pub fn sgemm_at(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= n * k);
    assert!(c.len() >= m * n);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            k as c_int,
            1.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

/// Matrix multiply: C = A^T @ B (no accumulation, overwrites C).
/// A is [k, m] row-major (transposed to [m, k]), B is [k, n] row-major, C is [m, n].
pub fn sgemm_ta(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    assert!(a.len() >= k * m);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            m as c_int,
            b.as_ptr(),
            n as c_int,
            0.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmul_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        vmul(&a, &b, &mut out);
        assert_eq!(out, [5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn vadd_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        vadd(&a, &b, &mut out);
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn vsub_order() {
        // vDSP_vsub: out = b - a (not a - b!)
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        vsub(&a, &b, &mut out);
        assert_eq!(out, [9.0, 18.0, 27.0]);
    }

    #[test]
    fn vsmul_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        vsmul(&a, 0.5, &mut out);
        assert_eq!(out, [0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn sve_sum() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sve(&a), 15.0);
    }

    #[test]
    fn maxv_finds_max() {
        let a = [1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(maxv(&a), 5.0);
    }

    #[test]
    fn expf_basic() {
        let a = [0.0, 1.0];
        let mut out = [0.0f32; 2];
        expf(&a, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn rsqrtf_basic() {
        let a = [4.0, 9.0, 16.0];
        let mut out = [0.0f32; 3];
        rsqrtf(&a, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((out[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn vsma_fused() {
        // out = a * 2.0 + b
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        vsma(&a, 2.0, &b, &mut out);
        assert_eq!(out, [12.0, 24.0, 36.0]);
    }
}
