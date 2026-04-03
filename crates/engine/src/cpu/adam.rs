//! AdamW optimizer step using vDSP.
//!
//! Standard AdamW with bias correction:
//!   m = beta1 * m + (1 - beta1) * grad
//!   v = beta2 * v + (1 - beta2) * grad²
//!   m_hat = m / (1 - beta1^t)
//!   v_hat = v / (1 - beta2^t)
//!   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

use super::vdsp;

pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Single AdamW step. Updates `param`, `m`, `v` in place.
/// `t` is the 1-indexed step count (for bias correction).
///
/// Uses two scratch buffers (tmp, tmp2) to avoid per-call heap allocations.
/// All bulk math goes through vDSP; only the final m/(sqrt(v)+eps) stays scalar.
pub fn step(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: u32,
    cfg: &AdamConfig,
) {
    let n = param.len();
    assert_eq!(grad.len(), n);
    assert_eq!(m.len(), n);
    assert_eq!(v.len(), n);
    assert!(t > 0);

    // Two reusable scratch buffers — no other heap allocations below.
    let mut tmp = vec![0.0f32; n];
    let mut tmp2 = vec![0.0f32; n];

    // m = beta1 * m + (1 - beta1) * grad
    // vsma: tmp = grad * (1-beta1) + m  ... but we need + beta1*m, not + m
    // Step 1: tmp = beta1 * m
    vdsp::vsmul(m, cfg.beta1, &mut tmp);
    // Step 2: m = (1-beta1) * grad + tmp   (vsma: out = a * scalar + b)
    vdsp::vsma(grad, 1.0 - cfg.beta1, &tmp, m);

    // v = beta2 * v + (1 - beta2) * grad²
    // Step 1: tmp = grad * grad
    vdsp::vmul(grad, grad, &mut tmp);
    // Step 2: tmp2 = beta2 * v
    vdsp::vsmul(v, cfg.beta2, &mut tmp2);
    // Step 3: v = (1-beta2) * tmp + tmp2   (vsma: out = a * scalar + b)
    vdsp::vsma(&tmp, 1.0 - cfg.beta2, &tmp2, v);

    // Bias correction
    let bc1 = 1.0 / (1.0 - cfg.beta1.powi(t as i32));
    let bc2 = 1.0 / (1.0 - cfg.beta2.powi(t as i32));

    // m_hat = m * bc1  → reuse tmp
    vdsp::vsmul(m, bc1, &mut tmp);
    // v_hat = v * bc2  → reuse tmp2
    vdsp::vsmul(v, bc2, &mut tmp2);

    // update = m_hat / (sqrt(v_hat) + eps)
    // The division has no clean vDSP pattern, so stays scalar.
    for i in 0..n {
        let update = tmp[i] / (tmp2[i].sqrt() + cfg.eps);
        param[i] -= cfg.lr * (update + cfg.weight_decay * param[i]);
    }
}

/// Fused single-pass AdamW step — zero allocation, one memory traversal.
///
/// Same math as `step()` + Metal shader, but processes each element exactly once
/// instead of 7+ separate vDSP passes. This reduces memory traffic from ~76 bytes/elem
/// to ~28 bytes/elem (read grad/m/v/param once, write m/v/param once).
///
/// `grad_scale` is applied inline (matches Metal shader's `gscale` parameter).
/// For no scaling, pass 1.0.
pub fn step_fused(
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
    grad_scale: f32,
) {
    let n = param.len();
    debug_assert_eq!(grad.len(), n);
    debug_assert_eq!(m.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert!(t > 0);

    let bc1 = 1.0f32 / (1.0 - beta1.powi(t as i32));
    let bc2 = 1.0f32 / (1.0 - beta2.powi(t as i32));
    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;

    // Single pass: read each array once, write param/m/v once.
    // LLVM auto-vectorizes this loop with NEON instructions.
    for i in 0..n {
        let g = grad[i] * grad_scale;
        let mi = beta1 * m[i] + one_minus_beta1 * g;
        let vi = beta2 * v[i] + one_minus_beta2 * g * g;
        m[i] = mi;
        v[i] = vi;
        let m_hat = mi * bc1;
        let v_hat = vi * bc2;
        let update = m_hat / (v_hat.sqrt() + eps);
        param[i] -= lr * (update + weight_decay * param[i]);
    }
}

/// Fused AdamW step that also writes a cached transposed view inline.
///
/// `param` is laid out row-major as [rows * cols]. `transpose` receives the
/// updated values in row-major [cols * rows] order.
pub fn step_fused_transpose(
    param: &mut [f32],
    transpose: &mut [f32],
    rows: usize,
    cols: usize,
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    grad_scale: f32,
) {
    let n = rows * cols;
    debug_assert_eq!(param.len(), n);
    debug_assert_eq!(transpose.len(), n);
    debug_assert_eq!(grad.len(), n);
    debug_assert_eq!(m.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert!(t > 0);

    let bc1 = 1.0f32 / (1.0 - beta1.powi(t as i32));
    let bc2 = 1.0f32 / (1.0 - beta2.powi(t as i32));
    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;

    for row in 0..rows {
        let row_off = row * cols;
        for col in 0..cols {
            let i = row_off + col;
            let g = grad[i] * grad_scale;
            let mi = beta1 * m[i] + one_minus_beta1 * g;
            let vi = beta2 * v[i] + one_minus_beta2 * g * g;
            m[i] = mi;
            v[i] = vi;
            let m_hat = mi * bc1;
            let v_hat = vi * bc2;
            let update = m_hat / (v_hat.sqrt() + eps);
            let new_param = param[i] - lr * (update + weight_decay * param[i]);
            param[i] = new_param;
            transpose[col * rows + row] = new_param;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_moves_toward_minimum() {
        // param=5.0, grad pointing toward 0 → param should decrease
        let mut param = [5.0f32; 4];
        let grad = [1.0f32; 4]; // positive grad = param too high
        let mut m = [0.0f32; 4];
        let mut v = [0.0f32; 4];
        let cfg = AdamConfig::default();

        let initial: f32 = param.iter().sum();
        for t in 1..=10 {
            step(&mut param, &grad, &mut m, &mut v, t, &cfg);
        }
        let final_sum: f32 = param.iter().sum();
        assert!(
            final_sum < initial,
            "param should decrease: {initial} -> {final_sum}"
        );
    }

    #[test]
    fn zero_grad_only_weight_decay() {
        let mut param = [1.0f32; 2];
        let grad = [0.0f32; 2];
        let mut m = [0.0f32; 2];
        let mut v = [0.0f32; 2];
        let cfg = AdamConfig {
            weight_decay: 0.1,
            ..Default::default()
        };

        let before = param[0];
        step(&mut param, &grad, &mut m, &mut v, 1, &cfg);
        // With zero grad, only weight decay acts → param shrinks
        assert!(param[0] < before, "weight decay should shrink param");
    }

    #[test]
    fn momentum_accumulates() {
        let mut param = [0.0f32; 2];
        let grad = [1.0f32; 2];
        let mut m = [0.0f32; 2];
        let mut v = [0.0f32; 2];
        let cfg = AdamConfig {
            weight_decay: 0.0,
            ..Default::default()
        };

        step(&mut param, &grad, &mut m, &mut v, 1, &cfg);
        // m should be (1-beta1)*grad = 0.1
        assert!((m[0] - 0.1).abs() < 1e-6, "m[0]={}", m[0]);
        // v should be (1-beta2)*grad² = 0.001
        assert!((v[0] - 0.001).abs() < 1e-6, "v[0]={}", v[0]);
    }
}
