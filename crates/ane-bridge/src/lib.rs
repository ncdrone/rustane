//! Safe Rust FFI bindings to Apple Neural Engine private APIs.
//!
//! Wraps the `ane` crate and extends it for training:
//! - Channel-interleaved IOSurface writes (dynamic weight pipeline)
//! - NEON-accelerated f32↔f16 conversion
//! - Training kernel patterns (DynMatmul, DualDynMatmul, fused SDPA/FFN)

// Re-export the ane crate's public API
pub use ane;

/// Write a single f32 value as fp16 directly to an IOSurface at a byte offset.
/// Bypasses the full-buffer f32→fp16 conversion for surgical updates.
///
/// # Safety
/// Caller must ensure offset is within the IOSurface allocation.
/// IOSurface must not be concurrently accessed by ANE during this call.
pub unsafe fn write_f16_at(surface: &ane::TensorData, element_offset: usize, value: f32) {
    let surf = surface.surface();
    surf.lockWithOptions_seed(objc2_io_surface::IOSurfaceLockOptions(0), std::ptr::null_mut());
    let base = surf.baseAddress().as_ptr().cast::<u16>();
    let fp16 = half::f16::from_f32(value);
    std::ptr::write(base.add(element_offset), fp16.to_bits());
    surf.unlockWithOptions_seed(objc2_io_surface::IOSurfaceLockOptions(0), std::ptr::null_mut());
}

/// Write a contiguous f32 slice as fp16 directly to an IOSurface starting at element_offset.
/// Much faster than full-buffer as_f32_slice_mut() for partial updates.
///
/// # Safety
/// Caller must ensure offset + data.len() is within the IOSurface allocation.
pub unsafe fn write_f16_bulk_at(surface: &ane::TensorData, element_offset: usize, data: &[f32]) {
    let surf = surface.surface();
    surf.lockWithOptions_seed(objc2_io_surface::IOSurfaceLockOptions(0), std::ptr::null_mut());
    let base = surf.baseAddress().as_ptr().cast::<u16>();
    for (i, &val) in data.iter().enumerate() {
        let fp16 = half::f16::from_f32(val);
        std::ptr::write(base.add(element_offset + i), fp16.to_bits());
    }
    surf.unlockWithOptions_seed(objc2_io_surface::IOSurfaceLockOptions(0), std::ptr::null_mut());
}

/// Stage activation into pre-staged IOSurface: writes IC fp16 values at stride SP.
/// This is the hot-path operation for single-token decode: one fp16 per channel.
///
/// # Safety
/// IOSurface must have been pre-staged with weights via copy_from_f32.
/// Caller must ensure channels * sp fits within allocation.
pub unsafe fn stage_activation_fp16(
    surface: &ane::TensorData,
    activation: &[f32],
    channels: usize,
    sp: usize,
) {
    let surf = surface.surface();
    surf.lockWithOptions_seed(objc2_io_surface::IOSurfaceLockOptions(0), std::ptr::null_mut());
    let base = surf.baseAddress().as_ptr().cast::<u16>();
    for c in 0..channels {
        let fp16 = half::f16::from_f32(activation[c]);
        std::ptr::write(base.add(c * sp), fp16.to_bits());
    }
    surf.unlockWithOptions_seed(objc2_io_surface::IOSurfaceLockOptions(0), std::ptr::null_mut());
}
