//! SSD streaming expert pager with pread-based parallel loading.
//!
//! Core abstraction for loading MoE expert weights on demand:
//! - ExpertPool: IOSurface-backed ring buffer with LRU eviction
//! - ExpertLoader: parallel pread loading (8 threads via std::thread::scope)
//! - ExpertPrefetcher: cross-layer gate similarity predictor

pub mod pool;
pub mod loader;
pub mod convert;
pub mod prefetch;

pub use pool::ExpertPool;
pub use loader::ExpertLoader;
pub use prefetch::ExpertPrefetcher;
