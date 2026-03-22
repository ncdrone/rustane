//! SSD streaming expert pager with pread-based parallel loading.
//!
//! Core abstraction for loading MoE expert weights on demand:
//! - ExpertPool: Least-Stale eviction (by minimum last_used_layer)
//! - ExpertLoader: parallel pread loading (8 threads via std::thread::scope)
//! - ExpertPrefetcher: disabled (no-op stub, research showed -18%)

pub mod pool;
pub mod loader;
pub mod convert;
pub mod prefetch;

pub use pool::ExpertPool;
pub use loader::{ExpertLoader, ExpertFileLayout};
pub use prefetch::ExpertPrefetcher;
