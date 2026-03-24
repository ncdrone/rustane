//! ExpertLoader: parallel pread-based expert weight loading from SSD.
//!
//! Uses libc::pread for thread-safe, zero-seek reads from expert weight files.
//! Parallel loading via std::thread::scope (simpler than GCD FFI).

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;

/// Expert weight file format metadata.
#[derive(Clone, Debug)]
pub struct ExpertFileLayout {
    /// Size of each expert's weights in bytes.
    pub expert_size: usize,
    /// Total number of experts in the file.
    pub num_experts: usize,
}

impl ExpertFileLayout {
    /// Byte offset for a given expert_id.
    pub fn offset(&self, expert_id: u32) -> i64 {
        (expert_id as usize * self.expert_size) as i64
    }
}

/// Loads expert weights from a file using pread (thread-safe, no seeking).
pub struct ExpertLoader {
    fd: File,
    pub layout: ExpertFileLayout,
}

// SAFETY: ExpertLoader uses pread which is thread-safe (no file position mutation).
// The fd is only used via as_raw_fd() + pread — safe for concurrent access.
unsafe impl Sync for ExpertLoader {}

impl ExpertLoader {
    /// Open an expert weight file.
    /// F_NOCACHE: bypass page cache for direct SSD→user DMA, reducing cache
    /// eviction of shared FFN weights and DRAM bandwidth contention.
    pub fn open(path: &str, layout: ExpertFileLayout) -> io::Result<Self> {
        let fd = File::open(path)?;
        unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_NOCACHE, 1); }
        Ok(Self { fd, layout })
    }

    /// Load a single expert's weights into the provided buffer.
    /// Uses pread for thread-safe access (no file position mutation).
    pub fn load_expert(&self, expert_id: u32, buf: &mut [u8]) -> io::Result<usize> {
        assert!(buf.len() >= self.layout.expert_size);
        let offset = self.layout.offset(expert_id);

        let n = unsafe {
            libc::pread(
                self.fd.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                self.layout.expert_size,
                offset,
            )
        };

        if n < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(n as usize)
        }
    }

    /// Load a partial region of an expert's weights.
    /// `offset_within`: byte offset from start of this expert's data.
    /// `len`: number of bytes to read.
    pub fn load_expert_partial(&self, expert_id: u32, buf: &mut [u8], offset_within: usize, len: usize) -> io::Result<usize> {
        assert!(buf.len() >= len);
        let base = self.layout.offset(expert_id);
        let n = unsafe {
            libc::pread(
                self.fd.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                len,
                base + offset_within as i64,
            )
        };
        if n < 0 { Err(io::Error::last_os_error()) } else { Ok(n as usize) }
    }

    /// Load multiple experts in parallel using std::thread::scope.
    /// Returns the number of bytes loaded per expert.
    pub fn load_experts_parallel(
        &self,
        expert_ids: &[u32],
        buffers: &mut [Vec<u8>],
        num_threads: usize,
    ) -> io::Result<()> {
        assert_eq!(expert_ids.len(), buffers.len());

        // Split work across threads
        let chunk_size = (expert_ids.len() + num_threads - 1) / num_threads;

        std::thread::scope(|s| {
            let mut handles = Vec::new();

            for (chunk_ids, chunk_bufs) in expert_ids
                .chunks(chunk_size)
                .zip(buffers.chunks_mut(chunk_size))
            {
                let fd_raw = self.fd.as_raw_fd();
                let layout = &self.layout;

                handles.push(s.spawn(move || {
                    for (id, buf) in chunk_ids.iter().zip(chunk_bufs.iter_mut()) {
                        let offset = layout.offset(*id);
                        buf.resize(layout.expert_size, 0);
                        let n = unsafe {
                            libc::pread(
                                fd_raw,
                                buf.as_mut_ptr() as *mut libc::c_void,
                                layout.expert_size,
                                offset,
                            )
                        };
                        if n < 0 {
                            return Err(io::Error::last_os_error());
                        }
                    }
                    Ok(())
                }));
            }

            for h in handles {
                h.join().expect("thread panicked")?;
            }
            Ok(())
        })
    }
}
