//! Test: fcntl(F_RDADVISE) prefetch hint doesn't error or crash.
//!
//! What was added: ExpertLoader::prefetch_experts() calls fcntl(F_RDADVISE) before pread
//! to give the kernel a head start on DMA for expert pages.
//!
//! Invariant: prefetch_experts is non-blocking and doesn't fail even for cached/uncached pages.

use std::io::Write;

#[test]
fn prefetch_does_not_error() {
    // Create a temp file with some data
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.bin");
    let expert_size = 1024;
    let num_experts = 8;
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&vec![0xABu8; expert_size * num_experts]).unwrap();
    }

    let layout = expert_pager::ExpertFileLayout {
        expert_size,
        num_experts,
    };
    let loader = expert_pager::ExpertLoader::open(path.to_str().unwrap(), layout).unwrap();

    // Prefetch all experts — should not panic or error
    loader.prefetch_experts(&[0, 1, 2, 3, 4, 5, 6, 7]);

    // Now pread should still work normally
    let mut buf = vec![0u8; expert_size];
    let n = loader.load_expert(0, &mut buf).unwrap();
    assert_eq!(n, expert_size);
    assert_eq!(buf[0], 0xAB);
    eprintln!("prefetch_experts + load_expert: OK");
}
