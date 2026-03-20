//! Integration test: pread loads correct bytes from expert weight files.
//!
//! Creates a temporary expert weight file, loads via pread, verifies correctness.
//! Tests both single and parallel loading.

use expert_pager::loader::{ExpertFileLayout, ExpertLoader};
use std::io::Write;

fn create_test_file(num_experts: usize, expert_size: usize) -> (tempfile::NamedTempFile, Vec<Vec<u8>>) {
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    let mut expected = Vec::new();

    for expert_id in 0..num_experts {
        let data: Vec<u8> = (0..expert_size)
            .map(|i| ((expert_id * 137 + i * 31) % 256) as u8)
            .collect();
        file.write_all(&data).expect("write");
        expected.push(data);
    }
    file.flush().expect("flush");

    (file, expected)
}

#[test]
fn pread_loads_correct_bytes() {
    let expert_size = 4096;
    let num_experts = 8;
    let (file, expected) = create_test_file(num_experts, expert_size);

    let layout = ExpertFileLayout { expert_size, num_experts };
    let loader = ExpertLoader::open(file.path().to_str().unwrap(), layout).unwrap();

    for expert_id in 0..num_experts {
        let mut buf = vec![0u8; expert_size];
        let n = loader.load_expert(expert_id as u32, &mut buf).unwrap();
        assert_eq!(n, expert_size);
        assert_eq!(buf, expected[expert_id], "expert {expert_id} data mismatch");
    }
}

#[test]
fn parallel_loads_all_correct() {
    let expert_size = 8192;
    let num_experts = 16;
    let (file, expected) = create_test_file(num_experts, expert_size);

    let layout = ExpertFileLayout { expert_size, num_experts };
    let loader = ExpertLoader::open(file.path().to_str().unwrap(), layout).unwrap();

    let expert_ids: Vec<u32> = (0..num_experts as u32).collect();
    let mut buffers: Vec<Vec<u8>> = vec![Vec::new(); num_experts];

    loader.load_experts_parallel(&expert_ids, &mut buffers, 4).unwrap();

    for (id, buf) in expert_ids.iter().zip(buffers.iter()) {
        assert_eq!(buf.len(), expert_size);
        assert_eq!(buf, &expected[*id as usize], "expert {id} parallel load mismatch");
    }
}

#[test]
fn subset_load() {
    let expert_size = 2048;
    let num_experts = 32;
    let (file, expected) = create_test_file(num_experts, expert_size);

    let layout = ExpertFileLayout { expert_size, num_experts };
    let loader = ExpertLoader::open(file.path().to_str().unwrap(), layout).unwrap();

    // Load only experts 5, 10, 20
    let ids = vec![5, 10, 20];
    let mut buffers = vec![Vec::new(); 3];
    loader.load_experts_parallel(&ids, &mut buffers, 2).unwrap();

    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(buffers[i], expected[id as usize]);
    }
}
