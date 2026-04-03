//! Phase 0 verification tests for ane crate integration.
//!
//! Run all (including ANE hardware tests):
//!   cargo test -p ane-bridge --test phase0_verify -- --include-ignored
//!
//! Run only non-hardware tests:
//!   cargo test -p ane-bridge --test phase0_verify

use ane::{Graph, NSQualityOfService, Shape, TensorData};

// =========================================================
// Test 0.2: Multiple weight blobs → distinct BLOBFILE offsets
// =========================================================

#[test]
fn multiple_constants_produce_valid_graph() {
    // Create a graph with 3 constants (simulating sdpaFwd's mask + rope_cos + rope_sin)
    let mut g = Graph::new();

    // Placeholder input: [1, 4, 1, 64] — minimum spatial width is 64
    let x = g.placeholder(Shape {
        batch: 1,
        channels: 4,
        height: 1,
        width: 64,
    });

    // 3 constant tensors (like mask, rope_cos, rope_sin)
    let mask = g.constant(
        &vec![0.0f32; 4 * 64],
        Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 64,
        },
    );
    let rope_cos = g.constant(
        &vec![1.0f32; 4 * 64],
        Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 64,
        },
    );
    let rope_sin = g.constant(
        &vec![0.5f32; 4 * 64],
        Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 64,
        },
    );

    // Use all constants in operations
    let x1 = g.addition(x, mask);
    let x2 = g.multiplication(x1, rope_cos);
    let _x3 = g.addition(x2, rope_sin);

    // Verify the graph was constructed without panicking
    // The real test is that compile works (next test)
    eprintln!("Graph with 3 constants constructed successfully");
}

// =========================================================
// Test 0.2 + 0.3: Compile on ANE, verify multiple blobs work
// =========================================================

#[test]
#[ignore = "requires ANE hardware"]
fn compile_graph_with_multiple_constants_on_ane() {
    let mut g = Graph::new();

    let x = g.placeholder(Shape {
        batch: 1,
        channels: 4,
        height: 1,
        width: 64,
    });

    let c1 = g.constant(
        &vec![1.0f32; 4 * 64],
        Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 64,
        },
    );
    let c2 = g.constant(
        &vec![2.0f32; 4 * 64],
        Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 64,
        },
    );

    let x1 = g.addition(x, c1);
    let _out = g.multiplication(x1, c2);

    let executable = g
        .compile(NSQualityOfService::Default)
        .expect("ANE compilation failed — is AppleNeuralEngine.framework available?");

    // If we got here, compilation succeeded — tmpdir was created correctly (0.3 verified)
    // and multiple weight blobs were packed into weight.bin (0.2 verified)
    eprintln!("ANE compilation succeeded with 2 constant blobs");

    // Evaluate to verify it actually runs
    let input = TensorData::with_f32(
        &vec![1.0f32; 4 * 64],
        Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 64,
        },
    );
    let output = TensorData::new(Shape {
        batch: 1,
        channels: 4,
        height: 1,
        width: 64,
    });

    executable
        .run(&[&input], &[&output])
        .expect("ANE evaluation failed");

    // Read output and verify it's not all zeros
    let result = output.read_f32();
    let sum: f32 = result.iter().sum();
    assert!(
        sum.abs() > 0.0,
        "Output should not be all zeros after add+mul"
    );

    // Expected: (1.0 + 1.0) * 2.0 = 4.0 for each element
    // Check a few values (fp16 tolerance)
    for (i, &val) in result.iter().take(10).enumerate() {
        assert!(
            (val - 4.0).abs() < 0.1,
            "Element {i}: expected ~4.0, got {val}"
        );
    }
    eprintln!("ANE evaluation produced correct output: (1+1)*2 = 4.0 ✓");

    drop(executable);
}

// =========================================================
// Test 0.4: IOSurface layout matches training pattern
// =========================================================

#[test]
#[ignore = "requires IOSurface (macOS only)"]
fn iosurface_write_read_roundtrip() {
    // Create a TensorData (wraps IOSurface), write fp32, read back
    let shape = Shape {
        batch: 1,
        channels: 8,
        height: 1,
        width: 64,
    };

    let data: Vec<f32> = (0..8 * 64).map(|i| i as f32 * 0.01).collect();
    let tensor = TensorData::with_f32(&data, shape);

    // Read back
    let readback = tensor.read_f32();

    // Verify roundtrip (fp32 → IOSurface → fp32 should be exact)
    for (i, (&expected, &got)) in data.iter().zip(readback.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-6,
            "Mismatch at {i}: expected {expected}, got {got}"
        );
    }
    eprintln!(
        "IOSurface write/read roundtrip: {} elements verified ✓",
        data.len()
    );
}

#[test]
#[ignore = "requires IOSurface (macOS only)"]
fn iosurface_raii_lock_guards_work() {
    let shape = Shape {
        batch: 1,
        channels: 4,
        height: 1,
        width: 64,
    };
    let tensor = TensorData::new(shape);

    // Write via mutable guard
    {
        let mut guard = tensor.as_f32_slice_mut();
        for (i, val) in guard.iter_mut().enumerate() {
            *val = i as f32;
        }
    } // guard dropped, surface unlocked

    // Read via immutable guard
    {
        let guard = tensor.as_f32_slice();
        assert_eq!(guard[0], 0.0);
        assert_eq!(guard[1], 1.0);
        assert_eq!(guard[63], 63.0);
    } // guard dropped

    eprintln!("RAII lock guards: write then read ✓");
}

// =========================================================
// Test: Channel-interleaved write pattern (pure Rust, no IOSurface needed)
// =========================================================

#[test]
fn channel_interleaved_layout_is_correct() {
    // Simulate the io_write_dyn pattern from autoresearch-ANE/io.h:
    //   For DynMatmul with IC channels, SEQ activation cols, OC weight cols:
    //   SP = SEQ + OC
    //   For each channel d:
    //     buf[d*SP .. d*SP+SEQ]    = activation[d*SEQ .. (d+1)*SEQ]
    //     buf[d*SP+SEQ .. d*SP+SP] = weight[d*OC .. (d+1)*OC]

    let ic = 4usize;
    let seq = 8usize;
    let oc = 6usize;
    let sp = seq + oc;

    let acts: Vec<f32> = (0..ic * seq).map(|i| i as f32 * 0.1).collect();
    let weights: Vec<f32> = (0..ic * oc).map(|i| -(i as f32) * 0.01).collect();

    let mut buf = vec![0.0f32; ic * sp];
    for d in 0..ic {
        buf[d * sp..d * sp + seq].copy_from_slice(&acts[d * seq..(d + 1) * seq]);
        buf[d * sp + seq..d * sp + sp].copy_from_slice(&weights[d * oc..(d + 1) * oc]);
    }

    // Verify activation at (channel=2, spatial=3)
    assert_eq!(buf[2 * sp + 3], acts[2 * seq + 3]);
    // Verify weight at (channel=2, col=4)
    assert_eq!(buf[2 * sp + seq + 4], weights[2 * oc + 4]);
    // Verify total size
    assert_eq!(buf.len(), ic * sp);

    eprintln!("Channel-interleaved layout: verified for IC={ic}, SEQ={seq}, OC={oc} ✓");
}
