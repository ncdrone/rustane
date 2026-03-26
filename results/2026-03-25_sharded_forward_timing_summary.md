# 600M Sharded Forward Timing Summary

Date: 2026-03-25

Command run:

```bash
cargo test -p engine --test bench_sharded_forward_timing --release -- --ignored --nocapture
```

Artifacts:

- JSON: [results/sharded_forward_timing/600m_sharded_forward_timing.json](/Users/USER/RustRover-Projects/rustane/results/sharded_forward_timing/600m_sharded_forward_timing.json)

## Key result

Per-layer sharded training forward at 600M with `FFN_SHARDS=4` measured:

- total layer wall: about `56-58 ms`
- FFN section wall: about `50-52 ms`
- FFN ANE wall sum: about `7.5-7.7 ms`
- FFN CPU sum: about `42-44 ms`

This means the remaining forward regression is **not primarily ANE compute**. It is dominated by **host-side CPU work inside the serial shard loop**.

## Main buckets

Per shard, the dominant cost is:

- `w2` staging: about `7.5-8.3 ms`

Secondary costs:

- `w13` staging: about `1.8-2.0 ms`
- gate CPU: about `0.66-0.71 ms`
- ANE dispatches (`w13` + `w2`): about `1.8-2.0 ms` combined
- merge + cache store: about `0.27-0.31 ms`

## Interpretation

The training-path FFN forward regression is still mostly caused by:

1. serial execution over 4 shards
2. repeated host-side IOSurface staging for each shard
3. especially the `w2` input staging path

The previously suspected full-buffer clone in `full_model.rs` was removed and did **not** materially change sweep timings. It was overhead, but not the main bottleneck.

## Practical conclusion

If we want `FFN_SHARDS=4` forward time to improve meaningfully, the next target is:

- parallelizing the sharded FFN forward loop, or
- reducing/restructuring per-shard `w2` staging cost

The backward work is no longer the primary regression.
