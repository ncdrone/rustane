# K2 Optimization Gossip
# Each iteration reads this FIRST and appends findings at the end.

## Current State
tok/s: 0.005 | baseline: 0.005 | wins: 0 | experiments: 2

## Bottleneck (update after profiling)
Not yet profiled. Expected similar to V3 but with:
- 50% less MLA compute (64 heads vs 128)
- 50% more expert data per layer (384 experts, 9 GB files)
- Smaller backbone (23.4 GB vs 34 GB)

## V3 Learnings (apply these — don't re-discover)
From 40+ V3 experiments, these patterns are PROVEN:
- Overlap wins when using DIFFERENT hardware resources (CPU || SSD, CPU || GPU)
- f16 CPU compute is a dead end (needs Metal)
- Apple BLAS already multi-threads internally for large matrices
- Allocation elimination is invisible at this scale (<50µs)
- Metal per-dispatch overhead is ~0.15ms — batching required
- Shared FFN overlap is load-bearing — removing it exposes pread to critical path
- AMX achieves ~150 GB/s per P-core (not 80 GB/s)

## Category Stats
- All V3 CPU-side categories are likely exhausted for K2 too (same code path)
- K2-specific opportunities: 384 experts (different cache dynamics), 64 heads (different MLA profile), n_group=1 (different routing behavior)

## Dead Ends (from V3, likely apply to K2)
- Manual NEON sgemv (AMX is 3x faster)
- f16 from mmap (page table overhead kills it)
- Explicit BLAS parallelization (Accelerate handles it)
- Metal overlap with CPU sgemv (Metal API has CPU overhead)

## Suggested Next
1. Profile per-component K2 timing (MLA with 64 heads — is O proj still dominant?)
2. Check expert pread latency (9 GB files vs V3's 5.5 GB — does file size matter?)
3. Test if cached-dense with only 1 dense layer still helps (K2 has 1 vs V3's 3)

## Iteration Log
[init] K2 first token: 414s cold prefill. Output "Hello\n". Correct.
[init] K2 10-token warm: "Paris France France..." — repetition from greedy + INT4 quantization. 0.005 tok/s.
