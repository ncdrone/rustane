# V3 Update Thread — 2026-03-23
# Flow Draft #110

## Tweet 1 — HOOK
> Image: t1_hook.png

Update: DeepSeek-V3 running at 1.4 tok/s on my MacBook Pro.

671 billion parameters.
355 GB of weights.
One M4 Max, no cloud.

Here's how I got here.

## Tweet 2 — THE SYSTEM
> Image: t2_loop.png

I built a bash script that runs Claude Opus in headless mode. Each iteration:

Read the codebase + 43 research docs.
Pick one experiment.
Implement it.
Benchmark 3 times.
Commit or revert.
Write findings for the next agent.

It ran 40+ experiments while I slept.

## Tweet 3 — THE WINS
> Image: t3_wins.png

5 wins out of 40 experiments. 12.5% hit rate.

Every win was the same pattern — overlap work that uses different hardware:

Pipeline f16 conversion behind MLA compute. (+14%)
Overlap SSD reads behind shared FFN. (+34%)
Defer conversion to GPU phase only. (+6.6%)
Cache dense layers permanently. (+15.9%)
Batch attention with sgemm. (+6.1%)

## Tweet 4 — THE TURN
> Image: t4_gossip.png

The part that surprised me most: the gossip file.

Each agent writes what it learned. The next one reads it first.

"AMX achieves 150 GB/s per core, not 80."
"Apple BLAS already multi-threads large matrices."
"Shared FFN overlap is load-bearing."

40 experiments of compound knowledge.
No single agent could have found this.

## Tweet 5 — CLOSER
> Image: t5_closer.png

Right now a 1-trillion parameter model is downloading to this same laptop.

Kimi-K2. Same architecture. Same engine.

The optimization loop is still running.

The research lab is the laptop now.
