---
name: perf
description: Muratori-style performance analysis — mechanical sympathy, cache behavior, branch analysis, and benchmark design. Use when optimizing hot paths or investigating throughput.
argument-hint: "[module, function, or 'full']"
---

Perform a Casey Muratori-style performance analysis of `$ARGUMENTS`. The goal is to understand what the hardware is actually doing, not what the code looks like.

## Phase 1: Identify the hot path

Read the target code and identify:

1. **The innermost loop** — where does the CPU spend >90% of its time?
2. **Iterations per second** — what's the theoretical throughput?
3. **Work per iteration** — how many FP ops, memory loads, branches?

For Tempura, the hot paths are typically:
- `annealer.rs`: the main SA loop (propose → evaluate → accept/reject)
- `parallel.rs`: per-replica SA loops + swap phase
- `population.rs`: per-member sweep loops + resampling phase
- `math.rs`: acceptance functions (Metropolis, Barker, fast_exp)

## Phase 2: Mechanical sympathy analysis

For each hot-path function, analyze:

### Memory access pattern
- **Sequential?** (good for prefetcher) or **random?** (cache miss per access)
- **Working set size** — does it fit in L1 (32KB), L2 (256KB), L3 (8MB)?
- **Struct layout** — AoS vs SoA? Are hot fields co-located?

### Branch behavior
- **Predictable?** Accept/reject at high T is ~50/50 (worst case for predictor)
- **Branchless alternatives?** `math::metropolis_branchless` exists for this reason
- **Cold path separation** — are diagnostics/recording behind a branch that's usually not-taken?

### Arithmetic intensity
- **FP operations per cache line loaded** — higher is better
- **Division or transcendentals** — each `exp()` is ~20 cycles, `ln()` is ~30 cycles
- **fast_exp opportunity** — can we use Schraudolph's trick (4% error) instead of libm exp?

## Phase 3: Benchmark design

If benchmarks don't exist for this code path, design them:

```rust
// In benches/ — use Criterion
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_[name](c: &mut Criterion) {
    // Setup outside the benchmark loop
    let [setup] = ...;

    c.bench_function("[name]", |b| {
        b.iter(|| {
            black_box([hot_path_call]);
        })
    });
}
```

Benchmark rules:
- Measure the **hot path in isolation** (not setup/teardown)
- Use `black_box()` to prevent dead code elimination
- Compare against a **baseline** (e.g., branchless vs branching)
- Report in **proposals/second** or **iterations/second**

## Phase 4: Optimization recommendations

For each finding, provide:

1. **What**: the specific bottleneck
2. **Why**: how it impacts throughput (cycles, cache misses, branch mispredicts)
3. **Fix**: concrete code change with expected speedup
4. **Risk**: correctness implications (does this break determinism? change numerical results?)

## Output format

```
## Performance Report: [target]

### Hot path: [function] ([estimated iterations/sec])

### Bottlenecks (ranked by impact)
1. [bottleneck] — est. [X]% of runtime
   Fix: [specific change]
   Speedup: ~[N]x
   Risk: [none / breaks determinism / changes results within tolerance]

### Already optimal
- [aspects that are well-optimized]

### Benchmark commands
cargo bench --bench [name]
```
