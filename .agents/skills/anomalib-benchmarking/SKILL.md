---
name: anomalib-benchmarking
description: >-
  Runs the anomalib benchmarking pipeline to train/evaluate a grid of model + dataset (+ category) combinations
  and collect metrics into a results CSV. Use when comparing multiple models/datasets/categories in one sweep, or
  authoring/editing a benchmark config YAML. Do not use for training a single model (see anomalib-training) or
  the tiled-ensemble pipeline (see anomalib-tiled-ensemble). For turning measured results into README/docs
  benchmark tables, see the benchmark-and-docs-refresh skill.
license: Apache-2.0
---

# Using the Benchmarking Pipeline

The benchmarking pipeline runs a grid of model/dataset/category combinations end-to-end (train + test)
and writes measured metrics to a CSV — use it to produce real, reproducible numbers rather than hand-editing
benchmark tables.

## Code locations

- `src/anomalib/pipelines/benchmark/pipeline.py` — `Benchmark`: top-level pipeline; picks
  `SerialRunner` or `ParallelRunner` based on configured accelerators and `torch.cuda.device_count()`.
- `src/anomalib/pipelines/benchmark/generator.py` — `BenchmarkJobGenerator`: expands the config
  (including `grid:` entries) into individual jobs.
- `src/anomalib/pipelines/benchmark/job.py` — `BenchmarkJob`: runs one model/dataset combination,
  times it, and saves results.
- `tools/experimental/benchmarking/benchmark.py` — thin CLI wrapper around `Benchmark`.
- `tools/experimental/benchmarking/sample.yaml` — example config to copy from.

## Running it

```bash
# Via the tools wrapper
python tools/experimental/benchmarking/benchmark.py --config tools/experimental/benchmarking/sample.yaml

# Via the anomalib CLI (registered pipeline subcommand)
anomalib benchmark --config tools/experimental/benchmarking/sample.yaml
```

## Config structure

```yaml
accelerator:
  - cuda
  - cpu

benchmark:
  seed: 42
  model:
    class_path:
      grid: [Padim, Patchcore]
  data:
    class_path: MVTecAD
    init_args:
      category:
        grid:
          - bottle
          - capsule
```

Any field can use `grid: [...]` to sweep multiple values — the generator produces the Cartesian
product of every `grid` field as separate jobs (here: 2 models × 2 categories = 4 jobs). Non-grid
fields are held constant across all jobs. `data.class_path` / `model.class_path` follow the same
`anomalib.data.*` / `anomalib.models.*` resolution as everywhere else in the repo (see
`anomalib-training`).

## Where results go

`BenchmarkJob.save(...)` writes one row per job into:

```bash
runs/benchmark/<timestamp>/results.csv
```

(`<timestamp>` is generated when results are saved via `BenchmarkJob.save()`, e.g.
`2026-08-24-10_30_00`.) Each row includes the
model/dataset/category combination and the measured metrics — this is the file to consume when
building or refreshing README/docs benchmark tables.

There is also a separate, narrower helper `tools/benchmark_mebin.py` that writes to
`results/mebin_benchmark.csv` for a specific benchmarking use case — prefer the pipeline above unless
you specifically need that script's behavior.

## Gotchas

- A `grid` sweep multiplies job count fast — check the Cartesian product size before launching a large
  sweep (e.g. 5 models × 10 categories = 50 full train+test runs).
- `accelerator: [cuda, cpu]` creates one runner per entry, so **every model/category combination runs
  once per accelerator** (doubling the total job count). This is not a device-pool selector — if you
  only want to benchmark on GPU, use `accelerator: [cuda]`.
- Never hand-write or infer numbers into README/docs benchmark tables — always source them from a
  `results.csv` produced by an actual run of this pipeline.

## Reviewer / self-check

- [ ] Config's `grid` fields produce the intended, bounded set of jobs (no accidental huge sweep).
- [ ] `model.class_path` / `data.class_path` values resolve to real exported classes.
- [ ] Benchmark run completed and `runs/benchmark/<timestamp>/results.csv` exists before citing numbers
      anywhere else.
- [ ] Test reference: `tests/integration/pipelines/test_benchmark.py` for how the pipeline is invoked
      programmatically if debugging job generation.
