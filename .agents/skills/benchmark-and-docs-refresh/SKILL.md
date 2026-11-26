---
name: benchmark-and-docs-refresh
description: Run or continue model benchmarks, collect measured results, and refresh README/docs benchmark sections from generated artifacts. Use when benchmark tables in model docs need to be created, updated, or corrected.
---

# Benchmark and Docs Refresh

Use this skill to update benchmark sections in model documentation from real benchmark outputs.

## Scope

This skill focuses on:

- running or continuing benchmarks
- collecting benchmark CSV results from `results/`
- updating benchmark tables in model READMEs
- updating matching docs pages when benchmark status changes

It does not own sample image export. Use `model-sample-image-export` for that.

## Request changes when

- incomplete benchmark coverage is presented;
- README or docs benchmark status drifts from the actual run state.

## Preferred Benchmark Workflow

Always prefer:

- `tools/experimental/benchmarking/benchmark.py`

with an appropriate config file.

If the stock benchmark path is insufficient for a specific model:

1. derive a small helper script from the benchmark workflow
2. keep it model-specific unless multiple models clearly need the same pattern
3. save measurable outputs such as CSV files under `results/`

## Required Evidence

Only publish benchmark values when they come from actual artifacts, for example:

- `results/<model>_benchmark.csv`
- benchmark-generated CSV files under `runs/` or `results/`
- model-specific run outputs that clearly record the measured metrics

Never infer missing values.

## Update Rules

When refreshing benchmark tables:

1. Read the target README and matching docs page first.
2. Read the benchmark artifact source.
3. Fill only the shot-settings and metrics that actually exist.
4. Leave unavailable rows blank or TODO.
5. Update status wording if the benchmark is still partial or still running.

## Table Conventions

Common sections to refresh:

- `### Image-Level AUC`
- `### Pixel-Level AUC`
- `### Image F1 Score`
- `### Pixel F1 Score`

If a README only contains placeholders, replace only the rows supported by measured results.

## Docs Synchronization Rules

If the README benchmark state changes, update the matching docs page under:

- `docs/source/markdown/guides/reference/models/image/<model>.md`
- `docs/source/markdown/guides/reference/models/video/<model>.md`

The docs page may stay shorter than the README, but it must not contradict it.

## Quality Checks

Before finishing:

1. Confirm the benchmark artifact still exists.
2. Confirm copied values exactly match the artifact.
3. Confirm averages are computed from measured values only.
4. Confirm incomplete rows remain clearly incomplete.
5. Confirm README/docs wording matches reality.

## Reviewer checklist

- Check that the artifact exists.
- Check that every copied value matches.
- Check that partial runs are labeled clearly.
- Check README and docs wording for consistency.

## Repo-Specific Notes

- Some benchmark jobs in this repo may require derived helper scripts.
- Some long runs are better continued in tmux/background sessions.
- A benchmark can be complete enough to fill a subset of rows without justifying all rows.
- Never replace TODOs with fabricated numbers.
