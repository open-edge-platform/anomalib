---
name: model-doc-sync
description: Keep anomalib model READMEs, docs pages, image assets, and benchmark/result references in sync
---

# Model README and Docs Sync

Use this skill when updating model documentation, benchmark tables, sample-result images, or docs reference pages for
an anomalib model.

## Purpose and scope

Use this skill to keep model READMEs, docs pages, images, and benchmark/sample references aligned.

## Request changes when

- a model README changes but the matching docs page is left stale;
- image references point to missing assets;
- benchmark tables do not match committed artifacts;
- sample-result sections imply coverage that the repo does not actually contain.

## Keep these surfaces aligned

- `src/anomalib/models/**/README.md`
- `docs/source/markdown/guides/reference/models/**`
- `docs/source/images/**`
- benchmark and run artifacts under `results/`

## Canonical repo paths

### Model READMEs

- Image model READMEs usually live at `src/anomalib/models/image/<model>/README.md`.
- Video model READMEs may live at `src/anomalib/models/video/<model>/README.md`.
- There are also category-level READMEs such as `src/anomalib/models/image/README.md` and `src/anomalib/models/video/README.md`.

### Model docs pages

- Image model docs pages usually live at `docs/source/markdown/guides/reference/models/image/<model>.md`.
- Video model docs pages usually live at `docs/source/markdown/guides/reference/models/video/<model>.md`.
- Keep `docs/source/markdown/guides/reference/models/**/index.md` up-to-date.
- The docs pages should also contain architecture image and description.

### Image assets

- Model images typically live at `docs/source/images/<model>/`.
- Common patterns include `architecture.*` and `results/0.png`, `results/1.png`, `results/2.png`.
- Some models use nonstandard names or multiple architecture images, for example `docs/source/images/cs_flow/`.

### Result artifacts

- Measured artifacts may exist under `results/<ModelName>/...`.
- Treat `results/` as evidence for benchmark/sample claims, not as a place to invent values from partial runs.

## Required workflow

### 1. Inspect before editing

For the target model:

1. Read the model README.
2. Read the matching docs page if it exists.
3. Inspect `docs/source/images/<model>/` or the repo-specific variant actually used.
4. Check whether measured artifacts already exist in `results/`.
5. Check whether referenced sample-result images exist and are still valid.

Never update only one of README or docs when both exist.

### 2. Keep these sections synchronized

If present in the README, verify that the docs page and assets do not contradict it:

- title and model name
- description
- architecture section and image references
- usage section
- benchmark section
- sample results section
- TODO notes about missing benchmarks or images

Docs pages do not need to duplicate the full README, but they must stay consistent with it.

### 3. Benchmark update rules

- Prefer the repository's existing benchmarking workflow, starting with `tools/experimental/benchmarking/`.
- Use measured results only.
- Source values from committed artifacts such as files under `results/`.
- Do not fabricate averages, rows, or per-category scores.
- If only part of the benchmark is complete, fill only the supported values and leave the rest clearly blank or TODO.
- If benchmarking is still in progress, say so explicitly.

### 4. Sample image rules

- Only add sample-result images from completed model outputs.
- Confirm the referenced output is not degenerate or misleading.
- Do not publish obviously broken masks, empty masks, NaN-driven outputs, or placeholder images as final examples.
- Copy or export valid final images into the matching `docs/source/images/...` location.
- If fewer than three valid sample images exist, use a precise TODO note instead of broken links or bad examples.

## README conventions

- Prefer repository-consistent image references such as `/docs/source/images/<model>/...` when the README already uses that pattern.
- If sample-result images are missing, leave a TODO note instead of a broken link.
- Keep benchmark/sample wording consistent with the actual artifacts checked into the repo.

## Docs page conventions

- Use the real docs-page path depth when referencing images.
- Many model docs pages act as reference wrappers around module docs, so keep them aligned with the README without forcing full duplication.
- If the docs page is a lightweight API/reference page, still ensure it does not contradict README claims about architecture, benchmarks, or results.

## Validation checklist

Before finishing:

1. README and docs page agree on benchmark and sample status.
2. Every referenced image path exists.
3. Benchmark tables match committed artifacts.
4. Any TODO left behind is accurate and specific.
5. Any helper script remains narrow and model-specific unless a general solution is clearly justified.

## Known repo pitfalls

- Some model/image paths do not match exactly, for example `csflow` vs `cs_flow`.
- Some models have architecture images but not three sample-result images.
- Some models have results artifacts under `results/` without a fully synced docs surface yet.
- Docs pages can drift from README changes even when image paths still resolve.

## Review prompts

- Did the README, docs page, and image assets all get checked together?
- Do benchmark values come from committed measured artifacts?
- Are sample-result images valid and non-misleading?
- Are any missing assets called out explicitly instead of hidden behind broken links?
- Is there any model-specific naming quirk that needs a manual path check?

## Reviewer checklist

- Check README and docs page together.
- Check every referenced image path.
- Check benchmark claims against committed artifacts.
- Check TODO notes for accuracy.
