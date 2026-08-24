---
name: anomalib-tiled-ensemble
description: >-
  Runs and configures the anomalib tiled-ensemble pipeline, which trains/evaluates one model per image tile and
  merges results (with optional seam smoothing) for high-resolution anomaly detection. Use when the user wants to
  train with image tiling, mentions "tiled ensemble", or needs to tune tiling/stride/seam-smoothing config. Do not
  use for regular single-model training (see anomalib-training) or the multi-model benchmarking pipeline (see
  anomalib-benchmarking).
license: Apache-2.0
---

# Using the Tiled Ensemble Pipeline

The tiled-ensemble pipeline splits each image into overlapping tiles, trains/evaluates a separate model
instance per tile position, then merges tile predictions (with optional seam smoothing) back into a
full-image anomaly map. Use it for high-resolution images where a single model can't see fine detail at
a manageable input size.

## Code locations

- `src/anomalib/pipelines/tiled_ensemble/train_pipeline.py` — `TrainTiledEnsemble`: composes the job
  graph (per-tile training, per-tile prediction, merge, seam smoothing, statistics) and picks
  `SerialRunner` or `ParallelRunner` based on the configured accelerator and available CUDA devices.
- `src/anomalib/pipelines/tiled_ensemble/test_pipeline.py` — `EvalTiledEnsemble`: runs
  inference/evaluation for an already-trained ensemble.
- `src/anomalib/pipelines/tiled_ensemble/components/` — individual job implementations (model
  training, prediction, merging, smoothing, metrics).
- `src/anomalib/pipelines/tiled_ensemble/components/utils/ensemble_engine.py` — `TiledEnsembleEngine`,
  an `Engine` subclass that customizes per-tile checkpoint/workspace naming.

## Running it

```bash
python tools/tiled_ensemble/train.py --config tools/tiled_ensemble/ens_config.yaml
python tools/tiled_ensemble/eval.py  --config tools/tiled_ensemble/ens_config.yaml \
                                     --root results/Padim/MVTecAD/bottle/v0
```

`train.py` runs `TrainTiledEnsemble().run()` which includes evaluation after training;
`eval.py` runs `EvalTiledEnsemble` to **re-run** evaluation against an existing results directory
(`--root`) — use it only when you want to evaluate again without retraining.

## Config structure

Start from `tools/tiled_ensemble/ens_config.yaml` and adjust the fields you need:

```yaml
seed: 42
accelerator: "cuda" # or "cpu"
default_root_dir: "results"

tiling:
  image_size: [256, 256] # size the full image is resized to before tiling
  tile_size: [128, 128] # size of each tile
  stride: 128 # tile stride; stride < tile_size gives overlapping tiles

normalization_stage: image
thresholding_stage: image

data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: ./datasets/MVTecAD
    category: bottle
    train_batch_size: 32
    eval_batch_size: 32
    num_workers: 8
    val_split_mode: from_test
    test_split_mode: from_dir

SeamSmoothing:
  apply: False
  sigma: 2
  width: 0.1

TrainModels:
  model:
    class_path: Padim
```

Key fields:

- `tiling.tile_size` / `tiling.stride` — the core tiling geometry; `stride < tile_size` produces
  overlap that `SeamSmoothing` then blends.
- `data.class_path` — any **image** `anomalib.data.*` datamodule that yields `ImageBatch` (see
  `anomalib-training` / `anomalib-adding-a-datamodule`). Video and depth datamodules are **not
  supported** — the tiled collater uses `ImageBatch.collate` internally.
- `TrainModels.model.class_path` — the model class trained per tile; any registered **image**
  `anomalib.models.*` model works (see `anomalib-adding-a-model`). Video models are not compatible
  with the image tiler.
- `SeamSmoothing.apply` — set `True` when you want Gaussian smoothing around tile boundaries; it affects
  both overlapping and non-overlapping tiles (`stride <= tile_size`), so enable it only when desired.

For a worked reference invocation with a full config, see
`tests/integration/pipelines/test_tiled_ensemble.py`.

## Gotchas

- The pipeline trains **one model instance per tile position**, not one shared model — total training
  cost scales with the number of tiles, not just image count. Budget accordingly before scaling up
  `tile_size`/`stride` combinations.
- `accelerator: cuda` with multiple visible GPUs triggers `ParallelRunner`, which trains multiple tile
  jobs concurrently across devices — set `accelerator: cpu` (or restrict visible devices) for
  deterministic single-process runs while debugging a config.
- Eval (`eval.py`) needs `--root` pointing at the exact output directory produced by the matching
  training run; it does not re-derive this automatically.
- `data.init_args` **must** include `val_split_mode` and `test_split_mode` — the pipeline reads these
  directly from the config before datamodule defaults are applied, and will raise `KeyError` if missing.

## Reviewer / self-check

- [ ] `tiling.tile_size`/`stride` chosen relative to `tiling.image_size` (stride ≤ tile_size).
- [ ] `data.class_path` and `TrainModels.model.class_path` both resolve to real, exported classes.
- [ ] `SeamSmoothing.apply` is intentional given whether tiles overlap.
- [ ] Training run completed and its `results/...` path is used correctly as `eval.py --root`.
