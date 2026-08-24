---
name: anomalib-training
description: >-
  Trains an anomalib model on a dataset via the Python API or CLI, including training on a custom folder-structured
  dataset with the Folder datamodule. Use when writing or debugging an anomalib training script/command, choosing
  Engine/Trainer arguments, or wiring a model + datamodule together. Do not use for adding a new model or
  datamodule from scratch (see anomalib-adding-a-model / anomalib-adding-a-datamodule), or for the tiled-ensemble
  or benchmarking pipelines (see anomalib-tiled-ensemble / anomalib-benchmarking).
license: Apache-2.0
---

# Training a Model on a Dataset

Training in anomalib always goes through `anomalib.engine.Engine`, which wraps a Lightning `Trainer`.

## Python API — standard benchmark dataset (MVTecAD)

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = MVTecAD(root="./datasets/MVTecAD", category="bottle", train_batch_size=32)
model = Patchcore()
engine = Engine(max_epochs=3)          # any Lightning Trainer kwarg can go here
engine.fit(model=model, datamodule=datamodule)
results = engine.test(model=model, datamodule=datamodule)
```

`Engine(**kwargs)` forwards unknown kwargs straight to the underlying Lightning `Trainer`
(`accelerator`, `devices`, `strategy`, `max_epochs`, `logger`, `callbacks`, `enable_checkpointing`,
`val_check_interval`, `barebones`, ...) — there is no separate "Trainer config object" to build.

Key `Engine` methods: `fit(model, datamodule=...)`, `train(...)` (fit + test in one call),
`test(model=None, datamodule=...)`, `predict(model=None, datamodule=..., dataset=..., data_path=...)`.
If `model`/`datamodule` are omitted from `test`/`predict`, the engine reuses the ones passed to `fit`.

## Python API — custom data with the Folder datamodule

Use `Folder` whenever your data is laid out as `root/normal_dir/*`, `root/abnormal_dir/*`, and
optionally `root/mask_dir/*` (segmentation masks) — no new dataset code needed:

```python
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine

datamodule = Folder(
    name="custom",             # required — used as the datamodule's display name
    root="./datasets/custom",
    normal_dir="good",         # required
    abnormal_dir="defect",     # optional: enables anomalous test/eval samples
    mask_dir="mask",           # optional: enables pixel-level (segmentation) evaluation
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
)
model = Padim()
engine = Engine(max_epochs=5)
engine.fit(model=model, datamodule=datamodule)
```

If you only have normal training images and unlabeled/normal-only test images, omit `abnormal_dir`
and `mask_dir` — `Folder` will still produce a valid train/test split via `normal_split_ratio`.

## CLI

```bash
# Standard dataset, defaults
anomalib train --model Patchcore --data anomalib.data.MVTecAD

# Override a datamodule field
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category transistor

# Override a trainer field
anomalib train --model anomalib.models.Padim --data anomalib.data.MVTecAD --trainer.max_epochs 3

# Custom Folder dataset from the CLI
anomalib train --model Padim --data anomalib.data.Folder \
  --data.name custom --data.root ./datasets/custom \
  --data.normal_dir good --data.abnormal_dir defect

# From a config file (jsonargparse; combine with any of the above overrides)
anomalib train --config path/to/config.yaml
```

CLI wiring lives in `src/anomalib/cli/cli.py` (`AnomalibCLI`); model/data classes are exposed as
jsonargparse subclass arguments, so `--model`/`--data` accept either a short name (`Padim`) or a full
class path (`anomalib.models.Padim`, `anomalib.data.Folder`).

## Choosing accelerator / devices

Pass standard Lightning kwargs to `Engine(...)`: `accelerator="gpu"|"cpu"|"xpu"`, `devices=1`. For
Intel XPU specifically, use `SingleXPUStrategy`/`XPUAccelerator` from `anomalib.engine`:

```python
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
engine = Engine(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
```

## Gotchas

- Not every model trains via gradient descent — training-free models (e.g. Padim) still go through
  `engine.fit(...)`; `Engine`/`Trainer` handles the single "epoch" needed to build their memory bank.
  You don't need special-case code for this.
- `Folder`'s `mask_dir` is what switches evaluation from image-level (classification) to pixel-level
  (segmentation) metrics — only pass it if you actually have per-pixel ground-truth masks.
- Results (checkpoints, logs, images) are written under `Engine`'s `default_root_dir` (`"results"` by
  default), nested by model/datamodule/category — check there first when debugging a run.

## Reviewer / self-check

- [ ] `Engine(...)` receives Trainer overrides as plain kwargs, not a hand-built `Trainer` object.
- [ ] Folder-dataset training specifies `normal_dir` and, if applicable, `abnormal_dir`/`mask_dir`
      matching the actual on-disk layout.
- [ ] CLI invocations use `anomalib.data.<Class>` / `anomalib.models.<Class>` paths that are actually
      exported (see `anomalib-adding-a-model` / `anomalib-adding-a-datamodule` for how exports work).
