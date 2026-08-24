---
name: anomalib-adding-a-model
description: >-
  Adds a new anomaly-detection model to anomalib under src/anomalib/models/. Use when implementing a new model
  architecture (image or video), wiring it into the AnomalibModule base class, registering it so get_model() and
  the CLI/config (jsonargparse) can discover it, and adding matching tests. Do not use for changes to existing
  model internals only (no new model), or for datamodule/dataset work (see anomalib-adding-a-datamodule).
license: Apache-2.0
---

# Adding a New Model

Models live under `src/anomalib/models/image/<model_name>/` (or `.../video/<model_name>/` for video models).
Every model is a `LightningModule` subclass of `AnomalibModule`
(`src/anomalib/models/components/base/anomalib_module.py`) that wraps a plain `torch.nn.Module`.

## Reference implementation

Read `src/anomalib/models/image/padim/` first — it is the smallest complete example:

- `torch_model.py` — `PadimModel(nn.Module)`: pure PyTorch forward pass. In training mode it returns
  intermediate embeddings; in eval mode it returns an `InferenceBatch(pred_score=..., anomaly_map=...)`.
- `anomaly_map.py` — `AnomalyMapGenerator(nn.Module)`: post-processing (score/map computation) kept out of
  `torch_model.py` for clarity. Not all models need a separate file for this.
- `lightning_model.py` — `Padim(MemoryBankMixin, AnomalibModule)`: the Lightning-facing wrapper. This is the
  class users construct (`Padim()`), pass to `Engine`, and reference from CLI/config as `anomalib.models.Padim`.
- `__init__.py` — exports the Lightning class only: `from .lightning_model import Padim`.
- `README.md` — usage + benchmark notes (see `anomalib-model-doc-sync`/`anomalib-model-sample-image-export`
  skills if those exist for docs work).

Only mix in `MemoryBankMixin` (`src/anomalib/models/components/base/memory_bank_module.py`) if the model
accumulates a memory bank / feature bank across training (as PaDiM and PatchCore do). Most models just subclass
`AnomalibModule` directly.

## Required members on your `AnomalibModule` subclass

```python
from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer

class MyModel(AnomalibModule):
    def __init__(self, some_param: int = 1, pre_processor=True, post_processor=True,
                 evaluator=True, visualizer=True) -> None:
        super().__init__(pre_processor=pre_processor, post_processor=post_processor,
                          evaluator=evaluator, visualizer=visualizer)
        self.model = MyTorchModel(some_param=some_param)

    @property
    def trainer_arguments(self) -> dict:
        """Default Trainer overrides for this model, e.g. {"max_epochs": 1} for training-free models."""
        return {"max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    def training_step(self, batch: Batch, *args, **kwargs):
        # Training-free models (e.g. Padim) still return a dummy loss for Lightning.
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def validation_step(self, batch: Batch, *args, **kwargs):
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    def configure_optimizers(self):
        # Return None for training-free / statistical models (Padim does this).
        return None
```

`AnomalibModule` provides working defaults you can override only when the model needs something different:

- `configure_pre_processor(image_size=None)` — resize + ImageNet normalization.
- `configure_post_processor()` — thresholding/normalization for `ONE_CLASS` models.
- `configure_evaluator()` — image/pixel AUROC and F1 metrics.
- `configure_visualizer()` — default `ImageVisualizer`.

`pre_processor` / `post_processor` / `evaluator` / `visualizer` constructor args each accept an instance, `True`
(use the configured default), or `False` (disable).

## Registration — how the model becomes discoverable

1. Add the export in `src/anomalib/models/image/__init__.py` (or `video/__init__.py`):

   ```python
   from .my_model import MyModel
   ```

   and add `"MyModel"` to `__all__` and to the `Available Models` docstring list at the top of the file.

2. That's the only registration step. `list_models()` and `get_model()`
   (`src/anomalib/models/__init__.py`) discover models by walking `AnomalibModule.__subclasses__()` and
   matching on `cls.__name__` (case-insensitive) — there is no separate name-string registry to update.
3. `get_model()` also accepts a dict/`DictConfig`/`Namespace` with `class_path` + `init_args`, restricted to
   modules listed in `ALLOWED_MODULES` (same file) — you don't need to touch that set for models already under
   `anomalib.models`.
4. Once exported, the model is usable as `anomalib.models.MyModel`, from `get_model("MyModel")`, and from the
   CLI: `anomalib train --model MyModel --data anomalib.data.MVTecAD`.

## Tests

Add `tests/unit/models/image/test_my_model.py` (or `.../video/...`). At minimum:

- `get_model("MyModel")` (and the equivalent PascalCase/snake_case variants) returns a `MyModel` instance.
- `model.trainer_arguments` is a `dict` and `model.learning_type` is the expected `LearningType`.
- A synthetic batch through `training_step`/`validation_step` produces the expected shapes (batch size,
  action/anomaly-map dimensions) without crashing.

See `tests/unit/models/test_model_utils.py` for the `get_model()` instantiation pattern used across the suite.

## Gotchas

- Keep `torch_model.py` importable and testable without Lightning — the `nn.Module.forward` contract
  (train-mode returns raw tensors/embeddings, eval-mode returns `InferenceBatch`) is what the base class and
  visualizers expect. Don't put Lightning-specific logic there.
- Training side effects (checkpointing, timing, compression, visualization) belong in
  `src/anomalib/callbacks/`-style Lightning callbacks or existing hooks — not inline in the model's
  `training_step`.
- Public constructor arguments must stay explicit and typed (no untyped `**kwargs` passthrough) so
  `jsonargparse` can expose them on the CLI/config surface.
- If the model becomes part of the public API, update the model's `README.md` and, if one exists, the matching
  page under `docs/source/markdown/guides/reference/models/`.

## Reviewer / self-check before opening a PR

- [ ] Model exported from `src/anomalib/models/image/__init__.py` (or `video/__init__.py`) and `__all__` updated.
- [ ] `trainer_arguments`, `learning_type`, `training_step`, `validation_step`, `configure_optimizers` implemented.
- [ ] `get_model("MyModel")` and `anomalib.models.MyModel` both resolve.
- [ ] Unit tests added under `tests/unit/models/`.
- [ ] `README.md` added in the model folder.
- [ ] `prek run --all-files` and `pytest tests/unit/models/ -k my_model` pass.
