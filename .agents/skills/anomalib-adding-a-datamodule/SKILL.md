---
name: anomalib-adding-a-datamodule
description: >-
  Adds a new dataset/datamodule to anomalib under src/anomalib/data/. Use when wiring a new data source into the
  AnomalibDataset/AnomalibDataModule base classes, exporting it so anomalib.data.<Name> and the CLI/config
  (jsonargparse) can discover it, and adding matching tests. Do not use for model architecture work (see
  anomalib-adding-a-model) or for training an existing datamodule (see anomalib-training).
license: Apache-2.0
---

# Adding a New Dataset / DataModule

anomalib splits data support into two layers per source, both under `src/anomalib/data/`:

- `datasets/image/<name>.py` — a `torch`-facing `AnomalibDataset` subclass (one dataset = one split).
- `datamodules/image/<name>.py` — a Lightning-facing `AnomalibDataModule` subclass that owns train/val/test
  dataloaders and split logic.

(Use `datasets/video/` and `datamodules/video/`, or `depth/`, for other modalities — the pattern is identical.)

## Base classes to implement against

- `AnomalibDataset` — `src/anomalib/data/datasets/base/image.py`
  - `__init__(self, augmentations=None)` — call via `super().__init__(...)`.
  - You must build a `pandas.DataFrame` and assign it to `self.samples`. Required columns include
    `image_path` and `split`; segmentation datasets also need `mask_path`. Set
    `samples.attrs["task"]` to `"classification"` or `"segmentation"`.
  - `collate_fn` defaults to `ImageBatch.collate`; override only for non-image batch types.
- `AnomalibDataModule` — `src/anomalib/data/datamodules/base/image.py`
  - Only abstract method you must implement: `_setup(self, _stage=None) -> None`, where you set
    `self.train_data` and `self.test_data` (and `self.val_data` if you don't rely on the base class's
    `val_split_mode` machinery).
  - The base class already implements `setup()`, `train_dataloader()`, `val_dataloader()`,
    `test_dataloader()`, and `from_config()` (jsonargparse subclass integration) — do not override these
    unless the data source genuinely needs custom dataloader construction.
  - Constructor should accept and forward: `train_batch_size`, `eval_batch_size`, `num_workers`,
    `train_augmentations` / `val_augmentations` / `test_augmentations` / `augmentations`,
    `test_split_mode` / `test_split_ratio`, `val_split_mode` / `val_split_ratio`, `seed`.

## Reference: MVTecAD (standard benchmark-style dataset)

- `src/anomalib/data/datasets/image/mvtecad.py` — `MVTecADDataset(AnomalibDataset)`; builds `self.samples`
  via the `make_mvtec_ad_dataset(root_category, split, extensions)` helper.
- `src/anomalib/data/datamodules/image/mvtecad.py` — `MVTecAD(AnomalibDataModule)`; `_setup()` constructs
  `MVTecADDataset(split=Split.TRAIN, root=self.root, category=self.category)` for train/test, and
  `prepare_data()` downloads the dataset archive if missing.

## Reference: Folder (generic custom-folder dataset — use this as your template for ad hoc data)

- `src/anomalib/data/datasets/image/folder.py` — `FolderDataset(AnomalibDataset)`, built via the
  `make_folder_dataset(...)` helper: collects filenames/labels from directories, builds the `samples`
  DataFrame, and attaches mask paths to abnormal samples when `mask_dir` is given.
- `src/anomalib/data/datamodules/image/folder.py` — `Folder(AnomalibDataModule)` constructor (key args):

  ```python
  Folder(
      name: str,                                              # required, becomes datamodule.name
      normal_dir: str | Path | Sequence[str | Path],           # required
      root: str | Path | None = None,
      abnormal_dir: str | Path | Sequence[str | Path] | None = None,
      normal_test_dir: str | Path | Sequence[str | Path] | None = None,  # separate normal images for test set
      mask_dir: str | Path | Sequence[str | Path] | None = None,   # for segmentation masks
      normal_split_ratio: float = 0.2,
      extensions: tuple[str] | None = None,
      train_batch_size: int = 32,
      eval_batch_size: int = 32,
      num_workers: int = 8,
      train_augmentations: Transform | None = None,
      val_augmentations: Transform | None = None,
      test_augmentations: Transform | None = None,
      augmentations: Transform | None = None,
      test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
      test_split_ratio: float = 0.2,
      val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
      val_split_ratio: float = 0.5,
      seed: int | None = None,
  )
  ```

  Note: `test_split_ratio` (inherited from `AnomalibDataModule`) controls the fallback split when
  `test_split_mode` is not `FROM_DIR`. `normal_split_ratio` is a Folder-specific field for splitting
  normal images into train/test subsets.

Use `Folder` directly (no new code needed) whenever the data is already laid out as
`root/normal_dir/*`, `root/abnormal_dir/*`, optionally `root/mask_dir/*`. Only write a brand-new
dataset/datamodule pair when the data needs custom parsing logic `Folder` can't express.

## Writing a brand-new datamodule (skeleton)

```python
# src/anomalib/data/datasets/image/my_dataset.py
from anomalib.data.datasets.base import AnomalibDataset

class MyDataset(AnomalibDataset):
    def __init__(self, root=None, augmentations=None, split=None):
        super().__init__(augmentations=augmentations)
        self.samples = make_my_dataset_samples(root=root, split=split)  # build the DataFrame yourself
```

```python
# src/anomalib/data/datamodules/image/my_dataset.py
from pathlib import Path

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.my_dataset import MyDataset
from anomalib.data.utils import Split

class MyDataModule(AnomalibDataModule):
    def __init__(
        self,
        root: str | Path = "./datasets/MyDataset",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=None,
        augmentations=None,
        test_split_mode=None,
        test_split_ratio: float = 0.2,
        val_split_mode=None,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
            num_workers=num_workers, train_augmentations=train_augmentations,
            val_augmentations=val_augmentations, test_augmentations=test_augmentations,
            augmentations=augmentations, test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio, val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio, seed=seed,
        )
        self.root = Path(root)

    def _setup(self, _stage=None) -> None:
        self.train_data = MyDataset(split=Split.TRAIN, root=self.root)
        self.test_data = MyDataset(split=Split.TEST, root=self.root)

    def prepare_data(self) -> None:
        ...  # optional: download/validate on rank-zero
```

## Registration — how the datamodule becomes discoverable

1. Export from the image (or video/depth) package `__init__.py` —
   `src/anomalib/data/datamodules/image/__init__.py`: add the import and `__all__` entry there first.

2. Then add the import and `__all__` entry in `src/anomalib/data/__init__.py`, alongside the existing
   `datamodules.image` import block:

```python
from .datamodules.image import (
    ...,
    MyDataModule,
)
```

Once exported, it is usable as `anomalib.data.MyDataModule`, and from the CLI:
`anomalib train --model Patchcore --data anomalib.data.MyDataModule --data.root ./datasets/mine`.

## Tests

Add `tests/unit/data/datamodule/image/test_my_dataset.py` following the pattern in
`tests/unit/data/datamodule/image/test_mvtec_ad.py`: a `datamodule` fixture that instantiates the
datamodule against a generated dummy dataset, calls `prepare_data()` + `setup()`, then reuses the
shared assertions in `tests/unit/data/datamodule/base/image.py` (batch shapes, split non-overlap, etc.).

### Add a dummy dataset generator (required for a new `DataFormat`)

Real datasets aren't checked into the repo — tests generate synthetic data on the fly via
`tests/helpers/data.py`. If your new datamodule corresponds to a new `DataFormat` value (i.e. it isn't
just `Folder` under another name), you must add a matching generator method:

1. Add the format to `ImageDataFormat` (or `VideoDataFormat`) in
   `src/anomalib/data/datamodules/image/__init__.py` (or `.../video/__init__.py`),
   e.g. `MY_DATASET = "my_dataset"`.
2. Implement `_generate_dummy_my_dataset_dataset(self) -> None` on `DummyImageDatasetGenerator`
   (`tests/helpers/data.py`) for image datasets, or `DummyVideoDatasetGenerator` for video datasets —
   the method name must be `_generate_dummy_{data_format.value}_dataset`;
   `DummyDatasetGenerator.generate_dataset()` dispatches to it via `getattr`. Build the on-disk layout
   your datamodule expects using the low-level `DummyImageGenerator`
   (`tests/helpers/data.py::DummyImageGenerator`) for image datasets, or `DummyVideoGenerator` for
   video datasets:

   ```python
   def _generate_dummy_my_dataset_dataset(self) -> None:
       """Generate dummy MyDataset dataset in a temporary directory."""
       dataset_category = "dummy"
       # normal train/test images
       for split in ("train", "test"):
           path = self.dataset_root / dataset_category / split / self.normal_category
           num_images = self.num_train if split == "train" else self.num_test
           for i in range(num_images):
               image_filename = path / f"{i:03}.png"
               self.image_generator.generate_image(label=LabelName.NORMAL, image_filename=image_filename)

       # abnormal test images + masks
       path = self.dataset_root / dataset_category / "test" / self.abnormal_category
       mask_path = self.dataset_root / dataset_category / "ground_truth" / self.abnormal_category
       for i in range(self.num_test):
           image_filename = path / f"{i:03}.png"
           mask_filename = mask_path / f"{i:03}_mask.png"
           self.image_generator.generate_image(LabelName.ABNORMAL, image_filename, mask_filename)
   ```

   See `_generate_dummy_mvtecad_dataset` and `_generate_dummy_folder_dataset` in the same file for the
   two canonical layouts (category-per-split-per-class vs. flat normal/abnormal/mask dirs) — mirror
   whichever matches your real dataset's directory structure.

3. The session-scoped `dataset_path` fixture in `tests/conftest.py` dispatches image formats to
   `DummyImageDatasetGenerator` and video formats to `DummyVideoDatasetGenerator` (skipping `folder`/
   `tabular`, which tests construct manually) — you only need to implement the `_generate_dummy_*`
   method on the appropriate generator class.
4. In your datamodule test, consume the generated data via the shared fixture:

   ```python
   @pytest.fixture()
   def datamodule(dataset_path: Path) -> MyDataModule:
       dm = MyDataModule(root=dataset_path / "my_dataset")
       dm.prepare_data()
       dm.setup()
       return dm
   ```

If your datamodule is just a thin wrapper around `Folder` (same on-disk convention, different
defaults), you don't need a new `DataFormat`/generator — reuse `_generate_dummy_folder_dataset` and
construct your datamodule directly against its output directory.

## Gotchas

- `self.samples` must be assigned (not mutated in place before assignment) — the `samples` setter on
  `AnomalibDataset` validates required columns and paths.
- Don't skip `samples.attrs["task"]` — post-processing and metrics branch on `"classification"` vs
  `"segmentation"`.
- Prefer `Folder` over a new dataset class whenever the on-disk layout is a plain normal/abnormal/mask
  directory split — writing a new class is only needed for non-standard parsing.
- Tests never touch real downloaded datasets. If you add a new `DataFormat`, you must also add a
  `_generate_dummy_<format>_dataset` method on the appropriate generator (`DummyImageDatasetGenerator`
  for image formats, `DummyVideoDatasetGenerator` for video formats) — otherwise the shared
  `dataset_path` fixture (`tests/conftest.py`) will raise `NotImplementedError` for that format.

## Reviewer / self-check before opening a PR

- [ ] `AnomalibDataset` subclass sets `self.samples` (DataFrame with required columns + `task` attr).
- [ ] `AnomalibDataModule` subclass implements `_setup()` only; no unnecessary overrides of
      `train_dataloader`/`val_dataloader`/`test_dataloader`.
- [ ] Datamodule exported from `src/anomalib/data/__init__.py` and `__all__` updated.
- [ ] `anomalib.data.MyDataModule` resolves and works from the CLI `--data` flag.
- [ ] Unit tests added under `tests/unit/data/datamodule/`.
- [ ] If a new `DataFormat` was introduced, a matching `_generate_dummy_*_dataset` method was added to
      `DummyImageDatasetGenerator` in `tests/helpers/data.py`.
