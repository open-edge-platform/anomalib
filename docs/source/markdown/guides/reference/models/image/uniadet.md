# UniADet

[UniADet](https://arxiv.org/abs/2601.05552) detects anomalies without text prompts. A frozen vision transformer
feeds separate classification and segmentation prototype heads. In few-shot mode, the detector also compares
patches with normal target images.

## Architecture

```{eval-rst}
.. image:: ../../../../../images/uniadet/architecture.svg
    :alt: UniADet architecture
```

Backbones: OpenAI CLIP ViT-L/14 (QuickGELU), DINOv2-R ViT-L/14, and DINOv3 ViT-L/16.

## Usage

No training is required with a pretrained head. In an Anomalib checkout, sign in with approved access to the
[Hugging Face checkpoints](https://huggingface.co/csgaobb/UniADet), then import a head:

```bash
uv run --extra huggingface python -m tools.uniadet.author \
  --backbone dinov2 --source-dataset visa --output results/UniADet/author_weights
```

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import UniADet

model = UniADet(
    backbone="dinov2", weights="results/UniADet/author_weights/dinov2_visa/head.pt", protocol="author", k_shot=0
)
Engine().test(model=model, datamodule=MVTecAD(root="datasets/MVTecAD", category="bottle"))
```

Set `k_shot=4` to use the same head with normal references from the training split or `few_shot_source`.
`Engine.fit` calibrates thresholds without updating weights or normalizing scores.
Full checkpoints retain the reference bank and reject changed reference settings; load a head to select new references.
Optional auxiliary head training uses `python -m tools.uniadet.train` and `protocol="legacy"`.

## Benchmarks and deployment

Author-head evaluations on MVTec AD and VisA match all [96 published metric means](https://github.com/open-edge-platform/anomalib/blob/main/results/UniADet/reproduction/paper_precision_comparison.csv)
within 0.07 percentage points. DINOv2-R four-shot image AUROC is 98.73% on MVTec AD and 96.92% on VisA.
Paper AP is trapezoidal precision-recall area, not Anomalib's standard average precision.
These results reproduce evaluation with pretrained heads, not the authors' training runs.

TorchInferencer, Engine, ONNX, and OpenVINO FP32/FP16 were checked at batches 1 and 3 for all three author
backbones at zero and four shots. Use the fixed encoder resolution and initialize references before few-shot
export. Use FP32 or FP16 for score and map agreement with Torch. Weight-only INT8, INT8_PTQ, and INT8_ACQ
can exceed the `0.05` maximum map error limit on author graphs.
NNCF's ACQ ranking also has an unresolved zero-output limitation; aggregate AUROC alone does not certify parity.
See the [model README](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uniadet)
for reproduction commands, timings and deployment reports.

The [authors confirm](https://github.com/gaobb/UniADet/issues/7#issuecomment-7075731743) that Apache-2.0 covers
original UniADet code and heads. Checkpoints are not bundled. Backbone weights retain separate licenses:
CLIP (MIT), DINOv2 (Apache-2.0), and DINOv3 (Meta's DINOv3 License).

## Sample results

Input, ground truth, and anomaly overlays on VisA for an independently trained DINOv2-R head at four shots.

```{eval-rst}
.. image:: ../../../../../images/uniadet/results/0.png
    :alt: VisA chewing-gum anomaly

.. image:: ../../../../../images/uniadet/results/1.png
    :alt: VisA PCB missing-pin anomaly

.. image:: ../../../../../images/uniadet/results/2.png
    :alt: VisA capsule anomaly
```

Photos and masks: VisA, Zou et al. (2022), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## API

```{eval-rst}
.. automodule:: anomalib.models.image.uniadet.lightning_model
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.uniadet.torch_model
   :members:
   :show-inheritance:
```
