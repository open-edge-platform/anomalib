# UniADet

An independent implementation of **One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection**
([paper](https://arxiv.org/abs/2601.05552), Bin-Bin Gao and Chengjie Wang).

UniADet learns separate normal/anomaly prototypes for classification and segmentation at five frozen transformer
blocks. The head is trained on **labeled anomalies from another dataset**, not on the target category. At deployment,
zero-shot inference uses those prototypes; few-shot inference also compares patches with normal target references.

## Architecture

![Independent UniADet architecture diagram](/docs/source/images/uniadet/architecture.svg)

The implementation uses `AnomalibModule`, `ImageBatch`, `InferenceBatch`, standard preprocessing/postprocessing callbacks,
and the existing export API. The target learning type is always zero-shot or few-shot. There is no Engine training-mode
workaround: auxiliary supervision runs through a standalone Lightning training tool.

| Encoder              | timm identifier                               | Input size | Head parameters |
| -------------------- | --------------------------------------------- | ---------- | --------------- |
| OpenAI CLIP ViT-L/14 | `vit_large_patch14_clip_quickgelu_336.openai` | 518        | 15,360          |
| DINOv2-R ViT-L/14    | `vit_large_patch14_reg4_dinov2.lvd142m`       | 518        | 20,480          |
| DINOv3 ViT-L/16      | `vit_large_patch16_dinov3.lvd1689m`           | 512        | 20,480          |

Use a timm version with these architectures; this integration is exercised with timm 1.0.29.
CLIP retains QuickGELU and its learned 768-dimensional projection. Register tokens are not spatial patches.

## Usage

First pretrain a head, using the full labeled **auxiliary test split**:

```bash
uv run python -m tools.uniadet.train \
  --dataset visa --root datasets/visa/visa_pytorch --backbone dinov2 \
  --output results/UniADet/training/dinov2_visa
```

The tool uses Adam (`lr=0.001`, `betas=(0.5, 0.999)`), 15 epochs, batch size 16, blocks 12/15/18/21/24,
and class-aware grid mosaic/cropping with probability 0.5. Only the head is optimized. The resulting `head.pt`
contains head tensors and configuration; `training.json` records the run. Encoder weights are not bundled in it.

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import UniADet

model = UniADet(
    backbone="dinov2",
    weights="results/UniADet/training/dinov2_visa/head.pt",
    k_shot=4,
)
data = MVTecAD(root="datasets/MVTecAD", category="bottle")
engine = Engine()
engine.test(model=model, datamodule=data)
```

Use `k_shot=0` for zero-shot inference. Few-shot setup samples only normal training images, with an explicit seed.
Alternatively, supply `few_shot_source` containing only normal images. Use a separate model/reference bank per category.
Reference features are checkpoint buffers, so deployment does not require the original reference directory.

`Engine.fit` performs target calibration, **not auxiliary head training**. For prediction without a datamodule,
save a full checkpoint after calibration and pass it as `ckpt_path` to `Engine.predict`; the existing Engine otherwise
runs validation before predicting with a zero-/few-shot model.

## Reproduction and benchmarks

Train on VisA to evaluate MVTec AD; train on MVTec AD to evaluate VisA. Evaluating a head on the dataset used for its
auxiliary supervision is test-label leakage. The helper rejects that combination.

```bash
uv run python -m tools.uniadet.benchmark \
  --weights results/UniADet/training/dinov2_visa/head.pt \
  --dataset mvtec --root datasets/MVTecAD \
  --output results/UniADet/benchmark/dinov2_mvtec
```

The default sweep covers 0/1/2/4 shots and reference seeds 10/20/30 (zero-shot is evaluated once), with image/pixel
AUROC and average precision. It reuses frozen encoder outputs across shot settings, uses 256-pixel evaluation masks,
and records category-level measurements and selected references. Compute macro category means, not pooled-dataset AUROC.

```bash
uv run python -m tools.uniadet.report \
  results/UniADet/benchmark/dinov2_mvtec results/UniADet/benchmark/dinov2_visa \
  --output results/UniADet/comparison.csv
```

The report helper rejects partial category/seed coverage, smoke-trained heads, obsolete training recipes,
duplicated rows, and nonfinite scores.
It averages categories within each seed before reporting seed mean/population standard deviation, and computes
signed percentage-point gaps to the explicitly attributed paper Tables 2/5/10/11.

**This is not yet a verified reproduction of the authors' executable implementation.** The paper specifies fusion
weights of 0.5, whereas the released demo documentation gives different weights and source-side score recalibration.
This implementation follows the paper equations. It uses torchvision/timm rather than the authors' vendored encoders
and Albumentations pipeline. The corrected recipe uses temperature 0.07, Gaussian prototype standard deviation 0.02,
training seed 10, focal exponent 2 with smoothing 1e-5, and linear-denominator two-class Dice with smoothing 1.
These factual parameters were checked against the authors' published declarations; no upstream code was copied.
Training also enables color jitter (brightness/contrast/saturation 0.4, hue 0.1, probability 0.5),
right-angle rotation (0.5), channel shuffle/grayscale (0.25 each), and horizontal/vertical flips (0.5 each).
Backbone implementations, augmentation sampling and score processing can still differ; matching headline numbers
is not asserted.

Full training/evaluation and real-backbone export measurements are being collected under `results/UniADet/`.
All six auxiliary runs are being restarted from scratch with corrected **recipe version 2**.
Initial-recipe measurements are archived under `results/UniADet/initial_recipe/`, with explicit deviations in
[provenance.json](/results/UniADet/initial_recipe/provenance.json): prototype standard deviation 1 instead of 0.02,
seed 1015 instead of 10, a different Dice/interpolation convention, and missing color/geometric transforms.
Cosine normalization does not cancel the initialization scale's effect on Adam's angular updates.
Those initial measurements must not be used as corrected-recipe or paper-reproduction results.
No corrected-recipe benchmark numbers are claimed until the complete replacement sweeps finish.
Real-IAD data support and benchmarks are reserved for a separate follow-up PR.

## Deployment

Standard `to_torch`, `to_onnx`, `to_openvino`, and `Engine.export` entrypoints are used. Export at the model's fixed
spatial resolution with dynamic batch size; initialize the normal reference bank first for few-shot deployment.
ONNX export uses the dynamo exporter and opset 18. Full Torch exports contain encoder weights and are for local use,
subject to those weights' licenses.

```bash
uv run python -m tools.uniadet.validate \
  --weights results/UniADet/training/dinov2_visa/head.pt \
  --target mvtec --target-root datasets/MVTecAD --category bottle \
  --auxiliary-root datasets/visa/visa_pytorch --shots 4 \
  --output results/UniADet/deployment/dinov2_bottle_4
```

This checks TorchInferencer, Engine, ONNX Runtime, and OpenVINO FP32/FP16/INT8/INT8_PTQ/INT8_ACQ for batches 1 and 3.
It additionally checks a raw-resolution image path through TorchInferencer and both Engine predict/test APIs.
`parity.json` records maximum/mean absolute error and RMSE for scores and maps. Tolerances are fixed before comparison:
`1e-5` for Torch/Engine, `1e-4` for ONNX/FP32, `0.005` for FP16, and `0.05` for INT8 variants.
Quantization is approximate, not bit-identical; conversion success alone does not establish accuracy.
The same report records CPU latency with four threads, one warmup and three measured repetitions per batch size.
These shared-host timings include in-model preprocessing/postprocessing, but exclude image decoding and compilation.
The CPU checks request FP32 floating-point execution: FP16 denotes compressed stored weights, not a native FP16
CPU-kernel benchmark. INT8 denotes weight-only compression; PTQ/ACQ also quantize activations.

PTQ and ACQ use disjoint auxiliary-domain calibration and validation subsets (300 images each by default).
ACQ uses image AUROC with maximum drop 0.01; metric state is reset between candidates. Both paths use NNCF's
transformer-aware recipe, including SmoothQuant, rather than generic convolutional-network quantization.
Fixed map resizing and Gaussian smoothing remain floating point: calibrating these operations on the auxiliary domain
can clip unseen-domain anomaly maps. This is not a fully integer
graph. Final target-dataset accuracy still needs separate measurement.

The legacy TorchInferencer requires `TRUST_REMOTE_CODE=1` for locally generated full-model exports. Never enable this
for an untrusted downloaded pickle. The helper scopes this flag to loading its own newly generated export.

## Provenance and licenses

No source from `gaobb/UniADet` is copied or vendored. Its repository currently has no published source-code license;
an MIT grant must not be assumed. The independent implementation is covered by anomalib's Apache-2.0 license.
The paper and upstream repository are acknowledged as the algorithm's origin, not as licensed source dependencies.

The encoder implementations come from Apache-2.0 timm (minimum 1.0.20 for DINOv3 and the required feature API).
Pretrained weights have separate terms, especially DINOv3's
custom Meta license; see each publisher's model card before use or redistribution. The authors' gated
`csgaobb/UniADet` head checkpoints could not be accessed with the current credentials (HTTP 403) and are not bundled.
Source/head-weight licensing clarification remains a gate before an upstream PR.

MVTec AD is CC BY-NC-SA 4.0; VisA's dataset license is CC BY 4.0. Training on public datasets does not remove their
use restrictions. No raw dataset files or checkpoints are committed. Documentation panels are separately attributed
derived VisA assets.

## Sample results

Sample panels will be regenerated from completed corrected-recipe checkpoints and visually inspected before use.
Earlier panels came from the initial recipe; their provenance remains archived and they are not current-model examples.
