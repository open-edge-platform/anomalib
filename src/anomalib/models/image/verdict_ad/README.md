# VerdictAD

VerdictAD uses a frozen typed decision model to refine an existing anomaly
detector's image scores and anomaly maps. This implementation combines Anomalib's
AnomalyDINO, PatchCore, or Dinomaly detector with global (GVR) and local (LVR) refinement from
[VerdictAD: Few-Shot Anomaly Detection via Decision-Guided Refinement](https://arxiv.org/abs/2611.02654).

The default is AnomalyDINO + VerdictAD. This uses Anomalib's AnomalyDINO feature
extraction, nearest-neighbor scoring and top-1% aggregation, rather than the
paper's separate intermediate-layer, max-pooled DINOv2-L baseline. Select
`detector="patchcore"` or `detector="dinomaly"` to select another detector.

## How it works

The first `n_shots` normal training images are saved as RGB decision references.
AnomalyDINO and PatchCore fit their memory banks from exactly those images.
Dinomaly trains its decoder and bottleneck on all normal training batches while
its encoder stays frozen. Training needs one category and at least `n_shots`
normal images; choose 1, 2 or 4 shots to match the paper's evaluation settings.
Seed the training loader to make reference selection reproducible.

With both refinement stages enabled, Clef-Flash produces 34 binary defect decisions per query:

- Two global views: the query alone, and the query with normal references.
- Sixteen overlapping crops, each compared with corresponding reference crops.
- Sixteen red-box views that retain the full image's context.

Reference grids contain four cells, repeating references cyclically for fewer
than four shots. Global views are 896 x 896; local views are 448 x 448.
Windows have normalized side length 0.4 and stride 0.2.

GVR averages the two global logits, then multiplies the raw image score by
`sigmoid(mean_logit) ** alpha`. LVR averages `sigmoid(logit) ** gamma` over all
windows covering a pixel and multiplies the raw anomaly map by that weight.
Default exponents are `alpha=0.5` and `gamma=0.25`.
Calibration, thresholding and metrics run **after** refinement.

LVR can suppress existing responses but cannot create evidence where the base
map is zero. The released prompts ask generic defect questions: this does not
implement explicit scratch-size tolerances or guarantee that an acceptable stain
will be ignored. Evaluate those requirements on representative accepted and
rejected examples.

## Setup and usage

Install the optional decision-model dependencies with `uv`:

```bash
uv pip install -e ".[huggingface]" "transformers>=5.14" "huggingface_hub>=1.24" "torch>=2.11" accelerate
```

Download a pinned revision of
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) separately,
including its Python release code, processor files and weights. For example,
replace `COMMIT_SHA` with the revision you have reviewed:

```bash
hf download Cloudflare/clef-flash --revision COMMIT_SHA --local-dir models/clef-flash
```

`decision_model_path` must point to that **trusted local release directory**.
Its `joint_schema_model.py` is executed on first refinement. Model construction
does not load Clef-Flash or import Transformers; DINO weights follow the existing
AnomalyDINO download behavior. The decision model runs locally on the detector's
device. No query images are sent to a hosted inference service.

```python
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import VerdictAD

seed_everything(42, workers=True)
model = VerdictAD(
    n_shots=4,
    decision_model_path="models/clef-flash",
    decision_batch_size=4,
)
data = MVTecAD(category="bottle", train_batch_size=4, eval_batch_size=1)
engine = Engine()
engine.fit(model=model, datamodule=data)
engine.test(model=model, datamodule=data)
```

To use PatchCore with the same reference-selection and refinement pipeline:

```python
model = VerdictAD(
    detector="patchcore",
    n_shots=4,
    decision_model_path="models/clef-flash",
    patchcore_backbone="wide_resnet50_2",
    patchcore_layers=("layer2", "layer3"),
    patchcore_coreset_sampling_ratio=0.1,
    patchcore_num_neighbors=9,
)
engine.fit(model=model, datamodule=data)
engine.test(model=model, datamodule=data)
```

The CLI exposes the same choice with `--model.detector patchcore`. Both detectors
fit only the first `n_shots` normal training images and use those images for the
decision references. The common preprocessor defaults to 896 × 896, including for
PatchCore. Supply a resize-only preprocessor to use another input size.

### Dinomaly

```python
model = VerdictAD(
    detector="dinomaly",
    dinomaly_encoder_name="vit_base_patch14_reg4_dinov2",
    n_shots=4,
    decision_model_path="models/clef-flash",
)
engine = Engine(max_steps=5000)
engine.fit(model=model, datamodule=data)
engine.test(model=model, datamodule=data)
```

Dinomaly uses its reconstruction loss, trainable bottleneck and decoder, and
StableAdamW optimizer with a warmup/cosine schedule advanced per training step.
Set the training duration through the Engine, for example `Engine(max_steps=5000)`.
Its RGB preprocessor resizes to
448 × 448 and center-crops to 392 × 392, matching the standalone model's geometry.
ImageNet normalization is applied inside the detector. Refinement runs during
validation, test, and prediction, before threshold calibration.

### Custom decision questions

Pass a partial dictionary to override any of the four question types:

```python
model = VerdictAD(
    detector="dinomaly",
    decision_model_path="models/clef-flash",
    decision_prompts={
        "free": "Does this object have an unacceptable defect? Ignore harmless colour variation.",
        "reference": "Image 1 is the query; image 2 contains normal references. Is the query defective?",
        "crop": "Does the crop in image 1 have an unacceptable defect compared with the reference crops in image 2?",
        "box": "Is there an unacceptable defect inside the red box in image 1 compared with image 2's references?",
    },
)
```

Omitted keys retain the released questions. Keys must be `free`, `reference`,
`crop`, or `box`, and values must be nonempty strings. Every question must use
**true = unacceptable defect**. Keep the image roles and local-region descriptions
consistent with the views above. The decision model's weights stay frozen.
Prompt overrides are saved in Lightning checkpoints. Validate custom prompts on
accepted and rejected examples and recalibrate thresholds after changing them.
Prompt wording alone does not establish physical defect-size tolerances.

The model is also discoverable as `get_model("VerdictAD")` and through the CLI:

```bash
anomalib train --model VerdictAD --model.n_shots 4 --model.decision_model_path models/clef-flash --data MVTecAD --data.category bottle
```

AnomalyDINO and PatchCore preprocessing resizes RGB images to 896 x 896 without normalization.
ImageNet normalization is applied only inside the base detector. Custom
preprocessing must retain RGB floating-point values in `[0, 1]` and use consistent
geometry for references and queries. Do not use externally normalized tensors.

Both the reference images and detector memory bank persist in checkpoints.
Clef-Flash code and weights remain external; keep the same pinned release
available after restoring, or explicitly override `decision_model_path` when
calling `VerdictAD.load_from_checkpoint`.

## Refining another detector

The tensor-only `refine_predictions` function is independent of AnomalyDINO:

```python
from anomalib.models.image.verdict_ad.refinement import refine_predictions

# Supply the raw, nonnegative outputs of your detector and aligned decision logits.
refined = refine_predictions(raw_predictions, global_logits, window_logits, boxes)
```

It accepts image scores `(B,)`, maps `(B, H, W)` or `(B, 1, H, W)`, two global
logits per image, and one local logit per normalized XYXY window. Apply it before
calibration. Use `ClefFlash.predict` and `BOXES * 2` from the `decider` module to
obtain the released view scheme. The query images must match the maps' spatial
coordinates, and references must match those used by the base detector.

## Validation and limitations

A one-shot CPU evaluation used all 83 MVTec AD bottle test images (20 normal,
63 defective). AnomalyDINO used pretrained `vit_small_patch14_dinov2` at 224 × 224
and the single normal reference `bottle/train/good/037.png`. Both runs used the
same raw detector outputs. Refinement used the default exponents and the authors'
[recorded Clef-Flash decisions for seed 42](https://huggingface.co/datasets/sunsean/VerdictAD/blob/4c00ac858a914cd5bdf6c3afc6a1164d21b89437/decisions/mvtec_k1_s42/bottle.json).
Pixel AUROC pools all pixels using ground-truth masks resized with nearest-neighbor interpolation.

| Model                   | Image AUROC | Pixel AUROC |
| ----------------------- | ----------: | ----------: |
| AnomalyDINO             |      0.9857 |      0.9861 |
| AnomalyDINO + VerdictAD |      0.9952 |      0.9866 |

This comparison covers one category and reference. It measures the effect of
applying recorded decisions. Live Clef-Flash inference and production acceptance
criteria require separate evaluation.

Offline tests cover the equations, overlap averaging, decision views and option
ordering, reference selection, model registration, and checkpoint restoration.
They replace pretrained feature extraction and decision inference; they do not
establish detection accuracy or reproduce the paper's results.

For an ablation, use identical references and set `alpha=0`, `gamma=0` to recover
the detector outputs without loading Clef-Flash. Setting only `gamma=0` skips all
32 local decisions. Setting only `alpha=0` skips both global decisions.
Changing refinement requires recalibrating thresholds. The upstream implementation reports decision batches
of four fitting on a 24 GB GPU; the combined detector also consumes memory.
This 9B decision model is intended for accuracy-oriented inspection and adds
substantial inference cost. GPU memory and runtime have not been measured here.

The complete PIL/custom-code decision pipeline supports eager Python inference.
Torch, ONNX and OpenVINO export are unsupported; restore Lightning checkpoints
with the local decision-model release instead. Use a single device, as configured
by the model.

## Attribution

The refinement equations, prompts and view construction are adapted from
[the official implementation](https://github.com/sgstq0811/VerdictAD-Jev-AD),
released under Apache-2.0. See the repository's `LICENSE` and
`third-party-programs.txt`. Clef-Flash weights and datasets retain their own licenses.
