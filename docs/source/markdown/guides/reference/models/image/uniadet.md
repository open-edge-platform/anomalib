# UniADet

UniADet is a language-free zero-/few-shot detector with a frozen vision transformer and independently learned
classification/segmentation prototypes. Auxiliary training uses labeled images from a different dataset;
target training images, when used, are normal few-shot references only.

## Architecture

```{eval-rst}
.. image:: ../../../../../images/uniadet/architecture.svg
    :alt: Independently drawn UniADet architecture
```

Backbones are OpenAI CLIP ViT-L/14 (QuickGELU), DINOv2-R ViT-L/14, and DINOv3 ViT-L/16 through timm.
Auxiliary training uses `python -m tools.uniadet.train`, not `Engine.fit`. Engine's target learning type remains
zero-shot/few-shot. A fixed per-category reference buffer supports deployment with dynamic batch sizes.

## Reproduction status

This is an independent implementation of [the published algorithm](https://arxiv.org/abs/2601.05552), not a port of
the unlicensed upstream source. Full benchmark and deployment results are being collected under `results/UniADet/`.
Replacement runs train from scratch with corrected recipe version 2: prototype standard deviation 0.02,
training seed 10, focal smoothing 1e-5, linear-denominator Dice, and the declared color/geometric augmentation parameters.
Initial-recipe measurements are archived under `results/UniADet/initial_recipe/`; `provenance.json` records their
incorrect initialization scale, seed, loss/interpolation convention, and missing color/geometric transforms.
They are not current-recipe results.
The report helper rejects obsolete recipes. Both corrected DINOv2-R directions are complete:

| Target   | Shots | Image AUROC | Pixel AUROC | Image AP | Pixel AP |
| -------- | ----- | ----------- | ----------- | -------- | -------- |
| MVTec AD | 0     | 91.31       | 91.79       | 95.67    | 46.83    |
| MVTec AD | 4     | 94.73       | 97.09       | 97.70    | 55.93    |
| VisA     | 0     | 89.08       | 95.86       | 91.36    | 31.87    |
| VisA     | 4     | 94.41       | 97.81       | 95.94    | 37.12    |

Values are macro category means in percent. Paper MVTec image/pixel AUROC is 93.5/92.4 and 98.7/97.7;
for VisA it is 91.3/95.9 and 96.9/98.6. Paper-level accuracy is not claimed as reproduced.
The complete default shot/seed coverage and paper gaps are in `results/UniADet/comparison.csv`.
Other corrected sweeps are still running; no missing results are inferred.
Paper and released-demo fusion settings differ.
The [model README](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uniadet)
documents the protocol, explicit implementation choices, commands, licenses, and validation tolerances.
Real-IAD is reserved for a separate follow-up PR.

The validation helper exercises file/tensor TorchInferencer, Engine predict/test, ONNX Runtime, and
OpenVINO FP32/FP16/INT8/PTQ/ACQ on batches 1 and 3. PTQ/ACQ use transformer-aware SmoothQuant;
fixed map resizing and smoothing stay floating point to avoid
auxiliary-domain range clipping. CPU checks request FP32 floating-point execution; FP16 refers to compressed weights,
and INT8 is weight-only compression rather than activation PTQ/ACQ.
Corrected DINOv2-R zero-/four-shot basic APIs and floating/weight-only exports pass; other encoders
and activation PTQ/ACQ measurements remain in progress.
Authors' gated checkpoints are unavailable to the current account, and licensing clarification is
required before opening an upstream PR. The encoder adapter requires timm 1.0.20 or newer.

## Sample results

Three VisA examples use the completed recipe-v2 MVTec-trained DINOv2-R head and four normal references (seed 10).
Panels show input, ground truth, and raw anomaly-map overlays on a fixed 0-to-1 scale.
Six candidates were visually inspected; three with distracting normal-region activations were excluded.
These are qualitative examples, not evidence of paper-level accuracy. Provenance is in `results/UniADet/sample_images.json`.

```{eval-rst}
.. image:: ../../../../../images/uniadet/results/0.png
    :alt: VisA chewing-gum anomaly

.. image:: ../../../../../images/uniadet/results/1.png
    :alt: VisA PCB missing-pin anomaly

.. image:: ../../../../../images/uniadet/results/2.png
    :alt: VisA capsule anomaly
```

Photographs/masks: VisA, Zou et al. (2022), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
resized and overlaid with model outputs. The colocated asset `LICENSE` records attribution and changes.

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
