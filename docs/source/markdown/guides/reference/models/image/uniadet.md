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
All six runs are restarting from scratch with corrected recipe version 2: prototype standard deviation 0.02,
training seed 10, focal smoothing 1e-5, linear-denominator Dice, and the declared color/geometric augmentation parameters.
Initial-recipe measurements are archived under `results/UniADet/initial_recipe/`; `provenance.json` records their
incorrect initialization scale, seed, loss/interpolation convention, and missing color/geometric transforms.
They are not current-recipe results.
The report helper rejects obsolete recipes. No corrected-recipe benchmark numbers are claimed yet.
Paper and released-demo fusion settings differ.
The [model README](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uniadet)
documents the protocol, explicit implementation choices, commands, licenses, and validation tolerances.
Real-IAD is reserved for a separate follow-up PR.

The validation helper exercises file/tensor TorchInferencer, Engine predict/test, ONNX Runtime, and
OpenVINO FP32/FP16/INT8/PTQ/ACQ on batches 1 and 3. PTQ/ACQ use transformer-aware SmoothQuant;
fixed map resizing and smoothing stay floating point to avoid
auxiliary-domain range clipping. CPU checks request FP32 floating-point execution; FP16 refers to compressed weights,
and INT8 is weight-only compression rather than activation PTQ/ACQ.
Authors' gated checkpoints are unavailable to the current account, and licensing clarification is
required before opening an upstream PR.

## Sample results

Sample panels will be regenerated from completed corrected-recipe checkpoints and visually inspected.
Earlier initial-recipe panels are not presented as current-model examples; their provenance remains archived.

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
