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
Both DINOv2-R cross-dataset directions are complete, covering all 15 MVTec AD and 12 VisA categories and default seeds:

| Target   | Shots | Image AUROC | Pixel AUROC | Image AP | Pixel AP |
| -------- | ----- | ----------- | ----------- | -------- | -------- |
| MVTec AD | 0     | 89.70       | 91.09       | 94.44    | 42.24    |
| MVTec AD | 4     | 96.05       | 96.62       | 97.96    | 54.71    |
| VisA     | 0     | 87.61       | 95.35       | 90.67    | 28.60    |
| VisA     | 4     | 95.16       | 97.56       | 96.40    | 36.33    |

Values are macro category means in percent. The corresponding paper image/pixel AUROC is 93.5/92.4 and 98.7/97.7.
For VisA, the corresponding paper pairs are 91.3/95.9 and 96.9/98.6.
These measurements remain below the paper; no published-paper accuracy is claimed as reproduced.
CLIP and DINOv3 are still running. All shot counts, seed variability and attributed paper gaps
are recorded in `results/UniADet/comparison.csv`. Paper and released-demo fusion settings differ.
The [model README](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uniadet)
documents the protocol, explicit implementation choices, commands, licenses, and validation tolerances.
Real-IAD is reserved for a separate follow-up PR.

The validation helper exercises TorchInferencer, Engine, ONNX Runtime, and OpenVINO FP32/FP16/INT8/PTQ/ACQ on batches
1 and 3. PTQ/ACQ use transformer-aware SmoothQuant; quantized models retain the fixed smoothing filter in floating point.
Authors' gated checkpoints are unavailable to the current account, and licensing clarification is
required before opening an upstream PR.

## Sample results

Two VisA examples use the completed MVTec-trained DINOv2-R head, four normal references and seed 10.
Panels show input, ground truth, and a raw anomaly-map overlay on a fixed 0-to-1 scale.
These are qualitative examples, not a paper-reproduction claim. Other candidates with distracting false positives
were excluded. Provenance is recorded in `results/UniADet/sample_images.json`.

```{eval-rst}
.. image:: ../../../../../images/uniadet/results/0.png
    :alt: VisA chewing-gum anomaly

.. image:: ../../../../../images/uniadet/results/1.png
    :alt: VisA PCB missing-pin anomaly
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
