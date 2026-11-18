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
the unlicensed upstream source. All six full training runs and cross-dataset benchmarks are complete;
remaining activation-deployment measurements are being collected under `results/UniADet/`.
Replacement heads were trained from scratch with corrected recipe version 2: prototype standard deviation 0.02,
training seed 10, focal smoothing 1e-5, linear-denominator Dice, and the declared color/geometric augmentation parameters.
Initial-recipe measurements are archived under `results/UniADet/initial_recipe/`; `provenance.json` records their
incorrect initialization scale, seed, loss/interpolation convention, and missing color/geometric transforms.
They are not current-recipe results.
The report helper rejects obsolete recipes. Both corrected directions for all three encoders are complete:

| Encoder  | Target   | Shots | Image AUROC | Pixel AUROC | Image AP | Pixel AP |
| -------- | -------- | ----- | ----------- | ----------- | -------- | -------- |
| CLIP     | MVTec AD | 0     | 90.26       | 91.33       | 95.19    | 40.88    |
| CLIP     | MVTec AD | 4     | 94.25       | 95.44       | 97.21    | 51.50    |
| CLIP     | VisA     | 0     | 88.04       | 95.59       | 90.02    | 27.39    |
| CLIP     | VisA     | 4     | 91.70       | 97.18       | 93.51    | 31.99    |
| DINOv2-R | MVTec AD | 0     | 91.31       | 91.79       | 95.67    | 46.83    |
| DINOv2-R | MVTec AD | 4     | 94.73       | 97.09       | 97.70    | 55.93    |
| DINOv2-R | VisA     | 0     | 89.08       | 95.86       | 91.36    | 31.87    |
| DINOv2-R | VisA     | 4     | 94.41       | 97.81       | 95.94    | 37.12    |
| DINOv3   | MVTec AD | 0     | 91.75       | 91.62       | 95.79    | 49.85    |
| DINOv3   | MVTec AD | 4     | 94.28       | 97.30       | 97.54    | 59.47    |
| DINOv3   | VisA     | 0     | 87.80       | 96.34       | 90.12    | 32.65    |
| DINOv3   | VisA     | 4     | 93.45       | 97.79       | 94.75    | 37.32    |

Values are macro category means in percent. Some zero-shot VisA metrics match/exceed the paper;
CLIP image AUROC is 88.04 versus 88.0. Four-shot MVTec image AUROC remains 94.25/94.73 for CLIP/DINOv2-R
versus the paper's 97.7/98.7. The paper's overall headline accuracy is not reproduced.
All 96 default metric/shot/dataset/encoder comparisons and paper gaps are in `results/UniADet/comparison.csv`.
No missing results or partial category/seed coverage are presented as complete.
Paper and released-demo fusion settings differ.
The [model README](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uniadet)
documents the protocol, explicit implementation choices, commands, licenses, and validation tolerances.
Real-IAD is reserved for a separate follow-up PR.

Complete 15-epoch FP32 training uses one RTX 3090 per run, batch size 16, with one head covering all target categories:

| Encoder  | VisA training (min) | MVTec training (min) | Peak allocated GPU (GiB) |
| -------- | ------------------- | -------------------- | ------------------------ |
| CLIP     | 42.29               | 45.42                | 2.890                    |
| DINOv2-R | 40.95               | 46.34                | 2.810                    |
| DINOv3   | 31.86               | 31.76                | 2.575                    |

These are shared-host whole-run timings, not hardware-matched author comparisons or per-category training costs.
Peak memory is PyTorch allocated memory, not total device usage.
Exact measurements and provenance are in `results/UniADet/training_summary.csv` and source manifests.

FP32 RTX 3090 inference uses five warmups and 30 synchronized CUDA-event repetitions, without autocast, compilation
or TF32. Inputs are already resized; model preprocessing, scoring and map postprocessing are included.
Image decoding, initial CPU resizing and checkpoint loading are excluded.

| Encoder  | Shots | Batch 1 (ms) | Batch 3 total (ms) | Batch 3 images/s |
| -------- | ----- | ------------ | ------------------ | ---------------- |
| CLIP     | 0     | 81.97        | 218.16             | 13.75            |
| CLIP     | 4     | 85.33        | 227.96             | 13.16            |
| DINOv2-R | 0     | 78.04        | 209.10             | 14.35            |
| DINOv2-R | 4     | 82.83        | 223.25             | 13.44            |
| DINOv3   | 0     | 61.84        | 162.14             | 18.50            |
| DINOv3   | 4     | 64.67        | 169.18             | 17.73            |

Values are median batch latencies, not end-to-end serving times. Task-owned GPU training/evaluation had finished;
the host remains shared. Protocol, head hashes and memory measurements are in `results/UniADet/gpu_inference.json`.

The validation helper exercises file/tensor TorchInferencer, Engine predict/test, ONNX Runtime, and
OpenVINO FP32/FP16/INT8/PTQ/ACQ on batches 1 and 3. PTQ/ACQ use transformer-aware SmoothQuant;
fixed map resizing and smoothing stay floating point to avoid
auxiliary-domain range clipping. CPU checks request FP32 floating-point execution; FP16 refers to compressed weights,
and INT8 is weight-only compression rather than activation PTQ/ACQ.
Corrected CLIP, DINOv2-R and DINOv3 zero-/four-shot basic APIs and floating/weight-only exports pass;
activation PTQ/ACQ measurements remain in progress.
Real CUDA TorchInferencer file/tensor and checkpoint-restored Engine predict/test also pass all six cases at
batches 1/3. CPU/CUDA FP32 map differences stay below `3.61e-5` with a predeclared `1e-4` cross-device tolerance.
Full measurements are in `results/UniADet/gpu_inference_api.json`.
Conversion-time CPU timings are contended; controlled CPU measurements follow the task-owned activation jobs.
ACQ ranks continuous task outputs rather than undefined single-image AUROCs while constraining aggregate image AUROC.
All three zero-shot PTQ/ACQ conversions are complete. Native ACQ preserves auxiliary image AUROC within 0.01
without reverting quantizers, so PTQ and ACQ have identical measured numerical errors.
Maximum map errors are 0.10547/0.06151/0.09730 for CLIP/DINOv2-R/DINOv3, exceeding the unchanged 0.05 tolerance.
Auxiliary AUROC preservation does not imply per-pixel amplitude equivalence; scores pass their numerical checks.
The numerical sample is MVTec bottle at batches 1/3, not an all-category quantized benchmark.
Native metrics are in `results/UniADet/activation_accuracy_control.json`.
Cached-IR replay found no batch-size effect; a narrow head-only exclusion did not resolve it and was not adopted.
Zero-shot activation quantization is not certified equivalent; four-shot cases are still being measured.
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
