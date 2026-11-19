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

### Author-checkpoint evaluation

Use the explicit `protocol="author"` with authorized heads imported by `python -m tools.uniadet.author`.
This preserves legacy defaults and uses raw DINO intermediate features, published CLIP positional interpolation,
rounded DINOv3 rotary periods, and executable scoring/256-square peak pooling. Head files are protocol-validated.
The author's ten prototype tensors are stacked without changing values; no upstream source is copied.
This is **author-head evaluation through anomalib**, not independently reproduced auxiliary training.

`python -m tools.uniadet.reproduce` evaluates both datasets/all backbones using original public splits,
VisA `1cls.csv` row order, category-local with-replacement reference seeds 10/20/30, and 0/1/2/4 shots
(zero-shot once). It uses original RGB images, OpenCV uint8 bilinear resizing, nearest binary masks,
256-square pixel evaluation and batch 32. Paper AP follows the released evaluator's **recall-sorted
trapezoidal PR area**, not standard stepwise average precision. Both definitions are retained separately in CSVs;
anomalib's standard metrics are unchanged.
The primary measurements use FP32; separate `--encoder-precision author` CLIP runs quantify its CUDA mixed precision.
Reference manifests, split identities and checkpoint/software/code provenance are recorded.
Generic Engine datamodule loading alone is not claimed to replay the helper's exact uint8 resize/VisA reference order.
All six full evaluations and both CLIP precision supplements are complete: **810 FP32 plus 270 supplemental
category/shot/seed rows**. All **96 image/pixel AUROC/paper-AP comparisons are within 0.07 percentage points**
of their published means, for both uniform FP32 and the prespecified author-loader precision policy.
The largest gap is 0.06956 points (four-shot VisA DINOv3 image AUROC, 97.03044 versus 97.1).
CLIP's precision variant changes any aggregate metric by less than 0.01 points.

Selected complete results use the **author-loader precision**: CLIP mixed FP16 with FP32 LayerNorm/head,
and FP32 DINO encoders. Values are category/seed macro means in percent, with paper-convention AP:

| Encoder  | Target   | Shots | Image AUROC | Pixel AUROC | Image AP | Pixel AP |
| -------- | -------- | ----- | ----------- | ----------- | -------- | -------- |
| CLIP     | MVTec AD | 0     | 92.40       | 91.83       | 96.08    | 42.81    |
| CLIP     | MVTec AD | 4     | 97.75       | 96.30       | 98.58    | 58.85    |
| CLIP     | VisA     | 0     | 87.96       | 95.76       | 89.11    | 27.98    |
| CLIP     | VisA     | 4     | 93.26       | 97.96       | 93.78    | 36.67    |
| DINOv2-R | MVTec AD | 0     | 93.50       | 92.44       | 97.23    | 50.85    |
| DINOv2-R | MVTec AD | 4     | 98.73       | 97.65       | 99.22    | 65.44    |
| DINOv2-R | VisA     | 0     | 91.27       | 95.87       | 90.99    | 32.73    |
| DINOv2-R | VisA     | 4     | 96.92       | 98.58       | 96.80    | 45.25    |
| DINOv3   | MVTec AD | 0     | 93.98       | 93.20       | 97.14    | 52.69    |
| DINOv3   | MVTec AD | 4     | 98.20       | 97.90       | 98.80    | 68.99    |
| DINOv3   | VisA     | 0     | 91.94       | 96.34       | 92.80    | 32.54    |
| DINOv3   | VisA     | 4     | 97.03       | 98.38       | 96.57    | 45.47    |

All 0/1/2/4-shot means, seed standard deviations, paper references and gaps are in
`results/UniADet/reproduction/paper_precision_comparison.csv`. The uniform FP32 baseline is in `comparison.csv`;
CLIP precision differences are in `clip_precision_deltas.csv`.
`verification.json` checks exact head conversion, all reference selections, split identities and final code hashes.
`protocol.json` records the released environment differences and dataset-specific details.
This is close **author-head evaluation**, not independent training reproduction or bit-exact replay of the unpublished
paper environment. See the model README for exact import/evaluation commands.

The changed author graph passes all **114 floating-export/API batch checks** for three encoders at zero/four shots:
CPU file/tensor TorchInferencer, CPU/CUDA Engine predict/test and CUDA TorchInferencer, ONNX, and OV FP32/FP16,
using MVTec bottle/VisA-trained heads at batches 1/3. Maximum map errors are `1.40e-5`/`3.36e-5`/`0.00264`
for ONNX/OV FP32/FP16, and `9.48e-6` between CPU/CUDA APIs.
Evidence is in `results/UniADet/reproduction/deployment_summary.csv` and per-case parity reports.
Quantization is not re-certified for the author graph; activation PTQ/ACQ remains deferred.

### Independently trained legacy recipe

This is an independent implementation of [the published algorithm](https://arxiv.org/abs/2601.05552), not a port of
the unlicensed upstream source. All six full training runs, cross-dataset benchmarks and export conversions are complete
under `results/UniADet/`. Activation formats fail the strict maximum-map criterion described below.
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
Legacy AP is standard average precision, so its historical AP gaps are not a metric-identical paper comparison.
No missing results or partial category/seed coverage are presented as complete.
Paper and released-demo fusion settings differ.
The FP32 component audit has now measured the effect of this difference: with unchanged recipe-v2 heads,
features and seed-10 references, the published executable's scoring raises four-shot MVTec image AUROC from
94.35/94.89/94.34 to **97.64/98.62/98.24** for CLIP/DINOv2-R/DINOv3 across all 15 categories.
The paper reports 97.7/98.7/98.2. These are single-reference-seed diagnostic image metrics, not replacement
three-seed benchmarks or new pixel results; the original tables above remain unchanged.
All 12 VisA categories are also checked: four-shot values change 91.71/94.55/93.95 to 92.80/96.62/95.41,
versus paper 93.3/96.9/97.1. Zero-shot VisA DINOv3 improves 87.80 to 92.10, but CLIP worsens 88.04 to 87.10.
The scorer is therefore not a universal fix, and residual gaps remain. Both directions are recorded in
`results/UniADet/diagnosis/scoring_comparison.csv`.
Reference fusion weight 0.8, sigmoid recalibration, smoothing and resize/peak-extraction ordering account for
this controlled change. The zero-shot executable also uses image-map weight 0.1.
The audit additionally confirms DINO intermediate final-LayerNorm, dataset-specific CAA and reference-RNG
differences from the author executable, plus training-map interpolation alignment.
A paired 15-epoch, three-auxiliary-category/four-target-category experiment measures the LayerNorm effect:
baseline zero-/four-shot image AUROC changes 89.58/94.83 to 91.26/96.45, but with executable scoring
94.54/98.68 changes to 94.22/98.73. It is not a uniformly beneficial change or full-dataset retraining.
The remaining training/reference differences have not all had their accuracy contributions isolated.
On the four-category probe, with-replacement sampling changes executable-scoring four-shot image AUROC
98.52 to 98.07; this does not establish parity with unavailable original author reference manifests.
Block selection, token exclusion, CLIP projection and macro aggregation match.
The published executable is not proof of the unpublished paper-run configuration.
Use `python -m tools.uniadet.diagnose` for fixed-protocol scoring ablations; measured artifacts and author questions
are under `results/UniADet/diagnosis/`. Legacy defaults and PTQ/ACQ policy remain unchanged.
The separate explicit author protocol implements the verified inference differences.
The [model README](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uniadet)
documents the protocol, explicit implementation choices, commands, licenses, and validation tolerances.
Real-IAD is reserved for a separate follow-up PR.

The following training, speed and quantization measurements use legacy recipe-v2 heads/scoring, not the author preset.
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

Controlled CPU timings use an Intel Core i9-10920X, four threads pinned to four physical cores, two warmups and five
repetitions, after task-owned conversion/training jobs finished. The host remains shared.
OpenVINO requests FP32 execution, including compressed FP16 weights. Median batch-1 latency is in **seconds**:

| Encoder  | Shots | Torch FP32 | OV FP32 | OV FP16 | OV INT8 | OV PTQ | OV ACQ |
| -------- | ----- | ---------- | ------- | ------- | ------- | ------ | ------ |
| CLIP     | 0     | 2.307      | 1.908   | 1.666   | 1.303   | 0.775  | 0.781  |
| CLIP     | 4     | 2.057      | 1.980   | 1.746   | 1.361   | 0.804  | 0.805  |
| DINOv2-R | 0     | 1.904      | 1.890   | 1.653   | 1.288   | 0.767  | 0.767  |
| DINOv2-R | 4     | 2.012      | 2.032   | 1.759   | 1.377   | 0.809  | 0.808  |
| DINOv3   | 0     | 1.405      | 1.376   | 1.233   | 0.968   | 0.579  | 0.578  |
| DINOv3   | 4     | 1.451      | 1.491   | 1.283   | 0.986   | 0.599  | 0.599  |

INT8 is weight-only. PTQ/ACQ speeds are measured but **not numerically certified** because their maps fail the
declared tolerance. This is not real-time CPU inference on this hardware.
All 72 timing entries, including batch-3 latency/throughput and repeated numerical checks, are in
`results/UniADet/cpu_inference.json`. Decoding, initial resizing, loading and compilation are excluded.

The validation helper exercises file/tensor TorchInferencer, Engine predict/test, ONNX Runtime, and
OpenVINO FP32/FP16/INT8/PTQ/ACQ on batches 1 and 3. PTQ/ACQ use transformer-aware SmoothQuant;
fixed map resizing and smoothing stay floating point to avoid
auxiliary-domain range clipping. CPU checks request FP32 floating-point execution; FP16 refers to compressed weights,
and INT8 is weight-only compression rather than activation PTQ/ACQ.
Corrected CLIP, DINOv2-R and DINOv3 zero-/four-shot basic APIs and floating/weight-only exports pass.
All activation PTQ/ACQ conversions run successfully, with strict numerical limitations below.
Real CUDA TorchInferencer file/tensor and checkpoint-restored Engine predict/test also pass all six cases at
batches 1/3. CPU/CUDA FP32 map differences stay below `3.61e-5` with a predeclared `1e-4` cross-device tolerance.
Full measurements are in `results/UniADet/gpu_inference_api.json`.
Use the controlled CPU measurements above rather than contended conversion-time timings.
ACQ ranks continuous task outputs rather than undefined single-image AUROCs while constraining aggregate image AUROC.
All six zero-/four-shot PTQ/ACQ cases are complete. Native ACQ preserves auxiliary image AUROC within 0.01
without reverting quantizers, so PTQ and ACQ have identical measured numerical errors.
Zero-shot maximum map errors are 0.10547/0.06151/0.09730 for CLIP/DINOv2-R/DINOv3;
four-shot errors are 0.06266/0.08991/0.06512. All exceed the unchanged 0.05 tolerance.
Auxiliary AUROC preservation does not imply per-pixel amplitude equivalence; scores pass their numerical checks.
The numerical sample is MVTec bottle at batches 1/3, not an all-category quantized benchmark.
Native metrics are in `results/UniADet/activation_accuracy_control.json`.
Cached-IR DINOv2-R zero-shot replay found no batch-size effect; a narrow head-only exclusion did not resolve it.
Four-shot reference-query and RGB-input probes also fail the fixed map tolerance; neither is adopted.
No broad precision exclusions or relaxed tolerances are used.
The complete `results/UniADet/deployment_summary.csv` contains 114 batch/format measurements, with 18 map failures
confined to activation PTQ/ACQ. All score checks pass. Use FP32, compressed FP16 or weight-only INT8 for the
checked numerical consistency; activation variants are not certified equivalent.
Authorized access to all six author heads is now verified; none are redistributed.
Their research/evaluation model-card terms are separate from source licensing, which still needs
clarification before an upstream PR. The encoder adapter requires timm 1.0.20 or newer.

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
