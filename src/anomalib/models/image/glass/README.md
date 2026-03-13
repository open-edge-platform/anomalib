# GLASS: A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization

This is the implementation of the [GLASS](https://arxiv.org/pdf/2407.09359) paper.

Model Type: Segmentation

## Description

GLASS is a unified framework that synthesizes both global and local anomalies for unsupervised anomaly detection. It addresses the limitations in coverage and controllability of existing anomaly synthesis strategies, particularly for weak defects that closely resemble normal regions.

The model consists of three branches during training:

- **Normal branch**: Extracts adapted normal features through a frozen feature extractor and trainable feature adaptor.
- **Global Anomaly Synthesis (GAS) branch**: Synthesizes near-in-distribution anomalies at the feature level using Gaussian noise guided by gradient ascent with truncated projection under manifold or hypersphere distribution constraints.
- **Local Anomaly Synthesis (LAS) branch**: Synthesizes far-from-distribution anomalies at the image level by overlaying augmented textures onto images using Perlin noise masks.

A shared discriminator is trained on features from all three branches. During inference, only the normal branch processes test images and the discriminator produces anomaly scores.

## Usage

`anomalib train --model Glass --data MVTecAD --data.category <category>`

**Note:** GLASS requires the [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) texture dataset for local anomaly synthesis. It will be downloaded automatically if not present.

**Note:** GLASS uses different distribution hypotheses per category (`svd=0` for manifold, `svd=1` for hypersphere). The default config uses `svd=0`. For optimal per-category results, set `svd` according to the table below:

| svd=0 (manifold)                                                           | svd=1 (hypersphere)                      |
| -------------------------------------------------------------------------- | ---------------------------------------- |
| Carpet, Grid, Leather, Tile, Wood, Capsule, Hazelnut, Metal Nut, Toothbrush, Zipper | Bottle, Cable, Pill, Screw, Transistor |

## Benchmarking

To reproduce per-category results on MVTec AD with the correct SVD settings, use the dedicated benchmark script:

```bash
python tools/experimental/benchmarking/benchmark_glass.py \
    --dataset-root ~/datasets/MVTecAD \
    --anomaly-source ~/datasets/dtd/dtd/images
```

## Benchmark

All results gathered with seed `0`, 100 epochs, best checkpoint selected by `combined_AUROC` (image + pixel) per category.

> [!NOTE]
> To get numbers similar to the paper, image size of 288x288 was used with per-category SVD settings.
> Best checkpoint was selected by `combined_AUROC` (image + pixel) per category.
> Training used batch size 8, learning rate 1e-4, 100 epochs, seed 0, and a sample limit of 392 per epoch.
> Paper reports I-AUROC 99.9% and P-AUROC 99.3% on MVTec AD.
> The gap is likely due to minor differences in augmentation pipeline (Perlin noise padding, DTD loading).

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| Wide ResNet-50 | 0.972 | 0.998  | 0.995 |  1.000  | 1.000 | 0.993 | 1.000  | 0.928 |  0.967  |  1.000   |   0.999   | 0.953 | 0.909 |   0.950    |   0.897    | 0.995  |

### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| Wide ResNet-50 | 0.965 | 0.992  | 0.981 |  0.999  | 0.992 | 0.978 | 0.973  | 0.938 |  0.981  |  0.968   |   0.846   | 0.974 | 0.971 |   0.976    |   0.911    | 0.988  |
