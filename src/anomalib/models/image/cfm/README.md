# CFM: Crossmodal Feature Mapping

This is the implementation of the [Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping](https://arxiv.org/abs/2312.04521) paper (CVPR 2024). Based on <https://github.com/CVLAB-Unibo/crossmodal-feature-mapping>.

Model Type: Segmentation

## Description

CFM is a lightweight multimodal anomaly detection method that leverages both RGB images and 3D point clouds. Unlike memory-bank-based approaches (M3DM, 3D-ADS), CFM trains two small MLP networks to map features between modalities: one predicts 3D features from RGB, and the other predicts RGB features from 3D. Training uses only nominal (defect-free) samples, so the mappings learn cross-modal consistency on normal data.

During inference, anomalies are detected as inconsistencies between observed features and their cross-modal predictions. The anomaly map is computed as the element-wise product of normalized L2 distances in both directions, amplifying regions where both modalities disagree.

### Feature Extraction

- **RGB**: DINO ViT-Base (patch size 8, 224×224 input) extracts 768-dim patch features.
- **3D**: Point-MAE pretrained PointTransformer groups the point cloud into local neighborhoods and produces 1152-dim features (concatenation of layers 3, 7, 11).

Both backbones are frozen; only the two mapping MLPs are trained.

### Anomaly Detection

For each spatial location, the model computes:

1. Normalized L2 distance between predicted RGB features and observed RGB features.
2. Normalized L2 distance between predicted 3D features and observed 3D features.
3. Element-wise product of the two distance maps (amplifies co-occurring anomalies).
4. Gaussian blur smoothing and top-k scoring for the image-level anomaly score.

## Architecture

For the full architecture diagram, see the [original paper](https://arxiv.org/abs/2312.04521).

## Usage

CFM requires a multimodal dataset with RGB and depth/point-cloud data (e.g., MVTec 3D-AD). Train one category at a time:

```bash
anomalib train --model anomalib.models.CFM --data anomalib.data.MVTec3D --data.category bagel --data.train_batch_size 1 --trainer.devices 1
```

Or with a config file:

```bash
anomalib train -c examples/configs/model/cfm.yaml --data anomalib.data.MVTec3D --data.category bagel --data.train_batch_size 1
```

Point-cloud operations are memory-intensive. If you hit OOM, keep batch size at 1, use a single device, and reduce `num_group`.

## Benchmark

All results gathered with seed `42` on [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

> **Note**: Benchmarks pending full evaluation. The table below will be populated after a complete training run across all categories.

### MVTec 3D-AD Dataset

| Metric      | Avg |
| ----------- | :-: |
| Image AUROC |  -  |
| Pixel AUROC |  -  |
| Image F1    |  -  |
