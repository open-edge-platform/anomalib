# SuperADD: Training-free Class-agnostic Anomaly Segmentation -- CVPR 2026 VAND 4.0 Workshop Challenge Industrial Track

This is the implementation of the [SuperADD](https://arxiv.org/abs/2605.14808) paper.

Model Type: Segmentation

## Description

SuperADD extracts multi-layer Vision Transformer token features from a pretrained DINOv3 backbone over overlapping image patches, builds a per-layer memory bank from normal training images using distance-based coreset subsampling, and detects anomalies by nearest-neighbor search against this memory bank.

## Architecture

![SuperADD Architecture](/docs/source/images/super_add/architecture.png "SuperADD Architecture")

## Usage

`anomalib train --model SuperAdd --data MVTecAD2 --data.category <category>`

The original papers authors use a very high input resolution to the model. To replicate their input stratetgy, scale the MVTecAD2 images by a factor of 0.625 and keep the aspect ratio of your images for better accuracy. Set the models paramters to `patch_size=640` and `patch_overlap=128`. These settings result in a high memory consumption.

Also the papers authors use brightness augmentation of the interval [0.8, 1.2]. This can be achieved by using [Data augmentations](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/transforms.html) in anomalib and adding `ColorJitter` with the respective brightness interval.

Note on thresholding: MVTec AD 2 provides only anomaly-free validation images, so anomalib's default F1-adaptive threshold degenerates to the maximum validation score. This maximum grows with the input resolution, which collapses F1 scores and flattens the visualized anomaly maps for multi-patch configurations even when the underlying maps are good (AUROC is unaffected). SuperADD therefore ships with `SuperADDPostProcessor`, which follows the original implementation and calibrates the threshold as the 95th percentile of the validation anomaly scores scaled by 1.421 — a resolution-independent statistic computed from normal images only. The percentile and factor are configurable via `SuperADD(post_processor=SuperADDPostProcessor(...))`.

Please be aware that the original paper uses additional post-processing steps to achieve the reported results. These steps include:

- Morphological closing using multiple thresholds and filling closed segmented areas
- Downscaling feature maps for evalutation by a factor of 4

### Sample Results

![Sample Result 1](/docs/source/images/super_add/result_images/0.png "Sample Result 1")

![Sample Result 2](/docs/source/images/super_add/result_images/1.png "Sample Result 2")
