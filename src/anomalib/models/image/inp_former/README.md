# INP-Former

This is an implementation of the INP-Former, based on the [official code](https://github.com/luow23/INP-Former).

The model was first presented at CVPR 2025: [Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection](https://arxiv.org/pdf/2503.02424)

Model Type: Segmentation

## Description

The INP-Former model implements an encoder-decoder architecture for anomaly detection
that extracts Intrinsic Normal Prototypes (INPs) directly from each test image, rather
than relying on prototypes stored from the training set. Features from a frozen
pre-trained Vision Transformer encoder are aggregated by an INP Extractor (cross-attention
with learnable query tokens) into a small set of INPs per image, fused through a
bottleneck, and reconstructed by an INP-Guided Decoder that uses the INPs as keys and
values to constrain its output to normal patterns.

## Usage

`anomalib train --model InpFormer --data MVTecAD --data.category <category>`
