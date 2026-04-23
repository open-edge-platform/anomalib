# AnomalyVFM

This is an implementation of the AnomalyVFM, based on the [official code](https://github.com/MaticFuc/AnomalyVFM).

The model was first presented at CVPR 2026: [AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](https://arxiv.org/abs/2601.20524)

Model Type: Segmentation

## Description

AnomalyVFM implements a zero-shot anomaly detector on top of pretrained VFMs.
It does so by first generating synthetic images using FLUX and training on top of them.
The model directly predicts an anomaly score and a anomaly mask.

This implementation supports the zero-shot setting.

## Architecture

![AnomalyVFM architecture](/docs/source/images/anomalyvfm/architecture.png "AnomalyVFM architecture")

Currently, the difference between ICPR and JIMS code is only the `adapt_cls_features` which controls whether the features used for classification head are adapted or not.
For ICPR this is set to True (i.e. the features for classification head are adapted), and for JIMS version this is False (which is also the default).

## Usage

`anomalib test --model AnomalyVFM --data MVTecAD --data.category <category>`

## MVTecAD AD results

The following results were obtained using this Anomalib implementation

| category   |   I-AUROC |   I-F1Max |   P-AUROC |   P-F1Max |     AUPRO |
| :--------- | --------: | --------: | --------: | --------: | --------: |
| bottle     |     98.37 |     97.64 |     94.92 |     69.85 |     90.11 |
| cable      |     91.92 |     88.14 |     87.54 |     18.80 |     56.55 |
| capsule    |     96.47 |     95.24 |     97.86 |     42.31 |     93.57 |
| carpet     |     99.84 |     98.89 |     99.65 |     75.47 |     98.69 |
| grid       |     99.92 |     99.13 |     97.42 |     43.16 |     90.91 |
| hazelnut   |     99.14 |     97.14 |     96.70 |     55.71 |     92.93 |
| leather    |     99.97 |     99.46 |     99.56 |     57.72 |     99.01 |
| metal_nut  |     98.19 |     97.30 |     66.28 |     28.84 |     83.53 |
| pill       |     95.40 |     95.27 |     87.63 |     38.55 |     91.75 |
| screw      |     97.94 |     95.55 |     99.36 |     47.91 |     96.55 |
| tile       |     99.42 |     98.82 |     96.24 |     74.48 |     93.44 |
| toothbrush |     91.81 |     93.10 |     92.65 |     36.26 |     91.68 |
| transistor |     78.87 |     72.22 |     68.75 |     15.45 |     54.03 |
| wood       |    100.00 |    100.00 |     96.23 |     70.38 |     78.82 |
| zipper     |     98.84 |     97.14 |     98.36 |     62.57 |     94.36 |
| average    | **96.41** | **95.00** | **91.94** | **49.16** | **87.06** |

For other results on other datasets (VisA, Real-IAD, MPDD, BTAD, KSDD, KSDD2, DAGM, DTD and more), refer to the [paper](https://arxiv.org/abs/2601.20524) or the [official code](https://github.com/MaticFuc/AnomalyVFM).
