# UniNet: Unified Contrastive Learning Framework for Anomaly Detection

This is the implementation of the UniNet model, a unified contrastive learning framework for anomaly detection presented in CVPR 2025.

Model Type: Classification and Segmentation

## Description

UniNet is a contrastive learning-based anomaly detection model that uses a teacher-student architecture with attention bottleneck mechanisms. The model is designed for diverse domains and supports both supervised and unsupervised anomaly detection scenarios. It focuses on multi-class anomaly detection and leverages domain-related feature selection to improve performance across different categories.

The model consists of:

- **Teacher Networks**: Pre-trained backbone networks that provide reference features for normal samples
- **Student Network**: A decoder network that learns to reconstruct normal patterns
- **Attention Bottleneck**: Mechanisms that help focus on relevant features
- **Domain-Related Feature Selection**: Adaptive feature selection for different domains
- **Contrastive Loss**: Temperature-controlled similarity computation between student and teacher features

During training, the student network learns to match teacher features for normal samples while being trained to distinguish anomalous patterns through contrastive learning. The model uses a weighted decision mechanism during inference to combine multi-scale features for final anomaly scoring.

## Architecture

The UniNet architecture leverages contrastive learning with teacher-student networks, incorporating attention bottlenecks and domain-specific feature selection for robust anomaly detection across diverse domains.

## Usage

`anomalib train --model UniNet --data MVTecAD --data.category <category>`

## Benchmark

All results gathered with seed `42`.

## [MVTecAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | Hazelnut | Leather | Metal Nut | Pill  | Screw | Tile  | Toothbrush | Transistor | Wood  | Zipper |
| :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| 0.956 | 0.999  | 0.982 |  0.939  | 0.896  | 0.996 |  0.999   |  1.000  |   1.000   | 0.816 | 0.919 | 0.970 |   1.000    |   0.984    | 0.993 | 0.945  |

### Pixel-Level AUC

|  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | Hazelnut | Leather | Metal Nut | Pill  | Screw | Tile  | Toothbrush | Transistor | Wood  | Zipper |
| :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| 0.976 | 0.989  | 0.983 |  0.985  | 0.973  | 0.992 |  0.987   |  0.993  |   0.984   | 0.964 | 0.992 | 0.965 |   0.992    |   0.923    | 0.961 | 0.984  |

### Image F1 Score

|  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | Hazelnut | Leather | Metal Nut | Pill  | Screw | Tile  | Toothbrush | Transistor | Wood  | Zipper |
| :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| 0.957 | 0.984  | 0.944 |  0.964  | 0.883  | 0.973 |  0.986   |  0.995  |   0.989   | 0.921 | 0.905 | 0.946 |   0.983    |   0.961    | 0.974 | 0.959  |

## Model Features

- **Contrastive Learning**: Uses temperature-controlled contrastive loss for effective anomaly detection
- **Multi-Domain Support**: Designed to work across diverse domains with domain-related feature selection
- **Flexible Training**: Supports both supervised and unsupervised training modes
- **Attention Mechanisms**: Incorporates attention bottlenecks for focused feature learning
- **Multi-Scale Features**: Leverages multi-scale feature matching for robust detection

## Parameters

- `student_backbone`: Backbone model for student network (default: "wide_resnet50_2")
- `teacher_backbone`: Backbone model for teacher network (default: "wide_resnet50_2")
- `temperature`: Temperature parameter for contrastive loss (default: 0.1)
