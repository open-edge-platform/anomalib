<h1 align="center"> Learning to Be a Transformer to Pinpoint Anomalies (IEEE Access) </h1>

<br>

:rotating_light: This page describes the method **"Learning to Be a Transformer to Pinpoint Anomalies"** published in [IEEE Access](https://ieeeaccess.ieee.org/).

The original work is by
[Alex Costanzino\*](https://alex-costanzino.github.io/), [Pierluigi Zama Ramirez\*](https://pierlui92.github.io/), [Giuseppe Lisanti](https://www.unibo.it/sitoweb/giuseppe.lisanti), and [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano). \* _Equal Contribution_

University of Bologna

<div class="alert alert-info">

<h2 align="center">

[Project Page](https://alex-costanzino.github.io/learning_to_be_a_transformer/) | [Paper](https://ieeexplore.ieee.org/document/11048772)

</h2>

## :bookmark_tabs: Table of Contents

1. [Introduction](#clapper-introduction)
2. [Datasets](#file_cabinet)
3. [Checkpoints](#inbox_tray)
4. [Usage](#usage)
5. [Contacts](#envelope-contacts)

</div>

## :clapper: Introduction

To efficiently deploy strong, often pre-trained feature extractors, recent Industrial Anomaly Detection and Segmentation (IADS) methods process low-resolution images, e.g., 224x224 pixels, obtained by downsampling the original input images. However, while numerous industrial applications demand the identification of both large and small defects, downsampling the input image to a low resolution may hinder a method's ability to pinpoint tiny anomalies.

The L2BT method introduces a Teacher-Student paradigm to leverage strong pre-trained features while processing high-resolution input images very efficiently. The core idea concerns training two shallow MLPs (the Students) on nominal images so as to mimic the mappings between the patch embeddings induced by the self-attention layers of a frozen Vision Transformer (the Teacher). Indeed, learning these mappings sets forth a challenging pretext task that small-capacity models are unlikely to accomplish on out-of-distribution data such as anomalous images.

The method can spot anomalies from high-resolution images and runs significantly faster than competitors, achieving state-of-the-art performance on MVTec AD and the best segmentation results on VisA. Novel evaluation metrics are also proposed to capture robustness to defect size, i.e., the ability to preserve good localisation from large anomalies to tiny ones. Evaluating the method with these metrics further highlights its superior performance.

<h4 align="center"></h4>

![architecture](../../../../../docs/source/images/l2bt/architecture.jpg)

:fountain_pen: If you find this work useful in your research, please cite:

```bibtex
@article{costanzino2025learning2be,
  author    = {Costanzino, Alex and Zama Ramirez, Pierluigi and Lisanti, Giuseppe and Di Stefano, Luigi},
  title     = {Learning to Be a Transformer to Pinpoint Anomalies},
  journal   = {IEEE Access},
  year      = {2025},
}
```

<h2 id="file_cabinet"> :file_cabinet: Datasets </h2>

The original paper evaluates L2BT on the following datasets:

- [VisA](https://github.com/amazon-science/spot-diff)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

Within **anomalib**, the model can be used with datasets supported by the framework.

<h2 id="inbox_tray"> :inbox_tray: Checkpoints </h2>

Within anomalib, checkpoints are automatically managed during training and stored in the experiment output directory. No external checkpoint management is required.

---

## Usage

This implementation integrates **L2BT** into the **anomalib** framework.

Training and inference are performed using the standard anomalib command-line interface.

### :hammer_and_wrench: Setup Instructions

Ensure that the required dependencies for **anomalib** are installed.

Refer to the anomalib installation instructions for environment setup.

### :rocket: Train L2BT

```bash
anomalib train \
  --model L2BT \
  --data Visa \
  --data.category capsules
```

> **Note:** If you encounter dataloader or shared-memory issues on your machine, you can reduce workers and batch size:
>
> ```bash
> anomalib train \
>   --model L2BT \
>   --data Visa \
>   --data.category capsules \
>   --data.num_workers 0 \
>   --data.train_batch_size 1 \
>   --data.eval_batch_size 1 \
>   --trainer.max_epochs 1
> ```

During training, anomalib automatically manages:

- dataset loading
- experiment logging
- checkpoint saving
- evaluation metrics

### :rocket: Inference L2BT

```bash
anomalib predict \
  --model L2BT \
  --data <PATH_TO_IMAGE_OR_FOLDER> \
  --ckpt_path <PATH_TO_CHECKPOINT>
```

Example:

```bash
anomalib predict \
  --model L2BT \
  --data datasets/visa/visa_pytorch/capsules/test \
  --ckpt_path results/L2BT/Visa/capsules/latest/weights/lightning/model.ckpt
```

The checkpoint is generated automatically after training and can be found in the results directory created by anomalib.

This command generates anomaly scores and anomaly maps for the selected dataset.

### Model Configuration

Model parameters can be configured using:

```text
examples/configs/model/l2bt.yaml
```

## Benchmark

All results gathered with seed `42` and `max_epochs=50`.

> **Note:** The original paper uses high-resolution images (1036×1036) with `lr=0.001`, whereas anomalib
> defaults to 224×224 with `lr=1e-4`. These differences explain the gap between reproduced and
> paper-reported metrics (paper: MVTec AD I-AUROC 0.988, VisA I-AUROC 0.964).

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|      |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| ---- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| L2BT | 0.977 | 1.000  | 0.958 |  0.934  | 1.000  | 0.999 |  1.000   |  1.000  |   1.000   | 0.969 | 0.864 | 1.000 |   0.969    |   0.974    | 0.990 | 0.996  |

### Image F1 Score

|      |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| ---- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| L2BT | 0.963 | 0.992  | 0.921 |  0.942  | 0.994  | 0.983 |  0.993   |  0.995  |   0.995   | 0.975 | 0.887 | 0.994 |   0.951    |   0.871    | 0.966 | 0.983  |

### Pixel-Level AUC

|      |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| ---- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| L2BT | 0.973 | 0.987  | 0.955 |  0.983  | 0.991  | 0.986 |  0.994   |  0.988  |   0.974   | 0.975 | 0.971 | 0.948 |   0.988    |   0.945    | 0.933 | 0.973  |

### Pixel F1 Score

|      |  Avg  | Bottle | Cable | Capsule | Carpet | Grid  | HazelNut | Leather | Metal Nut | Pill  | Screw | Tile  | ToothBrush | Transistor | Wood  | Zipper |
| ---- | :---: | :----: | :---: | :-----: | :----: | :---: | :------: | :-----: | :-------: | :---: | :---: | :---: | :--------: | :--------: | :---: | :----: |
| L2BT | 0.562 | 0.754  | 0.606 |  0.479  | 0.590  | 0.362 |  0.724   |  0.352  |   0.807   | 0.617 | 0.340 | 0.597 |   0.551    |   0.559    | 0.587 | 0.514  |

## [VisA Dataset](https://github.com/amazon-science/spot-diff)

### Image-Level AUC

|      |  Avg  | Candle | Capsules | Cashew | Chewinggum | Fryum | Macaroni1 | Macaroni2 | PCB1  | PCB2  | PCB3  | PCB4  | Pipe Fryum |
| ---- | :---: | :----: | :------: | :----: | :--------: | :---: | :-------: | :-------: | :---: | :---: | :---: | :---: | :--------: |
| L2BT | 0.911 | 0.932  |  0.891   | 0.955  |   0.987    | 0.946 |   0.890   |   0.772   | 0.917 | 0.863 | 0.825 | 0.959 |   0.994    |

### Image F1 Score

|      |  Avg  | Candle | Capsules | Cashew | Chewinggum | Fryum | Macaroni1 | Macaroni2 | PCB1  | PCB2  | PCB3  | PCB4  | Pipe Fryum |
| ---- | :---: | :----: | :------: | :----: | :--------: | :---: | :-------: | :-------: | :---: | :---: | :---: | :---: | :--------: |
| L2BT | 0.859 | 0.863  |  0.844   | 0.939  |   0.959    | 0.911 |   0.806   |   0.705   | 0.848 | 0.809 | 0.741 | 0.902 |   0.985    |

### Pixel-Level AUC

|      |  Avg  | Candle | Capsules | Cashew | Chewinggum | Fryum | Macaroni1 | Macaroni2 | PCB1  | PCB2  | PCB3  | PCB4  | Pipe Fryum |
| ---- | :---: | :----: | :------: | :----: | :--------: | :---: | :-------: | :-------: | :---: | :---: | :---: | :---: | :--------: |
| L2BT | 0.977 | 0.991  |  0.973   | 0.995  |   0.991    | 0.969 |   0.930   |   0.964   | 0.996 | 0.970 | 0.970 | 0.980 |   0.991    |

### Pixel F1 Score

|      |  Avg  | Candle | Capsules | Cashew | Chewinggum | Fryum | Macaroni1 | Macaroni2 | PCB1  | PCB2  | PCB3  | PCB4  | Pipe Fryum |
| ---- | :---: | :----: | :------: | :----: | :--------: | :---: | :-------: | :-------: | :---: | :---: | :---: | :---: | :--------: |
| L2BT | 0.386 | 0.308  |  0.335   | 0.648  |   0.573    | 0.456 |   0.175   |   0.116   | 0.628 | 0.197 | 0.281 | 0.363 |   0.545    |

## :envelope: Contacts

For questions regarding the original method, please contact:

alex.costanzino@unibo.it
