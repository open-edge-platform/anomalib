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

The L2BT method introduces a Teacher-Student paradigm to leverage strong pre-trained features while processing high-resolution input images very efficiently.
The core idea concerns training two shallow MLPs (the Students) on nominal images so as to mimic the mappings between the patch embeddings induced by the self-attention layers of a frozen Vision Transformer (the Teacher). Indeed, learning these mappings sets forth a challenging pretext task that small-capacity models are unlikely to accomplish on out-of-distribution data such as anomalous images.

The method can spot anomalies from high-resolution images and runs significantly faster than competitors, achieving state-of-the-art performance on MVTec AD and the best segmentation results on VisA. Novel evaluation metrics are also proposed to capture robustness to defect size, i.e., the ability to preserve good localisation from large anomalies to tiny ones. Evaluating the method with these metrics further highlights its superior performance.

<h4 align="center">

</h4>

<img src="./assets/architecture.jpg" alt="Alt text" style="width: 800px;" title="architecture">

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

Pretrained weights used in the original paper are available at the following link:

[Download checkpoints](https://drive.google.com/drive/folders/1cdcfW8cV_iURK_OwWkKjJFpzJvnJMbeb?usp=sharing)

Within the **anomalib** framework, checkpoints are automatically managed during training and stored in the experiment output directory.

---

## Usage

This implementation integrates **L2BT** into the **anomalib** framework.

Training and inference are performed using the standard anomalib command-line interface.

### :hammer_and_wrench: Setup Instructions

Ensure that the required dependencies for **anomalib** are installed.

Refer to the anomalib installation instructions for environment setup.

### :rocket: Inference L2BT

Inference can be performed using the standard **anomalib** command-line interface.

Example command:

```bash
anomalib predict \
  --config src/anomalib/models/image/l2bt/config.yaml \
  --ckpt_path <path_to_checkpoint>
```

The checkpoint is generated automatically after training and can be found in the results directory created by anomalib.

This command generates anomaly scores and anomaly maps for the selected dataset.

### :rocket: Train L2BT

Training can be performed using the anomalib training interface.

Example command:

```bash
anomalib train \
  --config src/anomalib/models/image/l2bt/config.yaml
```

During training, anomalib automatically manages:

- dataset loading
- experiment logging
- checkpoint saving
- evaluation metrics

Model parameters can be configured in:

```text
src/anomalib/models/image/l2bt/config.yaml
```

## :envelope: Contacts

For questions regarding the original method, please contact:

alex.costanzino@unibo.it
