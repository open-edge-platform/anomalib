<div align="center">

<img src="https://raw.githubusercontent.com/open-edge-platform/anomalib/main/docs/source/_static/images/logos/anomalib-wide-blue.png" width="600px" alt="Anomalib Logo - A deep learning library for anomaly detection">

**A library for benchmarking, developing and deploying deep learning anomaly detection algorithms**

---

[Key Features](#key-features) •
[Docs](https://anomalib.readthedocs.io/en/latest/) •
[Notebooks](examples/notebooks) •
[License](LICENSE)

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]()
[![lightning](https://img.shields.io/badge/lightning-2.2%2B-blue)]()
[![openvino](https://img.shields.io/badge/openvino-2024.0%2B-purple)]()

[![Pre-Merge Checks](https://github.com/open-edge-platform/anomalib/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/open-edge-platform/anomalib/actions/workflows/pre_merge.yml)
[![codecov](https://codecov.io/gh/open-edge-platform/anomalib/branch/main/graph/badge.svg?token=Z6A07N1BZK)](https://codecov.io/gh/open-edge-platform/anomalib)
[![Downloads](https://static.pepy.tech/personalized-badge/anomalib?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/anomalib)
[![snyk](https://snyk.io/advisor/python/anomalib/badge.svg)](https://snyk.io/advisor/python/anomalib)

[![ReadTheDocs](https://readthedocs.org/projects/anomalib/badge/?version=latest)](https://anomalib.readthedocs.io/en/latest/?badge=latest)
[![Anomalib - Gurubase docs](https://img.shields.io/badge/Gurubase-Ask%20Anomalib%20Guru-006BFF)](https://gurubase.io/g/anomalib)

<a href="https://trendshift.io/repositories/6030" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6030" alt="open-edge-platform%2Fanomalib | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

---

> 🌟 **Announcing v2.0.0 Release!** 🌟
>
> We're excited to announce the release of Anomalib v2.0.0! This version introduces significant improvements and customization options to enhance your anomaly detection workflows. Please be aware that there are several API changes between `v1.2.0` and `v2.0.0`, so please be careful when updating your existing pipelines. Key features include:
>
> - Multi-GPU support
> - New [dataclasses](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/dataclasses.html) for model in- and outputs.
> - Flexible configuration of [model transforms and data augmentations](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/transforms.html).
> - Configurable modules for pre- and post-processing operations via [`Preprocessor`](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/models/pre_processor.html) and [`Postprocessor`](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/models/post_processor.html)
> - Customizable model evaluation workflow with new [Metrics API](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/evaluation/metrics.html) and [`Evaluator`](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/evaluation/evaluator.html) module.
> - Configurable module for visualization via `Visualizer` (docs guide: coming soon)
>
> We value your input! Please share feedback via [GitHub Issues](https://github.com/open-edge-platform/anomalib/issues) or our [Discussions](https://github.com/open-edge-platform/anomalib/discussions)

# 👋 Introduction

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on visual anomaly detection, where the goal of the algorithm is to detect and/or localize anomalies within images or videos in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

<p align="center">
  <img src="https://raw.githubusercontent.com/open-edge-platform/anomalib/main/docs/source/_static/images/readme.png" width="1000" alt="A prediction made by anomalib">
</p>

## Key features

- Simple and modular API and CLI for training, inference, benchmarking, and hyperparameter optimization.
- The largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.
- [**Lightning**](https://www.lightning.ai/) based model implementations to reduce boilerplate code and limit the implementation efforts to the bare essentials.
- The majority of models can be exported to [**OpenVINO**](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) Intermediate Representation (IR) for accelerated inference on Intel hardware.
- A set of [inference tools](tools) for quick and easy deployment of the standard or custom anomaly detection models.

# 📦 Installation

Anomalib can be installed from PyPI. We recommend using a virtual environment and a modern package installer like `uv` or `pip`.

## 🚀 Quick Install

For a standard installation, you can use `uv` or `pip`. This will install the latest version of Anomalib with its core dependencies. PyTorch will be installed based on its default behavior, which usually works for CPU and standard CUDA setups.

```bash
# With uv
uv pip install anomalib

# Or with pip
pip install anomalib
```

For more control over the installation, such as specifying the PyTorch backend (e.g., XPU, CUDA and ROCm) or installing extra dependencies for specific models, see the advanced options below.

<details>
<summary><strong>💡 Advanced Installation: Specify Hardware Backend</strong></summary>

To ensure compatibility with your hardware, you can specify a backend during installation. This is the recommended approach for production environments and for hardware other than CPU or standard CUDA.

**Using `uv`:**

```bash
# CPU support (default, works on all platforms)
uv pip install "anomalib[cpu]"

# CUDA 12.4 support (Linux/Windows with NVIDIA GPU)
uv pip install "anomalib[cu124]"

# CUDA 12.1 support (Linux/Windows with NVIDIA GPU)
uv pip install "anomalib[cu121]"

# CUDA 11.8 support (Linux/Windows with NVIDIA GPU)
uv pip install "anomalib[cu118]"

# ROCm support (Linux with AMD GPU)
uv pip install "anomalib[rocm]"

# Intel XPU support (Linux with Intel GPU)
uv pip install "anomalib[xpu]"
```

**Using `pip`:**
The same extras can be used with `pip`:

```bash
pip install "anomalib[cu124]"
```

</details>

<details>
<summary><strong>🧩 Advanced Installation: Additional Dependencies</strong></summary>

Anomalib includes most dependencies by default. For specialized features, you may need additional optional dependencies. Remember to include your hardware-specific extra.

```bash
# Example: Install with OpenVINO support and CUDA 12.4
uv pip install "anomalib[openvino,cu124]"

# Example: Install all optional dependencies for a CPU-only setup
uv pip install "anomalib[full,cpu]"
```

Here is a list of available optional dependency groups:

| Extra         | Description                              | Purpose                                     |
| :------------ | :--------------------------------------- | :------------------------------------------ |
| `[openvino]`  | Intel OpenVINO optimization              | For accelerated inference on Intel hardware |
| `[clip]`      | Vision-language models                   | `winclip`                                   |
| `[vlm]`       | Vision-language model backends           | Advanced VLM features                       |
| `[loggers]`   | Experiment tracking (wandb, comet, etc.) | For experiment management                   |
| `[notebooks]` | Jupyter notebook support                 | For running example notebooks               |
| `[full]`      | All optional dependencies                | All optional features                       |

</details>

<details>
<summary><strong>🔧 Advanced Installation: Install from Source</strong></summary>

For contributing to `anomalib` or using a development version, you can install from source.

**Using `uv`:**
This is the recommended method for developers as it uses the project's lock file for reproducible environments.

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Create the virtual environment
uv venv

# Sync with the lockfile for a specific backend (e.g., CPU)
uv sync --extra cpu

# Or for a different backend like CUDA 12.4
uv sync --extra cu124

# To set up a full development environment
uv sync --extra dev --extra cpu
```

**Using `pip`:**

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Install in editable mode with a specific backend
pip install -e ".[cpu]"

# Install with development dependencies
pip install -e ".[dev,cpu]"
```

</details>

# 🧠 Training

Anomalib supports both API and CLI-based training approaches:

## 🔌 Python API

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize components
datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
```

## ⌨️ Command Line

```bash
# Train with default settings
anomalib train --model Patchcore --data anomalib.data.MVTecAD

# Train with custom category
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category transistor

# Train with config file
anomalib train --config path/to/config.yaml
```

# 🤖 Inference

Anomalib provides multiple inference options including Torch, Lightning, Gradio, and OpenVINO. Here's how to get started:

## 🔌 Python API

```python
# Load model and make predictions
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)
```

## ⌨️ Command Line

```bash
# Basic prediction
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTecAD \
                 --ckpt_path path/to/model.ckpt

# Prediction with results
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTecAD \
                 --ckpt_path path/to/model.ckpt \
                 --return_predictions
```

> 📘 **Note:** For advanced inference options including Gradio and OpenVINO, check our [Inference Documentation](https://anomalib.readthedocs.io).

# Training on Intel GPUs

> [!Note]
> Currently, only single GPU training is supported on Intel GPUs.
> These commands were tested on Arc 750 and Arc 770.

Ensure that you have PyTorch with XPU support installed. For more information, please refer to the [PyTorch XPU documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)

## 🔌 API

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Stfpm

engine = Engine(
    strategy=SingleXPUStrategy(),
    accelerator=XPUAccelerator(),
)
engine.train(Stfpm(), datamodule=MVTecAD())
```

## ⌨️ CLI

```bash
anomalib train --model Padim --data MVTecAD --trainer.accelerator xpu --trainer.strategy xpu_single
```

# ⚙️ Hyperparameter Optimization

Anomalib supports hyperparameter optimization (HPO) using [Weights & Biases](https://wandb.ai/) and [Comet.ml](https://www.comet.com/).

```bash
# Run HPO with Weights & Biases
anomalib hpo --backend WANDB --sweep_config tools/hpo/configs/wandb.yaml
```

> 📘 **Note:** For detailed HPO configuration, check our [HPO Documentation](https://open-edge-platform.github.io/anomalib/tutorials/hyperparameter_optimization.html).

# 🧪 Experiment Management

Track your experiments with popular logging platforms through [PyTorch Lightning loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html):

- 📊 Weights & Biases
- 📈 Comet.ml
- 📉 TensorBoard

Enable logging in your config file to track:

- Hyperparameters
- Metrics
- Model graphs
- Test predictions

> 📘 **Note:** For logging setup, see our [Logging Documentation](https://open-edge-platform.github.io/anomalib/tutorials/logging.html).

# 📊 Benchmarking

Evaluate and compare model performance across different datasets:

```bash
# Run benchmarking with default configuration
anomalib benchmark --config tools/experimental/benchmarking/sample.yaml
```

> 💡 **Tip:** Check individual model performance in their respective README files:
>
> - [Patchcore Results](src/anomalib/models/image/patchcore/README.md#mvtec-ad-dataset)
> - [Other Models](src/anomalib/models/)

# ✍️ Reference

If you find Anomalib useful in your research or work, please cite:

```tex
@inproceedings{akcay2022anomalib,
  title={Anomalib: A deep learning library for anomaly detection},
  author={Akcay, Samet and Ameln, Dick and Vaidya, Ashwin and Lakshmanan, Barath and Ahuja, Nilesh and Genc, Utku},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1706--1710},
  year={2022},
  organization={IEEE}
}
```

# 👥 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

<p align="center">
  <a href="https://github.com/open-edge-platform/anomalib/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=open-edge-platform/anomalib" alt="Contributors to open-edge-platform/anomalib" />
  </a>
</p>

<p align="center">
  <b>Thank you to all our contributors!</b>
</p>
