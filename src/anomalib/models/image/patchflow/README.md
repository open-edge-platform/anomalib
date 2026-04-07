# PatchFlow: Leveraging a Flow-Based Model with Patch Features

This is the implementation of the [PatchFlow: Leveraging a Flow-Based Model with Patch Features](https://arxiv.org/abs/2602.05238) paper.

Model Type: Segmentation

## Description

PatchFlow is a normalizing-flow-based anomaly detection model that combines local neighbor-aware patch features with a normalizing flow model and bridge the gap between the generic pretrained feature extractor and industrial product images by introducing an adapter module to increase the efficiency and accuracy of automated anomaly detection.
The model extracts features at multiple resolutions from a frozen backbone (EfficientNet or DINOv2), fuses them via local average pooling and upsampling, adapts the channel dimension with a 1×1 convolution, and passes the result through a single normalizing flow for density estimation.

## Key Features

- a patch-level anomaly detection framework that combines local neighborhood-aware
  features with flow-based likelihood modeling.
- introduces a lightweight feature adaptation module that aligns pretrained representations with industrial image distributions, improving flow stability and detection accuracy.
- Efficient bottlenecked coupling structure for normalizing flows, reducing computational complexity while maintaining expressive capacity.

## Usage

Train with DINOv2 backbone:

```bash
anomalib train --config examples/configs/model/patchflow/dino.yaml --data MVTecAD --data.category bottle
```

Train with EfficientNet backbone:

```bash
anomalib train --config examples/configs/model/patchflow/efficientnet.yaml --data MVTecAD --data.category bottle
```

Or via the Python API:

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchflow
from anomalib.engine import Engine

datamodule = MVTecAD(category="bottle")
model = Patchflow(backbone="dinov2_vit_base_14")
engine = Engine()
engine.fit(model, datamodule=datamodule)
predictions = engine.predict(model, datamodule=datamodule)
```

## Reference

```bibtex
@article{Zhang2026PatchFlowLA,
  title={PatchFlow: Leveraging a Flow-Based Model with Patch Features},
  author={Boxiang Zhang and Baijian Yang and Xiaoming Wang and Corey Vian},
  journal={ArXiv},
  year={2026},
  volume={abs/2602.05238},
  url={https://api.semanticscholar.org/CorpusID:285303490}
}
```
