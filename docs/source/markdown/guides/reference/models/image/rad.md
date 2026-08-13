# RAD

## Description

RAD (Retrieval-based Anomaly Detection) is a training-free framework that stores
anomaly-free features in a multi-layer memory bank and detects anomalies through
multi-level retrieval with position-aware patch matching.

The model uses a DINOv3 ViT-B/16 backbone to extract features from multiple transformer
blocks. At inference, it retrieves the most similar training images via CLS token
similarity, then performs patch-level anomaly scoring using cosine distance with
optional spatial position constraints.

## Usage

```bash
anomalib train \
   --model rad \
   --model.backbone vit_base_patch16_dinov3 \
   --model.layers "[3, 6, 9, 11]" \
   --model.k_image 150 \
   --model.use_positional_bank true \
   --model.pos_radius 1
```

## Key Parameters

| Parameter             | Default                   | Description                                                |
| --------------------- | ------------------------- | ---------------------------------------------------------- |
| `backbone`            | `vit_base_patch16_dinov3` | ViT backbone from timm                                     |
| `layers`              | `[3, 6, 9, 11]`           | Transformer block indices (0-based) for feature extraction |
| `k_image`             | `150`                     | Number of nearest training images for local memory         |
| `use_positional_bank` | `True`                    | Enable position-aware patch matching                       |
| `pos_radius`          | `1`                       | Spatial neighborhood radius for positional matching        |
| `max_ratio`           | `0.01`                    | Fraction of top anomaly pixels for image-level score       |

```{eval-rst}
.. automodule:: anomalib.models.image.rad.lightning_model
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.rad.torch_model
   :members:
   :show-inheritance:
```
