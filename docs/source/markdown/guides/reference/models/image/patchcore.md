# PatchCore

## Architecture

```{eval-rst}
.. image:: ../../../../../images/patchcore/architecture.jpg
    :alt: PatchCore Architecture
```

## Using Different Backbones

PatchCore extracts intermediate feature maps from a backbone network. When changing the backbone, the selected `layers` must match valid layer names in that backbone.

Example CLI usage:

```bash
anomalib train \
   --model patchcore \
   --model.backbone wide_resnet50_2 \
   --model.layers layer2 layer3 \
   --model.pre_trained true
```

Common examples:

| Backbone                | Example layers          |
| ----------------------- | ----------------------- |
| `resnet18`              | `layer2 layer3`         |
| `resnet50`              | `layer2 layer3`         |
| `wide_resnet50_2`       | `layer2 layer3`         |
| `mobilenetv3_large_100` | `blocks.4.1 blocks.6.0` |

The correct layer names depend on the selected backbone architecture. PatchCore uses
`timm` feature extraction, so valid layer names should be checked with
`timm.create_model(<backbone>, features_only=True).feature_info.module_name()`.
If a layer name is invalid, feature extraction will fail during training.

```{eval-rst}
.. automodule:: anomalib.models.image.patchcore.lightning_model
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.patchcore.torch_model
   :members:
   :show-inheritance:
```
