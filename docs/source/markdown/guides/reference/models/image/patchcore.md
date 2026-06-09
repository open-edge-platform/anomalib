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

| Backbone | Example layers |
|-----------|---------------|
| `resnet18` | `layer2 layer3` |
| `resnet50` | `layer2 layer3` |
| `wide_resnet50_2` | `layer2 layer3` |
| `efficientnet_b0` | `features.4 features.5` |
| `mobilenet_v3_large` | `features.6 features.12` |

The correct layer names depend on the model architecture. If a layer name is invalid, feature extraction will fail during training.

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
