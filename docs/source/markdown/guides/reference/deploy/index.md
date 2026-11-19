# Inference

ONNX and OpenVINO export use math scaled-dot-product attention during graph capture to avoid device-specific fused
attention layout failures. Training and PyTorch inference retain their existing attention backends.

```{eval-rst}
.. automodule:: anomalib.deploy
   :members:
   :exclude-members: Inferencer
   :show-inheritance:
```
