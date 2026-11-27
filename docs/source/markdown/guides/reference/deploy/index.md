# Inference

ONNX and OpenVINO export use math scaled-dot-product attention during graph capture to avoid device-specific fused
attention layout failures. Training and PyTorch inference retain their existing attention backends.

Re-export DFM and FRE models if an older ONNX or OpenVINO export rejects single-image inputs with a batch-size mismatch.

```{eval-rst}
.. automodule:: anomalib.deploy
   :members:
   :exclude-members: Inferencer
   :show-inheritance:
```
