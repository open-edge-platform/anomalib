# L2BT

L2BT (Learning to Be a Transformer to Pinpoint Anomalies) is an image anomaly detection model based on a teacher–student learning framework.

The method leverages a strong pretrained Vision Transformer as a frozen feature extractor and trains lightweight student networks to reproduce its internal feature mappings. Differences between teacher and student representations are used to detect anomalous regions in images.

This implementation integrates the L2BT method into the **anomalib** framework following the standard model interface. 
The original implementation of L2BT was designed to operate on the **VisA** dataset, and the default configuration provided in this integration uses **VisA**.

## Method Overview

L2BT follows a teacher–student paradigm composed of three main components:

**Teacher network**  
A frozen Vision Transformer that extracts patch-level representations from the input image.

**Student networks**  
Two lightweight MLP networks trained to approximate the feature mappings produced by the teacher.

**Anomaly map generator**  
Computes anomaly scores from the discrepancy between teacher and student features.

During training, the students learn to reproduce the teacher features on normal images.  
At inference time, anomalous regions produce larger discrepancies between teacher and student representations, which are converted into anomaly maps.

This design allows L2BT to process high-resolution images efficiently while maintaining strong localization performance for both large and small defects.

## Usage

Example training with anomalib:

```bash
anomalib train \
  --model l2bt \
  --dataset visa \
  --dataset.category capsules
```

Example inference:

```bash
anomalib predict \
  --model l2bt \
  --dataset visa \
  --dataset.category capsules
```

Model parameters can be configured in:

```
src/anomalib/models/image/l2bt/config.yaml
```

## Reference

If you use this model in your research, please cite the original paper:

```bibtex
@article{costanzino2025learning2be,
  author = {Costanzino, Alex and Zama Ramirez, Pierluigi and Lisanti, Giuseppe and Di Stefano, Luigi},
  title = {Learning to Be a Transformer to Pinpoint Anomalies},
  journal = {IEEE Access},
  year = {2025}
}
```