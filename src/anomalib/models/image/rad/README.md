# RAD

This is the implementation of the [RAD](https://arxiv.org/abs/2601.22763) paper.

Model Type: Segmentation

## Description

RAD (Retrieval-based Anomaly Detection) is a training-free framework that detects anomalies through multi-level retrieval with position-aware patch matching. It uses a DINOv3 ViT-B/16 backbone to extract features from multiple transformer blocks and stores them in a multi-layer memory bank during the fitting phase.

At inference, the model first retrieves the k most similar training images using CLS token similarity, then performs patch-level anomaly scoring using cosine distance against a local memory built from the retrieved images. Position-aware matching restricts each query patch to compare only against patches from nearby spatial locations.

## Architecture

The model consists of three stages:

1. **Feature Extraction**: Multi-layer ViT features (CLS + patch tokens) from blocks 3, 6, 9, 11 (0-based)
2. **Image-level Retrieval**: Find k nearest training images via CLS token cosine similarity
3. **Patch-level Scoring**: Position-aware cosine distance between test patches and retrieved memory patches, fused across layers

## Usage

`anomalib train --model Rad --data MVTecAD --data.category <category>`
