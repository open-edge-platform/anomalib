#!/usr/bin/env bash
# shellcheck shell=bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Getting Started with Anomalib Inference
# This example shows how to perform inference using Engine().predict() arguments.

echo "=== Anomalib Inference Examples ==="

echo -e "\n1. Basic Inference with Checkpoint Path"
echo "# Predict using a model checkpoint"
anomalib predict \
    --model EfficientAd \
    --ckpt_path "./results/EfficientAd/MVTecAD/bottle/latest/weights/lightning/model.ckpt" \
    --data path/to/image.jpg

echo -e "\n2. Inference with Directory Path"
echo "# Predict on all images in a directory"
anomalib predict \
    --model EfficientAd \
    --ckpt_path "./results/EfficientAd/MVTecAD/bottle/latest/weights/lightning/model.ckpt" \
    --data "./datasets/MVTecAD/bottle/test"

echo -e "\n3. Inference with Datamodule"
echo "# Use a datamodule for inference"
anomalib predict \
    --model Patchcore \
    --ckpt_path "./results/Patchcore/my_dataset/latest/weights/lightning/model.ckpt" \
    --data.class_path anomalib.data.Folder \
    --data.init_args.name "my_dataset" \
    --data.init_args.root "./datasets/my_dataset" \
    --data.init_args.normal_dir "good" \
    --data.init_args.abnormal_dir "defect"

echo -e "\n4. Inference with Return Predictions"
echo "# Request prediction batches; visualizations are still saved"
anomalib predict \
    --model EfficientAd \
    --ckpt_path "./results/EfficientAd/MVTecAD/bottle/latest/weights/lightning/model.ckpt" \
    --data path/to/image.jpg \
    --return_predictions true

echo -e "\nNote: Replace paths according to your setup."
echo "Visualizations are saved in the experiment's images directory by default, independently of --return_predictions."
echo "Use --model.visualizer false to disable visualization saving."
