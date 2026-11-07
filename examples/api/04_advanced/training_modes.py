"""Training modes and checkpoint control examples.

This example demonstrates different training modes in Anomalib:
1. Standard training with default checkpointing
2. Custom checkpoint callback configuration
3. Barebones mode for fast training/testing without checkpoint overhead

Note:
    Under the hood, `Engine` uses Lightning's `Trainer` to manage the training
    workflow. So, most Trainer arguments can be passed to the Engine constructor.
    This includes parameters like
    `max_epochs`, `enable_checkpointing`, `barebones`, `logger`, `callbacks`, etc.

    For more details on available parameters, see:
    https://lightning.ai/docs/pytorch/stable/common/trainer.html
"""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Fre

# Initialize model and data
model = Fre()
datamodule = MVTecAD(category="bottle")

print("1. Standard Training with Default Checkpointing")
print("-" * 50)
# 1. Standard training (checkpoints saved automatically)
engine = Engine(max_epochs=5)
engine.fit(model=model, datamodule=datamodule)

print(f"Checkpoint saved at: {engine.best_model_path}")


# 2. Custom checkpoint callback
# Example: don't save any checkpoints (useful for quick tests)

print("2. Custom Checkpoint Callback")
print("-" * 50)
checkpoint_callback = ModelCheckpoint(save_top_k=0)

engine = Engine(max_epochs=5, callbacks=[checkpoint_callback])
print("Training with custom checkpoint callback...")
engine.fit(model=model, datamodule=datamodule)
print(f"Checkpoint path: {engine.best_model_path}")
print()


# 3. Barebones Mode for Maximum Speed
# Barebones mode: minimal overhead, no checkpointing, no model summary, etc.
# Useful for benchmarking with minimal overhead.
# See Lightning docs: https://lightning.ai/docs/pytorch/stable/common/trainer.html#barebones

print("3. Barebones Training Mode")
print("-" * 50)

# Initialize model and data
model = Fre()
datamodule = MVTecAD(category="bottle")

# Create engine with barebones mode enabled
engine = Engine(
    max_epochs=5,
    barebones=True,  # Minimal overhead, no checkpoint saving
)

# Train in barebones mode
print("Training in barebones mode (fastest)...")
engine.fit(model=model, datamodule=datamodule)

# Metrics are still captured and returned even in barebones mode
print("\nTesting and retrieving metrics...")
results = engine.test(model=model, datamodule=datamodule)

# Print results
print("\nTest Results:")
for metric, value in results[0].items():
    print(f"{metric}: {value:.4f}")
