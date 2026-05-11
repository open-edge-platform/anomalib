---
name: models-data
description: Reviews anomalib model, data, callback, metric, and CLI integration conventions
---

# Anomalib Models and Data Review

Use this skill when reviewing changes under models, data, callbacks, metrics, pipelines, deployment, or CLI integration.

## Purpose and scope

Use this skill for architectural fit. It is most useful when a change affects how anomalib models are built,
configured, loaded, or connected to data, callbacks, metrics, or CLI/config entrypoints.

## Request changes when

- a model does not fit the established `AnomalibModule`-based architecture;
- structured data is replaced with ad hoc dictionaries where anomalib already has typed item/batch dataclasses;
- callback or engine behavior bypasses existing Lightning or anomalib extension points;
- public metrics, models, or CLI components are added without matching exports, docs, or config compatibility;
- user-facing constructor/config surfaces become opaque or harder to serialize.

## Models

- New trainable models should fit the existing `AnomalibModule`-based architecture.
- Review whether the model integrates cleanly with existing model discovery and loading patterns in `src/anomalib/models/__init__.py`.
- Prefer Lightning hooks and existing framework extension points over ad hoc training side effects inside model bodies.
- Check that configurable constructor arguments are explicit and compatible with the repo's `jsonargparse`-driven config flow.

## Data and dataclasses

- Data should move through anomalib's typed dataclass system rather than ad hoc dictionaries where the library already has structured item/batch types.
- Changes to shared dataclasses should preserve validation and batching behavior.
- `src/anomalib/data/dataclasses/generic.py` is the main reference for `FieldDescriptor`, typed fields, update behavior, and batch/item patterns.
- If a dataclass surface changes, review both runtime behavior and the corresponding public documentation.

## Callbacks and engine integration

- Training side effects such as checkpointing, timing, compression, or visualization should follow the callback-based patterns already present in `src/anomalib/callbacks/`.
- New callback-style behavior should align with Lightning callback usage instead of bypassing the engine lifecycle.
- Do not approve model/callback changes that tightly couple to trainer internals when a documented hook already exists.

## Metrics

- Metrics should align with the torchmetrics-based patterns in `src/anomalib/metrics/base.py`.
- Review whether image-level and pixel-level metric handling remains clear and consistent.
- If a metric becomes part of the public API, verify that exports and docs are updated too.

## CLI and config compatibility

- Review whether user-configurable types remain compatible with `jsonargparse` and the existing CLI/config structure.
- Prefer explicit constructor parameters over opaque config plumbing when the component is user-facing.
- If a new public component is configurable, ask whether config-driven usage and import paths are documented and tested.

## Repo-grounded review anchors

- `src/anomalib/models/__init__.py`
- `src/anomalib/data/dataclasses/generic.py`
- `src/anomalib/callbacks/__init__.py`
- `src/anomalib/metrics/base.py`
- `src/anomalib/cli/cli.py`

## Review prompts

- Does the change fit anomalib's module, data, and callback architecture?
- Is structured data still flowing through the established dataclass/batch system?
- Will this remain usable from config files and CLI entrypoints?
- Are public exports, docs, and integration points updated alongside the code?

## Reviewer checklist

- Check model architecture fit.
- Check typed data flow.
- Check callback and metric integration.
- Check CLI/config compatibility.
- Check exports and docs for new public surfaces.
