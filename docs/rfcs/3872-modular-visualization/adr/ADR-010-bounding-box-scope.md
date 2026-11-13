# ADR-010: Deferral of Object Detection and Bounding Box Visualization from Phase 1

- **Status:** `PROPOSED`
- **Scope:** Scope Boundaries & Task Types

---

## Context
Earlier discussions proposed including bounding box annotations in the initial `VisualScene` contract. However, when I audited Anomalib's core task enum (`src/anomalib/__init__.py:65-72`), I confirmed that `TaskType` defines only `CLASSIFICATION` and `SEGMENTATION`. Object detection (`DETECTION`) is not implemented or supported in core Anomalib. Passing `task="detection"` to the legacy visualizer crashes with `ValueError: Unknown task type: detection`.

---

## Proposed Decision
Strictly exclude bounding box annotations from Phase 1 scope:
1. `VisualScene` initially supports image, anomaly map, opacity map, prediction mask, ground truth mask, score, and label.
2. If object detection is introduced to core Anomalib in a future release, bounding boxes must be introduced as a first-class domain representation rather than through ad-hoc metadata packing.

---

## Alternatives Considered
- *Adding Bounding Box Fields in Phase 1:* Rejected because it introduces unverified coordinate schemas and normalization conventions without a supporting model architecture in Anomalib.
- *Packing Bounding Boxes into Metadata:* Rejected because `metadata` should not serve as an unofficial substitute for domain models.

---

## Consequences and Trade-offs
- Prevents scope creep and keeps the initial package focused on active Anomalib tasks.
- Detection visualization is cleanly deferred to Phase 2.

---

## Evidence & Source References
- Task enum: `src/anomalib/__init__.py:65-72`
- Scope discussion in RFC: [`../RFC.md#4-requirements-and-proposed-goals`](../RFC.md#4-requirements-and-proposed-goals)
