# ADR-008: Conditional Consumer Adapters and Explicit BGR to RGB Color Transformation

- **Status:** `PROPOSED`
- **Scope:** Adapter Architecture & Pipeline Integration

---

## Context
Earlier drafts proposed introducing mandatory separate adapters for every consumer (`LightningBatchAdapter`, `InferencerResultAdapter`, `StudioWorkerAdapter`), adding redundant abstraction layers without evidence.

---

## Proposed Decision
1. Apply the **Conditional Adapter Principle**: Callers construct `VisualScene` directly whenever their data structures are compatible. Minimal adapters are introduced only where structural impedance genuinely requires them (e.g. `BatchToVisualSceneAdapter` for unpacking batched NCHW PyTorch tensors).
2. In Studio's in-memory live streaming pipeline, converting BGR to RGB is an explicit color-space transformation that must allocate a **newly allocated RGB array buffer** (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`). This ensures the original BGR frame is not modified in-place, preserving its integrity for downstream video sinks. Once converted, Studio constructs `VisualScene` directly without requiring an adapter.

---

## Alternatives Considered
- *Mandatory Separate Adapters for Every Consumer:* Rejected because Studio's in-memory worker already possesses HWC uint8 frames and 2D float32 anomaly maps once converted to RGB.
- *In-Place BGR to RGB Conversion:* Rejected because in-place mutation corrupts raw video buffers used by downstream BGR streaming sinks.

---

## Consequences and Trade-offs
- Eliminates unnecessary wrapper layers and keeps the architecture lean.
- Preserves memory isolation and sink integrity in multi-threaded streaming pipelines.

---

## Evidence & Source References
- Studio inference worker: `application/backend/src/workers/inference.py:209-240`
- Integration pathways in RFC: [`../RFC.md#9-library-and-studio-integration`](../RFC.md#9-library-and-studio-integration)
