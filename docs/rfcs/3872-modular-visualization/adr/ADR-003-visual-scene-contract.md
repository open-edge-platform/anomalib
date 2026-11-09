# ADR-003: Canonical In-Memory Contract (`VisualScene` Dataclass)

- **Status:** `PROPOSED`
- **Scope:** Domain Model & Data Contract

---

## Context
Visualization inputs originate from diverse sources with incompatible memory representations:
1. PyTorch Lightning callbacks: GPU/CPU 4D tensors (`ImageBatch`).
2. OpenVINO inferencer: Host NumPy NCHW float32 arrays (`NumpyImageBatch`).
3. Studio video workers: Live OpenCV camera frames (HWC uint8 in BGR format) and 2D anomaly heatmaps.
4. Studio REST API: Serialized Base64 PNG payloads.

Without a canonical domain representation, each visualization routine must anticipate and branch on multiple container formats.

---

## Proposed Decision
Define `VisualScene` as the canonical in-memory data contract for visualization inputs:
1. Defined as a `@dataclass(frozen=True)` holding pure NumPy host arrays.
2. Immutability guarantee: Provides shallow frozen immutability on object attributes. Downstream rendering routines treat underlying array buffers as strictly read-only and never mutate them in-place.
3. Explicit color space: Requires `image` as an $H \times W \times 3$ uint8 array in RGB color order.
4. Separate arrays for raw model anomaly scores (`anomaly_map: 2D float32`) and spatial alpha transparency (`opacity_map: 2D float in [0.0, 1.0]`).
5. Reference-preserving construction ($O(1)$) without deep copying.

---

## Alternatives Considered
- *Reusing PyTorch `ImageBatch`:* Rejected because it forces non-PyTorch runtimes (OpenVINO, ONNX, Studio) to install PyTorch.
- *Reusing Legacy `ImageResult`:* Rejected due to brittle binary mask handling (`{0, 255}` crash in `__post_init__`) and Matplotlib figure state coupling.
- *Dictionary-based Contract:* Rejected due to lack of static type safety and schema validation.

---

## Consequences and Trade-offs
- Provides a clear, stable contract across the entire project.
- Guarantees zero deep-copy memory allocation overhead during scene construction.
- Requires callers to pass valid NumPy arrays conforming to the contract.

---

## Evidence & Source References
- Contract specification in RFC: [`../RFC.md#6-canonical-visualization-data-contract-visualscene`](../RFC.md#6-canonical-visualization-data-contract-visualscene)
