# ADR-004: Raw Anomaly Map Preservation, Display Normalization, and Opacity Map Separation

- **Status:** `PROPOSED`
- **Scope:** Data Integrity & Normalization

---

## Context
Existing visualizer implementations conflate three distinct mathematical concepts:
1. Raw continuous model distance scores (e.g. Mahalanobis distance, reconstruction error), which may be uncalibrated, negative, or unbounded ($[0, \infty)$ or $(-\infty, \infty)$).
2. Display-normalized maps scaled to $[0.0, 1.0]$ for colormapping.
3. Per-pixel alpha transparency maps used for blending.

When raw model outputs are scaled in-place or assumed to be strictly bounded in $[0.0, 1.0]$, metric distance semantics are destroyed. Furthermore, constant anomaly maps cause division-by-zero crashes when scaled with `np.ptp`, and uncalibrated models or non-finite values (`NaN`, $\pm\infty$) corrupt downstream graphics pipelines.

---

## Proposed Decision
1. **Raw Score Preservation:** `VisualScene.anomaly_map` preserves raw, continuous model output values. Downstream routines never modify raw anomaly scores in-place.
2. **Display-Only Normalization on Isolated Copy:** `normalize_for_visualization(raw_map)` operates strictly on an allocated copy:
   - Non-finite values are sanitized via `np.nan_to_num(norm_map, nan=min_f, posinf=max_f, neginf=min_f)`.
   - Constant anomaly maps ($\max - \min < 10^{-7}$) deterministically map to zero arrays, avoiding division-by-zero crashes.
3. **Distinct Opacity Map:** `VisualScene.opacity_map` is an independent optional field, strictly bounded in $[0.0, 1.0]$ and verified finite (`np.all(np.isfinite)`), used specifically for score-weighted spatial blending.

---

## Alternatives Considered
- *Forcing Raw Outputs to [0, 1] in Domain Models:* Rejected because it destroys raw metric distance semantics of models such as PaDiM or Patchcore.
- *In-Place Clamping / Normalization:* Rejected because mutating caller prediction arrays corrupts downstream metric evaluation.

---

## Consequences and Trade-offs
- Preserves numerical integrity of raw model outputs for debugging and metric calculation.
- Guarantees robust, deterministic rendering without division-by-zero crashes on constant anomaly maps.
- Requires allocating an isolated working copy when normalizing for display.

---

## Evidence & Source References
- Divide-by-zero bug: `src/anomalib/utils/post_processing.py:236-239`
- Normalization policy in RFC: [`../RFC.md#62-normalization-and-sanitization-policy`](../RFC.md#62-normalization-and-sanitization-policy)
