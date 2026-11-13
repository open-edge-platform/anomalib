# ADR-005: Dual-Mode Alpha Blending Support (Uniform vs. Score-Weighted)

- **Status:** `PROPOSED`
- **Scope:** Rendering Semantics & Visual Consistency

---

## Context
Codebase inspection revealed divergent alpha blending semantics across Anomalib tools:
1. **Library Visualizer:** Implements uniform global alpha blending across the entire image canvas ($I_{\text{out}} = \alpha \cdot H + (1 - \alpha) \cdot I_{\text{base}}$), where $\alpha \in [0.0, 1.0]$ is a global scalar.
2. **Studio Backend:** Implements score-weighted per-pixel blending ($I_{\text{out}}(x, y) = \Omega(x, y) \cdot H(x, y) + (1 - \Omega(x, y)) \cdot I_{\text{base}}(x, y)$), where transparency is modulated per pixel based on the anomaly score.

Because of this divergence, identical predictions render visibly differently between the library evaluation tools and the Studio live inspection view.

---

## Proposed Decision
Support both blending modes within the modular rendering engine behind a unified interface:
1. **Uniform Blending:** Applied when a scalar float $\alpha$ is supplied or when `opacity_map is None` in uniform mode.
2. **Score-Weighted Blending:** Applied when a 2D float `opacity_map` is provided in `VisualScene`, or derived dynamically as `normalize_for_visualization(anomaly_map) * global_alpha`.

---

## Alternatives Considered
- *Forcing Uniform Blending Everywhere:* Rejected because Studio users rely on score-weighted blending to keep normal background regions uncolored.
- *Forcing Score-Weighted Blending Everywhere:* Rejected because it alters established visual behavior in library evaluation benchmarks and test figure outputs.

---

## Consequences and Trade-offs
- Eliminates visual discrepancy by allowing either semantic to be selected explicitly.
- Unifies blending algorithms into a single optimized implementation.

---

## Evidence & Source References
- Library uniform blend: `src/anomalib/visualization/image/functional.py:280-305`
- Studio score-weighted blend: `application/backend/src/utils/visualization.py:63-67`
- Mathematical specification in RFC: [`../RFC.md#7-annotation-and-composition-design`](../RFC.md#7-annotation-and-composition-design)
