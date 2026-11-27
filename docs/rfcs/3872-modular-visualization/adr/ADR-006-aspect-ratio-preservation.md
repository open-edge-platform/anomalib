# ADR-006: Aspect Ratio Preservation via Letterbox Resizing

- **Status:** `PROPOSED`
- **Scope:** Layout & Spatial Alignment

---

## Context
In `src/anomalib/visualization/image/item_visualizer.py:351`, the visualizer resizes all image fields using `image.resize(field_size)` directly, where `field_size=(256, 256)`. When passed a standard 16:9 industrial inspection frame (`1920x1080`), it squashes the image non-uniformly into a square, distorting defect aspect ratios and spatial geometry.

---

## Proposed Decision
Multi-panel layout generators and visualizer grids must implement letterbox resizing with padding rather than direct non-uniform resizing. Images are scaled proportionally to fit within target dimensions, with neutral padding bars added as necessary to preserve native aspect ratios.

---

## Alternatives Considered
- *Retaining Direct Non-Uniform Resize:* Rejected because distorting defect geometry impairs visual quality in industrial inspection.
- *Center Cropping:* Rejected because cropping discards valid image content at the peripheries where edge defects often occur.

---

## Consequences and Trade-offs
- Preserves defect geometry and camera aspect ratios.
- Multi-panel comparison grids may display neutral padding bars when formatting non-square images.

---

## Evidence & Source References
- Direct resize call: `src/anomalib/visualization/image/item_visualizer.py:351`
- Defect analysis in RFC: [`../RFC.md#33-empirical-code-audit-and-defect-verification`](../RFC.md#33-empirical-code-audit-and-defect-verification)
