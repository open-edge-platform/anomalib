# ADR-001: Extraction of Visualization to a Standalone Distribution / Subpackage

- **Status:** `PROPOSED`
- **Scope:** Package Architecture & Distribution

---

## Context
Visualization functionality in Anomalib is currently duplicated across three independent Python implementations:
1. Modern Library Visualizer: `src/anomalib/visualization/image/`
2. Legacy Library Visualizer: `src/anomalib/utils/visualization/image.py`
3. Studio Backend Visualizer: `application/backend/src/utils/visualization.py`

This causes high maintenance overhead, divergent visual rendering (uniform vs. score-weighted alpha blending), and repeated implementations of basic primitives like Jet lookup tables, status badges, and image resizing.

---

## Proposed Decision
Extract visualization out of `anomalib` core into a separate, modular package tentatively named **`anomalib-viz`** (or subpackage `anomalib.viz`), shared by both the core library and the Studio application.

---

## Alternatives Considered
- *Status Quo (Maintain 3 separate implementations):* Rejected due to continuous maintenance drag, duplicate bug surfaces, and inconsistent user experiences.
- *In-Tree Monolithic Submodule without Standalone Distribution:* Rejected because edge deployment scenarios (e.g. OpenVINO on edge gateways or microservices) would still be forced to install heavy deep learning training dependencies merely to render an overlay.

---

## Consequences and Trade-offs
- Provides a single, clean boundary for visualization across the library and application.
- Allows edge inference runtimes and Studio streaming workers to render overlays without installing PyTorch or Lightning.
- Requires maintaining package publishing automation if released as an independent distribution on PyPI.

---

## Evidence & Source References
- Issue #3872 description: [open-edge-platform/anomalib#3872](https://github.com/open-edge-platform/anomalib/issues/3872)
- AST scan report: [`../RFC.md#32-ast-dependency-and-coupling-analysis`](../RFC.md#32-ast-dependency-and-coupling-analysis)
