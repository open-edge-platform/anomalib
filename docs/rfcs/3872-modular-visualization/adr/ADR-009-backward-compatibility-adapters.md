# ADR-009: Phased Compatibility Adapters for Existing Lightning Models and Visualizers

- **Status:** `PROPOSED`
- **Scope:** Backward Compatibility & Migration

---

## Context
Static AST analysis across the repository revealed that **34 concrete Lightning model classes** declare `visualizer: Visualizer | bool = True` in their `__init__` constructors. Furthermore, `AnomalibModule:458` defines `configure_visualizer() -> ImageVisualizer`, and `_VisualizationCallback` hooks into Lightning test loops. Refactoring all 34 model files immediately would introduce widespread breaking changes across the entire repository.

---

## Proposed Decision
Provide backward-compatible adapter wrappers for `AnomalibModule.configure_visualizer()`, `_VisualizationCallback`, and legacy `Visualizer` classes during Phase 1:
1. Retain constructor signatures across all 34 model classes.
2. Route legacy visualizer calls through `BatchToVisualSceneAdapter` into the new rendering engine.
3. Emit deprecation warnings throughout a proposed two-release deprecation window across Anomalib v2.x, scheduling legacy wrapper removal for Anomalib v3.0 (subject to maintainer approval in OPEN-06).

---

## Alternatives Considered
- *Immediate Breaking Refactor of All 34 Models:* Rejected because it introduces massive breaking changes across the library in a single release.
- *Indefinite Dual Maintenance:* Rejected because maintaining two competing visualizer codebases indefinitely incurs high maintenance overhead.

---

## Consequences and Trade-offs
- Preserves 100% compatibility for existing training and evaluation scripts during the migration window.
- Zero model files require immediate code modifications in Phase 1.

---

## Evidence & Source References
- AST model scan in RFC: [`../RFC.md#32-ast-dependency-and-coupling-analysis`](../RFC.md#32-ast-dependency-and-coupling-analysis)
- Migration roadmap in RFC: [`../RFC.md#11-backward-compatibility-and-migration`](../RFC.md#11-backward-compatibility-and-migration)
