# Architecture Decision Records: Modular Visualization (`anomalib-viz`)

This directory contains the ten Architectural Decision Records (ADRs) accompanying the modular visualization architecture proposal for Anomalib Issue #3872. Each record captures the context, proposed decision, alternatives evaluated, trade-offs, and supporting codebase evidence for a specific architectural choice.

---

## 1. Governance and ADR Conventions

In accordance with open source governance standards and issue #3872 instructions:
- **Status `PROPOSED`:** Every decision is formally submitted for project maintainer review and consensus. No decision is marked `ACCEPTED` prior to maintainer approval.
- **Open Decisions:** Architectural choices with multiple viable engineering paths requiring maintainer direction are cataloged separately in [**`../RFC.md#132-open-questions-for-maintainer-steering`**](../RFC.md#132-open-questions-for-maintainer-steering).
- **Pre-Implementation Review Gate:** In alignment with issue #3872 instructions, implementation of these decisions will begin only after the RFC and ADRs are reviewed and approved by maintainers (documented in [**`../RFC.md#12-governance-and-pre-implementation-review-gate`**](../RFC.md#12-governance-and-pre-implementation-review-gate)).
- **Format:** Records follow the Michael Nygard ADR format (Context, Proposed Decision, Alternatives Considered, Consequences and Trade-offs, Supporting Evidence).

---

## 2. ADR Master Index

| ADR ID | Decision Title | Scope | Proposed Status | Source File |
| :--- | :--- | :--- | :---: | :--- |
| **[ADR-001](ADR-001-package-structure.md)** | Extraction of Visualization to a Standalone Distribution / Subpackage | Packaging | `PROPOSED` | [`ADR-001-package-structure.md`](ADR-001-package-structure.md) |
| **[ADR-002](ADR-002-dependency-boundaries.md)** | Strict Pure NumPy Core (Zero PyTorch or Lightning in Core Contract) | Dependencies | `PROPOSED` | [`ADR-002-dependency-boundaries.md`](ADR-002-dependency-boundaries.md) |
| **[ADR-003](ADR-003-visual-scene-contract.md)** | Canonical In-Memory Contract (`VisualScene` Dataclass) | Data Model | `PROPOSED` | [`ADR-003-visual-scene-contract.md`](ADR-003-visual-scene-contract.md) |
| **[ADR-004](ADR-004-raw-anomaly-and-opacity-separation.md)** | Raw Anomaly Map Preservation, Display Normalization, and Opacity Map Separation | Data Integrity | `PROPOSED` | [`ADR-004-raw-anomaly-and-opacity-separation.md`](ADR-004-raw-anomaly-and-opacity-separation.md) |
| **[ADR-005](ADR-005-dual-mode-alpha-blending.md)** | Dual-Mode Alpha Blending Support (Uniform vs. Score-Weighted) | Rendering | `PROPOSED` | [`ADR-005-dual-mode-alpha-blending.md`](ADR-005-dual-mode-alpha-blending.md) |
| **[ADR-006](ADR-006-aspect-ratio-preservation.md)** | Aspect Ratio Preservation via Letterbox Resizing | Layout | `PROPOSED` | [`ADR-006-aspect-ratio-preservation.md`](ADR-006-aspect-ratio-preservation.md) |
| **[ADR-007](ADR-007-rendering-transport-separation.md)** | Separation of Pixel Rendering from Output Sinks (Storage, Streaming, Logging) | Architecture | `PROPOSED` | [`ADR-007-rendering-transport-separation.md`](ADR-007-rendering-transport-separation.md) |
| **[ADR-008](ADR-008-conditional-consumer-adapters.md)** | Conditional Consumer Adapters and Explicit BGR to RGB Color Transformation | Architecture | `PROPOSED` | [`ADR-008-conditional-consumer-adapters.md`](ADR-008-conditional-consumer-adapters.md) |
| **[ADR-009](ADR-009-backward-compatibility-adapters.md)** | Phased Compatibility Adapters for Existing Lightning Models and Visualizers | Compatibility | `PROPOSED` | [`ADR-009-backward-compatibility-adapters.md`](ADR-009-backward-compatibility-adapters.md) |
| **[ADR-010](ADR-010-bounding-box-scope.md)** | Deferral of Object Detection and Bounding Box Visualization from Phase 1 | Scope Boundary | `PROPOSED` | [`ADR-010-bounding-box-scope.md`](ADR-010-bounding-box-scope.md) |
