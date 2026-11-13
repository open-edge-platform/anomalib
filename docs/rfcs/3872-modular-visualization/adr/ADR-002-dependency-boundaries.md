# ADR-002: Strict Pure NumPy Core (Zero PyTorch or Lightning in Core Contract)

- **Status:** `PROPOSED`
- **Scope:** Core Dependency Boundaries

---

## Context
The modern library visualizer (`src/anomalib/visualization/image/visualizer.py:53`) directly imports PyTorch and Lightning. In `src/anomalib/visualization/image/functional.py:187-278`, it builds a 256-color Jet palette by interpolating a 9-color anchor tensor using `torch.nn.functional.interpolate`. Users deploying models with lightweight inference engines (such as OpenVINO on CPU or embedded devices) are forced to install gigabytes of deep learning dependencies merely to render an inspection image.

---

## Proposed Decision
The core domain model (`VisualScene`), normalization routines, and annotator interfaces must contain zero imports of `torch`, `torchvision`, `lightning`, or `torchmetrics`, establishing a dependency lower bound of `numpy >= 1.26.0`. NumPy 1.26 supports the core Python 3.10–3.12 environments. Python 3.13 environments require a Python 3.13-compatible NumPy release; the current Studio lockfile resolves NumPy 2.3.5. The lower bound allows the resolver to select a compatible release, but does not imply that NumPy 1.26 supports Python 3.13.

---

## Alternatives Considered
- *Retaining PyTorch Tensor Support in Core:* Rejected because it re-introduces heavy framework dependencies for standalone inference runtimes.
- *Optional Soft Imports inside Functions:* Rejected because runtime conditional imports add overhead and fragile error handling compared to a strict architectural boundary.

---

## Consequences and Trade-offs
- The core package is lightweight, fast to install, and universally compatible across CPU, CUDA, and edge environments.
- Callers holding PyTorch GPU/CPU tensors must convert them to NumPy arrays via adapters before invoking `anomalib-viz` core functions.

---

## Evidence & Source References
- PyTorch colormap interpolation: `src/anomalib/visualization/image/functional.py:187-278`
- Analysis in RFC: [`../RFC.md#10-dependencies-and-package-boundaries`](../RFC.md#10-dependencies-and-package-boundaries)
