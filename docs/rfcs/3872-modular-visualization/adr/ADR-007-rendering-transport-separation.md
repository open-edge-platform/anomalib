# ADR-007: Separation of Pixel Rendering from Output Sinks (Storage, Streaming, Logging)

- **Status:** `PROPOSED`
- **Scope:** Architecture & Separation of Concerns

---

## Context
In the existing implementations, pixel generation is tightly coupled to output destinations:
1. `src/anomalib/utils/visualization/image.py` mixes Matplotlib canvas generation with interactive display (`plt.show()`) and file saving (`plt.savefig()`).
2. `application/backend/src/services/model_service.py:344` couples OpenCV colormapping directly to Base64 PNG string encoding.
3. `src/anomalib/visualization/image/visualizer.py` couples rendering directly to PyTorch Lightning callback hooks.

---

## Proposed Decision
Strictly decouple pixel rasterization from downstream output sinks:
1. The rendering pipeline operates purely in-memory, outputting standard RGB NumPy arrays or PIL Images.
2. Output destinations (saving to disk, streaming over WebRTC/MJPEG, logging to Weights & Biases or TensorBoard) are handled by dedicated consumer handlers outside the rendering engine.

---

## Alternatives Considered
- *Monolithic Visualizer Handling All Sinks:* Rejected because mixing I/O and network operations with rasterization impairs testability and code reuse.

---

## Consequences and Trade-offs
- Maximizes testability: rendering algorithms can be verified in memory without disk or network mocks.
- Sinks can be composed and chained freely by consumers.

---

## Evidence & Source References
- Backend protocol in RFC: [`../RFC.md#8-rendering-and-output-separation`](../RFC.md#8-rendering-and-output-separation)
