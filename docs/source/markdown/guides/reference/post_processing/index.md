# Post-processing

::::{grid} 1 2 2 2
:gutter: 3
:padding: 2

:::{grid-item-card} {octicon}`gear` Base Post-processor
:link: base-post-processor
:link-type: ref

Base class for post-processing.

+++
[Learn more »](base-post-processor)
:::

:::{grid-item-card} {octicon}`gear` One-class Post-processor
:link: one-class-post-processor
:link-type: ref

Post-processor for one-class anomaly detection.

+++
[Learn more »](one-class-post-processor)
:::

:::{grid-item-card} {octicon}`gear` MEBin Post-processor
:link: mebin-post-processor
:link-type: ref

MEBin (Main Element Binarization) post-processor from
[AnomalyNCD](https://arxiv.org/abs/2410.14379). Uses per-image adaptive
thresholds based on connected-component stability analysis.

+++
[Learn more »](mebin-post-processor)
:::
::::
(base-post-processor)=

## Base Post-processor

```{eval-rst}
.. automodule:: anomalib.post_processing.post_processor
   :members:
   :show-inheritance:
```

(one-class-post-processor)=

## One-class Post-processor

```{eval-rst}
.. automodule:: anomalib.post_processing.one_class
   :members:
   :show-inheritance:
```

(mebin-post-processor)=

## MEBin Post-processor

MEBin (Main Element Binarization) is an adaptive binarization method introduced in the
[AnomalyNCD](https://arxiv.org/abs/2410.14379) paper. It
determines per-image thresholds by sweeping thresholds across the anomaly map, counting
connected components at each level, and selecting the threshold at the endpoint of the
longest stable interval.

MEBin is precision-oriented: it favours high-confidence anomaly regions over exhaustive
coverage. Pixel-level F1 may be lower than the default post-processor, but the
resulting masks are well-suited for downstream tasks such as anomaly classification.

```{eval-rst}
.. automodule:: anomalib.post_processing.mebin_post_processor
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.post_processing.mebin
   :members:
   :show-inheritance:
```
