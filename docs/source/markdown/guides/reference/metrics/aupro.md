# AUPRO

```{note}
Anomalib v0.3 to v2.2.x normalized the area under the PRO curve by the last FPR value
of the averaged per-region curve instead of by `fpr_limit`. When that value fell short
of the limit, the reported AUPRO was biased compared with the MVTec AD reference
evaluation and could exceed 1. Since v2.3.0 the curve is clipped exactly at
`fpr_limit` and the area is normalized by `fpr_limit`, matching the reference. AUPRO
scores computed with older versions should be recomputed before comparing them with
newer results. See [#3718](https://github.com/open-edge-platform/anomalib/issues/3718).
```

```{eval-rst}
.. autoclass:: anomalib.metrics.AUPRO
   :members:
   :show-inheritance:
```
