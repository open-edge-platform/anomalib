# Metrics

Anomalib provides a comprehensive set of metrics for evaluating anomaly detection model performance. All metrics extend TorchMetrics' functionality with Anomalib-specific features.

## Available Metrics

### Area Under Curve Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} AUROC
:link: anomalib.metrics.AUROC
:link-type: doc

Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between normal and anomalous samples.
:::

:::{grid-item-card} AUPR
:link: anomalib.metrics.AUPR
:link-type: doc

Area Under the Precision-Recall curve. Particularly useful for imbalanced datasets.
:::

:::{grid-item-card} AUPRO
:link: anomalib.metrics.AUPRO
:link-type: doc

Area Under the Per-Region Overlap curve. Evaluates pixel-level anomaly localization performance.
:::

:::{grid-item-card} AUPIMO
:link: anomalib.metrics.AUPIMO
:link-type: doc

Area Under the Per-Image Missed Overlap curve. Advanced metric for evaluating localization quality.
:::

::::

### F1 Score Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} F1Score
:link: anomalib.metrics.F1Score
:link-type: doc

Standard F1 score for binary classification. Harmonic mean of precision and recall.
:::

:::{grid-item-card} F1Max
:link: anomalib.metrics.F1Max
:link-type: doc

Maximum F1 score across all possible thresholds. Useful for finding optimal operating points.
:::

::::

### Threshold Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} F1AdaptiveThreshold
:link: anomalib.metrics.F1AdaptiveThreshold
:link-type: doc

Automatically determines the optimal threshold by maximizing F1 score.
:::

:::{grid-item-card} ManualThreshold
:link: anomalib.metrics.ManualThreshold
:link-type: doc

Uses a manually specified threshold for classification.
:::

::::

### Other Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} PRO
:link: anomalib.metrics.PRO
:link-type: doc

Per-Region Overlap score for evaluating pixel-level localization.
:::

:::{grid-item-card} PIMO
:link: anomalib.metrics.PIMO
:link-type: doc

Per-Image Missed Overlap for assessing localization errors.
:::

:::{grid-item-card} PGn
:link: anomalib.metrics.PGn
:link-type: doc

Presorted Good with n% bad samples missed. Measures false negative rate at specific operating points.
:::

:::{grid-item-card} PBn
:link: anomalib.metrics.PBn
:link-type: doc

Presorted Bad with n% good samples misclassified. Measures false positive rate at specific operating points.
:::

:::{grid-item-card} MinMax
:link: anomalib.metrics.MinMax
:link-type: doc

Normalizes anomaly scores to [0, 1] range using min-max scaling.
:::

:::{grid-item-card} AnomalyScoreDistribution
:link: anomalib.metrics.AnomalyScoreDistribution
:link-type: doc

Analyzes and tracks the distribution of anomaly scores for model diagnostics.
:::

::::

### Utility Classes

::::{grid} 2
:gutter: 2

:::{grid-item-card} AnomalibMetric
:link: anomalib.metrics.AnomalibMetric
:link-type: doc

Base class for all Anomalib metrics. Extends TorchMetrics with field-based updates.
:::

:::{grid-item-card} Evaluator
:link: anomalib.metrics.Evaluator
:link-type: doc

Orchestrates multiple metrics for comprehensive model evaluation.
:::

:::{grid-item-card} BinaryPrecisionRecallCurve
:link: anomalib.metrics.BinaryPrecisionRecallCurve
:link-type: doc

Computes precision-recall curves for binary classification tasks.
:::

::::

## API Reference

```{eval-rst}
.. automodule:: anomalib.metrics
   :members: AUROC, AUPR, AUPRO, AUPIMO, F1Score, F1Max, F1AdaptiveThreshold, ManualThreshold, PRO, PIMO, PGn, PBn, MinMax, AnomalyScoreDistribution, AnomalibMetric, Evaluator, BinaryPrecisionRecallCurve, create_anomalib_metric
   :undoc-members:
   :show-inheritance:
```
