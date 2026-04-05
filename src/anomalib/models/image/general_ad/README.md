# GeneralAD

GeneralAD is a discriminative anomaly detection method introduced in
"GeneralAD: Anomaly Detection Across Domains by Attending to Distorted Features".

This anomalib integration ports the core method from the original
[GeneralAD repository](https://github.com/LucStrater/GeneralAD) into anomalib's
standard model interface so it can be trained with `Engine`, used from the CLI,
and evaluated with anomalib's built-in metrics and post-processing pipeline.
