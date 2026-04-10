---
name: model-sample-image-export
description: Export, validate, and publish model sample-result images into docs/source/images and reference them from README/docs pages. Use when model sample images are missing, outdated, or suspected to be invalid.
---

# Model Sample Image Export

Use this skill to create or refresh sample-result images for model documentation.

## Scope

This skill focuses on:

- selecting completed trained checkpoints or finished benchmark runs
- exporting prediction/sample images
- copying or saving them into `docs/source/images/<model>/results/`
- updating README/docs sample-result references
- rejecting broken or misleading outputs

It does not own benchmark table maintenance. Use `benchmark-and-docs-refresh` for that.

## Request changes when

- sample images come from incomplete or untrusted runs;
- published outputs are clearly degenerate or misleading;
- README or docs references point to missing image files;
- the docs surface implies three valid examples when fewer trustworthy outputs exist.

## Required Source Quality

Only use sample images from:

- completed trained checkpoints
- completed benchmark runs with valid prediction outputs
- finished model outputs that can be traced back to a real run artifact
- if trained checkpoint is available or completed benchmark is available, schedule a few runs to train the model to
  generate more sample images

Do not use:

- incomplete runs
- partially written checkpoints
- outputs with empty/degenerate masks
- outputs driven by NaNs or obviously broken predictions

## Required Workflow

1. Identify candidate checkpoints/runs in `results/`.
2. Verify the run is complete enough to trust.
3. If verification fails, schedule a few runs to train the model on a few categories.
4. Generate predictions from the checkpoint/run.
5. Inspect output quality before publishing images.
6. Save the selected images into `docs/source/images/<model>/results/`.
7. Update README/docs references.

## Preferred Output Layout

- `docs/source/images/<model>/results/0.png`
- `docs/source/images/<model>/results/1.png`
- `docs/source/images/<model>/results/2.png`

If you have fewer than 3 trustworthy images, train the model on a few more categories to generate more sample images.

## README Update Pattern

Preferred pattern:

```md
### Sample Results

![Sample Result 1](/docs/source/images/<model>/results/0.png "Sample Result 1")
```

Repeat for additional images.

## Docs Update Pattern

Preferred docs-page pattern:

````md
    ## Sample Results

    ```{eval-rst}
    .. image:: ../../../../../images/<model>/results/0.png
    ```
````

## Validation Rules

Before publishing an image:

1. Check that the referenced file exists.
2. Check that the image is visually plausible.
3. Check that the mask/anomaly region is not obviously wrong.
4. Check that the sample came from a trained or otherwise valid completed run.
5. If a model/category output is degenerate, exclude it and say so explicitly.

## Reviewer checklist

- Check run completeness.
- Check image quality.
- Check exported file existence.
- Check README and docs references.

## Repo-Specific Notes

- In this repo, some completed checkpoints can still produce bad masks.
- If generic visualization helpers fail, derive a narrow exporter for the specific model/run.
- Keep exporter scripts focused and traceable to the chosen checkpoints.
- When in doubt, prefer fewer trustworthy sample images over a full set of misleading ones.
