# Contributing to Anomalib

This document is the checklist for code and PRs: setup, DCO, structure for models/data, and review expectations. For **bugs, discussions, and feature requests**, use [Issues](https://github.com/open-edge-platform/anomalib/issues) and [Discussions](https://github.com/open-edge-platform/anomalib/discussions) as in the project README.

## Contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [DCO sign-off](#dco-sign-off)
4. [Environment and checks](#environment-and-checks)
5. [Adding a new model](#adding-a-new-model)
6. [Adding a new dataset](#adding-a-new-dataset)
7. [Extending other components](#extending-other-components)
8. [Bug fixes](#bug-fixes)
9. [Pull requests](#pull-requests)
10. [PR checklist](#pr-checklist)
11. [Maintainers: do’s and don’ts](#maintainers-dos-and-donts)

---

## Quick start

1. Fork the repo; branch from latest `main` ([branch naming](#pull-requests)).
2. Install dev deps, hook **prek**, run **prek** + **pytest** before push ([below](#environment-and-checks)).
3. Sign commits with **DCO** (`git commit -s`).
4. Open a PR whose **title** is Conventional Commits ([below](#pull-requests))—it becomes the squash merge message.

---

## Repository layout

The architecture follows a hierarchical structure:

- **Top Level:** Configs and the CLI.
- **Middle Level:** **`Engine`** (train/val/test/predict).
- **Bottom Level Components:**
  - **Data (`src/anomalib/data/`):** **`AnomalibDataModule` + dataset**.
  - **Models:** **`AnomalibModule`** (`src/anomalib/models/components/base/anomalib_module.py`).

> **Note:** New models are only discoverable if their subclass is **imported** with the rest of `anomalib.models`—`get_model` / `list_models` use `AnomalibModule.__subclasses__()`.

---

## DCO sign-off

This project uses the [Developer Certificate of Origin (DCO)](https://developercertificate.org/). **Every commit** must end with a `Signed-off-by` line (real name and email).

**Signing a new commit:**
```bash
git commit -s -m "fix(data): describe change"
```

**If you already committed without sign-off:**
```bash
git rebase HEAD~N --signoff   # N = commits to fix
git push --force-with-lease origin <branch>
```

> Contributions are accepted under **Apache-2.0** (see `LICENSE`).

---

## Environment and checks

### 1. Setup

```bash
conda create -n anomalib_dev python=3.10 && conda activate anomalib_dev
pip install -e .[dev]    # or: anomalib install --option dev
prek install
```

### 2. Pre-push Checks

Before you push your branch, always run:

```bash
prek run --all-files
pytest   # or: pytest path/to/test_file.py
```

### 3. Guidelines
- **Local WIP:** Use `git commit --no-verify` for local work in progress only (**do not** leave this for the final PR).
- **Style Rules:** Regulated by **Ruff** in `pyproject.toml`.
- **Testing:** Prefer fast tests under `tests/unit/`.

---

## Adding a new model

### 1. File Structure
Mirror an existing image model (e.g. `src/anomalib/models/image/padim/`):

```text
src/anomalib/models/image/<your_model>/
  ├── __init__.py           # export YourModel
  ├── lightning_model.py    # YourModel(AnomalibModule), …
  └── torch_model.py        # nn.Module (typical)
```

### 2. Registration (Required)
Without these imports, the class never appears in `list_models` / `get_model` (this is a **common mistake**!):
1. Export from `src/anomalib/models/image/__init__.py`.
2. Export from `src/anomalib/models/__init__.py` and add to **`__all__`** there.

### 3. Example Config (CLI / YAML workflows)
```yaml
# examples/configs/model/your_model.yaml
model:
  class_path: anomalib.models.YourModel
  init_args:
    backbone: resnet18
```

### 4. Tests
- At least construction on **CPU**.
- Forward/shape or light smoke testing where cheap.
- Regression test if you are fixing a bug.

---

## Adding a new dataset

### Steps:
1. **Dataset Class:** Subclass **`AnomalibDataset`** (`src/anomalib/data/datasets/base/image.py`).
2. **DataModule:** Wrap with a **`LightningDataModule`** subclass of **`AnomalibDataModule`** (`src/anomalib/data/datamodules/base/image.py`).
3. **References:** Use **`Folder`** (`src/anomalib/data/datamodules/image/folder.py`) and the base image datamodule as references before inventing a new layout.

### Edge cases to handle explicitly (and test):
- Missing masks/labels.
- Optional masks per split.
- Tiny images.
- Ensure `task` (classification vs segmentation) matches what metrics expect.

---

## Extending other components

For elements like:
- **Post-processors:** `src/anomalib/post_processing/`
- **Pre-processors:** *Same as above*
- **Metrics:** `src/anomalib/metrics/`
- **Visualization:** `src/anomalib/visualization/`
- **Utils:** `src/anomalib/utils/`, `src/anomalib/data/utils/`

**Guidelines:**
1. Follow existing modules.
2. Keep APIs small.
3. Add tests.
4. Expose behavior via YAML **`class_path` + `init_args`** when users need to select it from the CLI.

---

## Bug fixes

1. **Reproduce** on current `main` (minimal command or short script; note `python` / `torch` versions if relevant).
2. **Fix** with the smallest change that solves it (avoid drive-by refactors in the same PR unless agreed).
3. **Add a regression test** that fails before and passes after (usually `tests/unit/...`).

---

## Pull requests

### 1. Title (Required)
Squash merge uses the PR title as the final commit subject. 

**Format:**
```text
<type>(<scope>): <description in lowercase, no period at end>
```
**Examples:** `feat(model): add yourmodel`, `fix(data): handle empty mask dir`

- **Allowed Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
- **Allowed Scopes:** `data`, `model`, `metric`, `utils`, `cli`, `docs`, `ci`, `engine`, `visualization`, `benchmarking`, `logger`, `openvino`, `notebooks`.

### 2. Branch names
**Format:**
```text
<type>/<scope>/<short-description>
```

### 3. Description
- **One coherent change per PR.**
- State intent, non-goals, and how to verify. 
- API or behavior changes need docstrings and, when useful, a config snippet.

### 4. Submitting
Push to your fork and open a PR against the upstream repo; use the provided PR template.

---

## PR checklist

- [ ] **DCO** on every commit (`git commit -s` or `rebase --signoff`).
- [ ] **`prek run --all-files`** and **`pytest`** pass.
- [ ] New behavior covered by tests; bugfixes include a regression test.
- [ ] User-facing changes: docstrings / docs / `CHANGELOG.md` as appropriate.
- [ ] **PR title** matches Conventional Commits.
- [ ] PR is focused (no stray binaries, debug noise, or unrelated refactors).

---

## Maintainers: do’s and don’ts

**Do:** 
- Copy structure from a neighboring feature; 
- Prefer composition (pre/post, evaluator); 
- Keep PRs review-sized; 
- Default tests should run on **CPU** unless marked for GPU.

**Don’t:** 
- Skip model registration; 
- Silently change existing defaults; 
- Merge “WIP”/vague PR titles; 
- Drop tests on research-only grounds.
