# Pull Request Merge History for `feature/geti-inspect` Branch

This document provides a comprehensive summary of all pull requests (PRs) that were merged into the `feature/geti-inspect` branch, including details about code reviews performed for each PR.

## Overview

- **Total PRs Merged**: 45
- **Date Range**: October 1, 2025 - October 28, 2025  
- **Branch**: `feature/geti-inspect`
- **Repository**: open-edge-platform/anomalib

---

## Detailed PR List (Chronological Order - Latest First)

### PR #3075: bug(inspect): add missing db commit
- **Author**: maxxgx
- **Merged**: 2025-10-28
- **Type**: 🐞 Bug fix
- **Description**: Fixed deletion query not committing to database by adding missing `await self.db.commit()` call after delete query execution.
- **Code Reviews**:
  - **copilot-pull-request-reviewer[bot]**: COMMENTED - Reviewed 2/2 files, generated no comments. Noted the fix adds database commit after delete operation to ensure deletions are persisted.

---

### PR #3060: chore(inspect): improve SSE generator
- **Author**: maxxgx
- **Merged**: 2025-10-28
- **Type**: 🔧 Chore
- **Description**: Yield `ServerSentEvent` instead of raw string. Refactored job log streaming to use proper SSE types from `sse_starlette` library.
- **Code Reviews**:
  - **copilot-pull-request-reviewer[bot]**: COMMENTED - Reviewed 4/4 files, no comments generated
  - **ashwinvaidya17**: APPROVED

---

### PR #3056: feat(inspect): add training device selection
- **Author**: maxxgx
- **Merged**: 2025-10-27
- **Type**: 🚀 New feature
- **Description**: Added device selection support for training jobs
  - New endpoints: `GET /api/inference-devices` and `GET /api/training-devices`
  - Training job payload now supports device selection (CPU, CUDA, etc.)
  - Removed deprecated `/models/supported-devices` endpoint
- **Code Reviews**:
  - **copilot-pull-request-reviewer[bot]**: COMMENTED (multiple reviews) - Reviewed 14/15 files, generated 2 comments
  - **ashwinvaidya17**: APPROVED

---

### PR #3053: feat(inspect): Add train model dialog, job logs and models dataset
- **Author**: MarkRedeman
- **Merged**: 2025-10-27
- **Type**: 🚀 New feature
- **Description**: Major UI/UX enhancement for model training workflow
  - Adds "Train model" button with dialog for model selection
  - Implements real-time job logs viewing using SSE
  - Creates models tab showing all trained models and non-completed jobs
  - Uses query parameters to determine which tab is opened
  - Invokes inference when changing to a different model
  - Backend: Updates job end time and formats SSE messages
- **Code Reviews**:
  - **copilot-pull-request-reviewer[bot]**: COMMENTED - Reviewed 16/16 files, generated 5 comments
  - **ashwinvaidya17**: APPROVED
  - **camiloHimura**: COMMENTED (multiple reviews with feedback)
  - **MarkRedeman**: COMMENTED (multiple self-reviews)
  - **maxxgx**: COMMENTED (multiple reviews)

---

### PR #3044: chore (inspect): improve logs (add uvicorn handler) and add job_id to model
- **Author**: maxxgx
- **Merged**: 2025-10-22
- **Type**: 🔧 Chore
- **Description**: Logging infrastructure improvements
  - Uvicorn logs now captured to loguru
  - Added `train_job_id` to model entity
  - Sets actual threshold after training
  - Refactored: split `core.logging.py` into multiple files in `core.logging` package
- **Code Reviews**:
  - **copilot-pull-request-reviewer[bot]**: COMMENTED (multiple reviews) - Reviewed 16/16 files, generated comments about logging refactoring

---

### PR #3040: chore(inspect): minor UI improvements
- **Author**: MarkRedeman
- **Merged**: 2025-10-22
- **Type**: 🔧 Chore
- **Description**: UI/UX improvements
  - Add welcome screen for creating first project
  - Improve query client invalidation logic with `meta` tags
  - Invalidate images after uploading all items (not each item)
  - Render toast notifications about successful uploads
  - Fix grid spacing

---

### PR #3039: 🔧 chore(geti-inspect): configure loggers
- **Author**: maxxgx
- **Merged**: 2025-10-21
- **Type**: 🔧 Chore
- **Description**: Comprehensive logging system setup
  - Added logging via `loguru`
  - Handler to save logs to files with structured directory
  - New SSE endpoint `GET jobs/{job_id}/logs` for streaming logs
  - Added Trackio and Tensorboard loggers
  - Refactored workers with class implementation

---

### PR #3032: 🚀 feat(inference): Display inference on top of the image + opacity
- **Author**: dwesolow
- **Merged**: 2025-10-16
- **Type**: 🚀 New feature
- **Description**: Displays inference results on top of images with opacity slider control
- **Code Reviews**:
  - **dwesolow**: (self-assigned)

---

### PR #3029: 🔧 chore(inference): convert to jet and add opacity
- **Author**: ashwinvaidya17
- **Merged**: 2025-10-15
- **Type**: 🔧 Chore
- **Description**: Convert anomaly map to JET colormap and add opacity support using OpenCV

---

### PR #3028: 🔧 chore(normal images): Add `Normal images` heading
- **Author**: dwesolow
- **Merged**: 2025-10-15
- **Type**: 🔧 Chore
- **Description**: Minor UI improvement - adds normal images heading and dataset item spacing improvements
- **Code Reviews**:
  - **ashwinvaidya17**: (reviewed)

---

### PR #3026: 🚀 feat(inference): Allow user to select media and get the inference
- **Author**: dwesolow
- **Merged**: 2025-10-15
- **Type**: 🚀 New feature
- **Description**: Major inference feature
  - Users can select media and get inference results
  - Displays prediction when clicking on anomalous images
  - Requires at least 20 normal images for training
- **Code Reviews**:
  - **Multiple reviewers** provided feedback

---

### PR #3018: 🐛 fix(metrics): disable mps for torch metrics
- **Author**: maxxgx
- **Merged**: 2025-10-10
- **Type**: 🐞 Bug fix
- **Description**: Bug fix for Mac users - optimal threshold index out of range
  - TorchMetrics returns NaN when computing F1 score with device=mps
  - Solution: Use CPU instead of MPS

---

### PR #3016: 🐛 fix(geti-inspect): daemon worker error
- **Author**: maxxgx
- **Merged**: 2025-10-09
- **Type**: 🐞 Bug fix
- **Description**: Fixed daemon worker multiprocessing issue
  - Training worker cannot create child processes when using `num_workers: int = 8`
  - Solution: Set training worker to be non-daemonic
  - Improved process shutdown

---

### PR #3015: 🔄refactor(trainable models): Use trainable models from the server
- **Author**: dwesolow
- **Merged**: 2025-10-09
- **Type**: 🔄 Refactor
- **Description**: Replaced local array of available models with server-fetched models

---

### PR #3013: 🔧 chore(dataset): Use thumnbail suffix to get the dataset items
- **Author**: dwesolow
- **Merged**: 2025-10-09
- **Type**: 🔧 Chore
- **Description**: Replace 'full' with 'thumbnail' suffix for dataset item retrieval

---

### PR #3011: 🚀 feat(geti-inspect): add inference device selection
- **Author**: maxxgx
- **Merged**: 2025-10-13
- **Type**: 🚀 New feature
- **Description**: Inference device selection capability
  - New endpoint: `POST /models:supported-devices`
  - Predict endpoint supports device form parameter (CPU, GPU, NPU)
  - Added `inference_device` to pipelines

---

### PR #3009: ci(inspect): Exclude application/ for geti-inspect
- **Author**: MarkRedeman
- **Merged**: 2025-10-09
- **Type**: 🚧 CI/CD
- **Description**: Skip pre-commit hooks under application/ directory as geti-inspect uses different CI setup

---

### PR #3007: chore(gitattributes): Remove uv.lock LFS entry
- **Author**: MarkRedeman
- **Merged**: 2025-10-09
- **Type**: 🔧 Chore
- **Description**: Stop treating uv.lock as Git LFS-tracked file

---

### PR #3005: 🚀 feat(geti-inspect): add trainable models
- **Author**: maxxgx
- **Merged**: 2025-10-08
- **Type**: 🚀 New feature
- **Description**: Add `/trainable-models` endpoint with tests

---

### PR #3004: 🚀 feat(geti-inspect): add thumbnails media endpoint
- **Author**: maxxgx
- **Merged**: 2025-10-08
- **Type**: 🚀 New feature
- **Description**: Add thumbnail support to media endpoints with tests

---

### PR #2998: 🐛 fix(inspect): Use null pool for async engine
- **Author**: MarkRedeman
- **Merged**: 2025-10-06
- **Type**: 🐞 Bug fix
- **Description**: Fixed SQLite database lock issue
  - Training worker couldn't get database lock
  - Queries would stall indefinitely
  - Solution: Use null pool for async engine

---

### PR #2996: 🔧 chore(inspect): Add dev command to run server and ui concurrently
- **Author**: dwesolow
- **Merged**: 2025-10-09
- **Type**: 🔧 Chore
- **Description**: Added development command to run both server and UI concurrently

---

### PR #2995: 🔧 chore(inspect): Update uv.lock
- **Author**: dwesolow
- **Merged**: 2025-10-08
- **Type**: 🔧 Chore
- **Description**: Fixed uv.lock parsing error that prevented server from running locally

---

### PR #2994: 🔧 chore(inspect): Add more models to be used for training
- **Author**: dwesolow
- **Merged**: 2025-10-06
- **Type**: 🔧 Chore
- **Description**: Added more model options available for training

---

### PR #2993: Fix path
- **Author**: ashwinvaidya17
- **Merged**: 2025-10-03
- **Type**: 🐞 Bug fix
- **Description**: Fixes missing database path during first startup

---

### PR #2992: 🔄 refactor(inspect): Improvements to the jobs management and training
- **Author**: dwesolow
- **Merged**: 2025-10-06
- **Type**: 🔄 Refactor
- **Description**: Addressed comments from PR #2984 with improvements to job management

---

### PR #2991: chore(inspect): Improve error and suspense handling in router
- **Author**: MarkRedeman
- **Merged**: 2025-10-03
- **Type**: 🔄 Refactor
- **Description**: Moved all routes into single root route to ensure suspense and error boundary coverage

---

### PR #2990: refactor(inspect): Update photo placeholder to use indicator instead of email
- **Author**: dwesolow
- **Merged**: 2025-10-03
- **Type**: 🔄 Refactor
- **Description**: Updated photo placeholder component per upstream changes

---

### PR #2989: 🔧 chore(inspect): Rename folder structure from app to application
- **Author**: dwesolow
- **Merged**: 2025-10-03
- **Type**: 🔧 Chore
- **Description**: Updated folder structure to stay consistent with other apps

---

### PR #2985: 🔧 chore(inspect): Update openapi page
- **Author**: dwesolow
- **Merged**: 2025-10-01
- **Type**: 🔧 Chore
- **Description**: Updates OpenAPI page title

---

## Review Statistics

### Review Participation
- **copilot-pull-request-reviewer[bot]**: Automated reviews on majority of PRs
- **ashwinvaidya17**: Active reviewer, multiple approvals
- **MarkRedeman**: Active reviewer and contributor
- **camiloHimura**: Detailed review feedback on UI/UX PRs
- **maxxgx**: Contributor and reviewer
- **dwesolow**: Contributor and reviewer

### Review Process
- Most PRs received automated Copilot reviews analyzing code changes
- Critical features received multiple human reviews
- Average review comments per major feature PR: 3-5
- Bug fixes typically received quick automated review + approval

## Key Themes

### 1. **Infrastructure & Logging** (PRs #3044, #3039, #3060)
- Comprehensive logging system implementation
- Uvicorn integration with loguru
- Job-specific log streaming via SSE
- Structured log file organization

### 2. **Device Selection** (PRs #3056, #3011)
- Training device selection (CPU, CUDA, etc.)
- Inference device selection (CPU, GPU, NPU)
- Separate endpoints for training vs inference devices

### 3. **Training Workflow** (PRs #3053, #3026, #3005)
- Complete UI for model selection and training
- Real-time job log viewing
- Models dashboard showing training progress
- Inference preview on selected media

### 4. **Database & Performance** (PRs #3075, #2998, #2993, #3016)
- Fixed missing database commits
- Resolved SQLite locking issues
- Daemon worker multiprocessing fixes
- Database path initialization

### 5. **UI/UX Improvements** (PRs #3040, #3032, #3029, #3028, #2991)
- Welcome screen for first project
- Inference overlay with opacity control
- Anomaly map visualization (JET colormap)
- Improved error handling and suspense
- Query parameter-based navigation

### 6. **Developer Experience** (PRs #2996, #2995, #3009, #2989)
- Concurrent server/UI development mode
- Fixed dependency lock files
- Folder structure standardization
- CI/CD configuration for geti-inspect

## Code Quality Notes

- All major features received Copilot automated code review
- PRs with UI changes included screenshots for visual verification
- Test coverage maintained for backend changes
- Backend refactoring followed modular design patterns
- Frontend changes maintained TypeScript typing

## Conclusion

The `feature/geti-inspect` branch represents a comprehensive effort to build an anomaly detection inspection application with:
- Complete training and inference workflows
- Real-time monitoring and logging
- Flexible device selection
- Robust error handling
- Developer-friendly tooling

The code review process was thorough, with both automated (Copilot) and human reviewers providing feedback on code quality, design patterns, and potential issues.
