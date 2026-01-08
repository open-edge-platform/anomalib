# Geti Inspect Documentation

Geti Inspect is a full-stack application for fine-tuning and deploying anomaly detection models at the edge. It provides an intuitive web interface backed by a powerful FastAPI server that integrates with the [anomalib](https://github.com/open-edge-platform/anomalib) library.

## Overview

Geti Inspect enables users to:

- **Train anomaly detection models** using state-of-the-art algorithms from anomalib
- **Deploy models** for real-time inference using OpenVINO optimization
- **Stream video** from various sources (webcams, IP cameras, video files)
- **Visualize results** with real-time anomaly heatmaps and metrics
- **Export models** in multiple formats for edge deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Geti Inspect                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐              ┌─────────────────────────┐   │
│  │                 │   REST API   │                         │   │
│  │   Frontend      │◄────────────►│   Backend               │   │
│  │   (React)       │   WebRTC     │   (FastAPI)             │   │
│  │                 │              │                         │   │
│  └─────────────────┘              └───────────┬─────────────┘   │
│                                               │                 │
│                                               ▼                 │
│                                   ┌─────────────────────────┐   │
│                                   │       anomalib          │   │
│                                   │   (ML Engine)           │   │
│                                   └─────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Documentation Structure

| Document | Description |
|----------|-------------|
| [Getting Started](./getting-started.md) | Quick start guide for developers |
| [Backend](./backend.md) | Backend architecture and API documentation |
| [Frontend](./frontend.md) | Frontend architecture and development guide |
| [Configuration](./configuration.md) | Configuration options and environment variables |
| [API Reference](./api-reference.md) | REST API endpoints reference |
| [Deployment](./deployment.md) | Production deployment guide |

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 24.2.0+
- npm 11.3.0+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/open-edge-platform/anomalib.git
   cd anomalib/application
   ```

2. **Start the backend**
   ```bash
   cd backend
   uv sync
   ./run.sh
   ```

3. **Start the frontend** (in a new terminal)
   ```bash
   cd ui
   npm install
   npm start
   ```

4. **Open the application**
   
   Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

## Features

### Model Training

- Support for multiple anomaly detection architectures (PatchCore, PaDiM, EfficientAD, and more)
- Automated dataset management and preprocessing
- Training progress monitoring with real-time metrics
- Dataset snapshot management for reproducibility

### Inference Pipeline

- Real-time inference with OpenVINO optimization
- WebRTC-based video streaming for low-latency visualization
- Support for multiple video sources (webcams, IP cameras, video files)
- Configurable anomaly thresholds and visualization options

### Project Management

- Multi-project support with isolated datasets and models
- Media library for image management
- Export capabilities for deployment

## Tech Stack

### Backend

- **Framework**: FastAPI with async/await support
- **Database**: SQLite with SQLAlchemy ORM and Alembic migrations
- **ML Engine**: anomalib with OpenVINO inference
- **Video Processing**: OpenCV, aiortc (WebRTC)
- **Task Queue**: Background workers for training and inference

### Frontend

- **Framework**: React 19 with TypeScript
- **Build Tool**: Rsbuild
- **State Management**: TanStack Query (React Query)
- **UI Components**: react-aria-components
- **Styling**: SCSS modules
- **Routing**: React Router v6

## License

Copyright (C) 2025 Intel Corporation

SPDX-License-Identifier: Apache-2.0

