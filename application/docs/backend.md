# Backend Documentation

The Geti Inspect backend is a FastAPI application that provides REST APIs for managing anomaly detection projects, training models, and running inference pipelines.

## Architecture Overview

```
backend/
├── src/
│   ├── api/                    # REST API layer
│   │   └── endpoints/          # Route handlers
│   ├── core/                   # Application core
│   │   └── lifecycle.py        # App lifecycle management
│   ├── db/                     # Database layer
│   │   ├── engine.py           # SQLAlchemy engine
│   │   └── schema.py           # ORM models
│   ├── alembic/                # Database migrations
│   ├── entities/               # Domain entities
│   ├── pydantic_models/        # Request/Response schemas
│   ├── repositories/           # Data access layer (DAL)
│   ├── services/               # Business logic layer
│   ├── utils/                  # Utility functions
│   ├── webrtc/                 # WebRTC streaming
│   ├── workers/                # Background workers
│   ├── main.py                 # Application entry point
│   ├── cli.py                  # CLI commands
│   └── settings.py             # Configuration
├── tests/                      # Test suite
├── pyproject.toml              # Dependencies
└── run.sh                      # Startup script
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI 0.116+ |
| Database | SQLite with SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| ML Engine | anomalib with OpenVINO |
| Video Streaming | aiortc (WebRTC), OpenCV |
| Task Management | Background workers |
| Validation | Pydantic v2 |
| CLI | Click |

## Core Components

### API Endpoints

The API is organized into logical resource groups:

| Router | Path | Description |
|--------|------|-------------|
| `project_router` | `/api/projects` | Project CRUD operations |
| `model_router` | `/api/models` | Model management |
| `media_router` | `/api/media` | Image/media management |
| `pipeline_router` | `/api/pipelines` | Pipeline configuration |
| `active_pipeline_router` | `/api/active-pipelines` | Running pipeline control |
| `job_router` | `/api/jobs` | Background job status |
| `source_router` | `/api/sources` | Video input sources |
| `sink_router` | `/api/sinks` | Output destinations |
| `trainable_model_router` | `/api/trainable-models` | Available model architectures |
| `device_router` | `/api/devices` | Hardware device info |
| `capture_router` | `/api/captures` | Image capture |
| `snapshot_router` | `/api/snapshots` | Dataset snapshots |
| `webrtc_router` | `/api/webrtc` | WebRTC signaling |

### Services Layer

Services contain the business logic:

| Service | Responsibility |
|---------|----------------|
| `ProjectService` | Project lifecycle management |
| `ModelService` | Model training, loading, inference |
| `TrainingService` | Training job orchestration |
| `MediaService` | Image storage and retrieval |
| `PipelineService` | Pipeline configuration |
| `ActivePipelineService` | Running pipeline management |
| `VideoStreamService` | Video capture and streaming |
| `DispatchService` | Output dispatching (webhook, MQTT, filesystem) |
| `MetricsService` | Performance metrics collection |
| `DatasetSnapshotService` | Dataset version management |

### Repository Pattern

Data access is abstracted through repositories:

```python
# Example: ProjectRepository
class ProjectRepository:
    async def get_all(self) -> list[ProjectDB]:
        """Get all projects."""
        
    async def get_by_id(self, project_id: UUID) -> ProjectDB | None:
        """Get a project by ID."""
        
    async def create(self, project: ProjectCreate) -> ProjectDB:
        """Create a new project."""
        
    async def update(self, project_id: UUID, data: ProjectUpdate) -> ProjectDB:
        """Update an existing project."""
        
    async def delete(self, project_id: UUID) -> None:
        """Delete a project."""
```

### Background Workers

Long-running tasks are handled by background workers:

| Worker | Purpose |
|--------|---------|
| `TrainingWorker` | Executes model training jobs |
| `InferenceWorker` | Runs inference on video streams |
| `DispatchWorker` | Sends results to configured sinks |

## Database Schema

### Core Tables

```sql
-- Projects
projects (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)

-- Models
models (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    format VARCHAR NOT NULL,  -- 'openvino', 'pytorch', etc.
    project_id UUID REFERENCES projects(id),
    job_id UUID REFERENCES jobs(id),
    created_at TIMESTAMP
)

-- Media
media (
    id UUID PRIMARY KEY,
    filename VARCHAR NOT NULL,
    label VARCHAR,  -- 'normal', 'anomalous'
    project_id UUID REFERENCES projects(id),
    created_at TIMESTAMP
)

-- Jobs
jobs (
    id UUID PRIMARY KEY,
    type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    project_id UUID REFERENCES projects(id),
    progress FLOAT,
    error_message TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
)

-- Pipelines
pipelines (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    project_id UUID REFERENCES projects(id),
    model_id UUID REFERENCES models(id),
    source_id UUID REFERENCES sources(id),
    sink_id UUID REFERENCES sinks(id),
    config JSON
)

-- Sources (video inputs)
sources (
    id UUID PRIMARY KEY,
    type VARCHAR NOT NULL,  -- 'webcam', 'ip_camera', 'video_file'
    config JSON NOT NULL,
    project_id UUID REFERENCES projects(id)
)

-- Sinks (output destinations)
sinks (
    id UUID PRIMARY KEY,
    type VARCHAR NOT NULL,  -- 'webhook', 'mqtt', 'filesystem'
    config JSON NOT NULL,
    project_id UUID REFERENCES projects(id)
)
```

### Migrations

Database migrations are managed with Alembic:

```bash
# Create a new migration
uv run src/cli.py create-db-revision -m "add new column"

# Run migrations
uv run src/cli.py migrate

# Check migration status
uv run src/cli.py check-db
```

## API Design Patterns

### Request/Response Models

All endpoints use Pydantic models for validation:

```python
# Request model
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None

# Response model
class ProjectResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
```

### Error Handling

Custom exceptions are handled by global exception handlers:

```python
# Custom exception
class ProjectNotFoundError(Exception):
    def __init__(self, project_id: UUID):
        self.project_id = project_id

# Exception handler (registered in exception_handlers.py)
@app.exception_handler(ProjectNotFoundError)
async def project_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Project {exc.project_id} not found"}
    )
```

### Dependency Injection

FastAPI's dependency injection is used for services:

```python
from fastapi import Depends

async def get_project_service(
    db: AsyncSession = Depends(get_db_session)
) -> ProjectService:
    repository = ProjectRepository(db)
    return ProjectService(repository)

@router.get("/projects/{project_id}")
async def get_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
):
    return await service.get_project(project_id)
```

## Training Pipeline

### Training Flow

```
1. User creates training job
   └── POST /api/projects/{id}/train

2. Job is queued
   └── TrainingService creates JobDB with status='pending'

3. TrainingWorker picks up job
   └── Status changes to 'running'

4. Training executes
   ├── Load images from MediaService
   ├── Create anomalib datamodule
   ├── Initialize model (PatchCore, EfficientAD, etc.)
   ├── Run training with progress callbacks
   └── Export model to OpenVINO format

5. Training completes
   ├── Model artifacts saved to filesystem
   ├── ModelDB record created
   └── Job status set to 'completed'
```

### Supported Models

The backend supports all anomalib model architectures:

- **PatchCore** - Memory bank-based anomaly detection
- **PaDiM** - Probabilistic modeling of normal features
- **EfficientAD** - Efficient student-teacher network
- **STFPM** - Student-Teacher Feature Pyramid Matching
- **Reverse Distillation** - Knowledge distillation approach
- And more...

## Inference Pipeline

### Real-time Inference

```
1. User starts pipeline
   └── POST /api/active-pipelines

2. Pipeline initializes
   ├── Load OpenVINO model
   ├── Initialize video source
   └── Set up output sink

3. Inference loop
   ├── Capture frame from source
   ├── Run anomaly detection
   ├── Generate heatmap overlay
   ├── Stream via WebRTC
   └── Dispatch to configured sinks

4. Pipeline stops
   └── DELETE /api/active-pipelines/{id}
```

### Video Sources

| Type | Configuration |
|------|---------------|
| Webcam | Device index or path |
| IP Camera | RTSP/HTTP URL |
| Video File | Local file path |

### Output Sinks

| Type | Configuration |
|------|---------------|
| Webhook | URL, headers, method |
| MQTT | Broker, topic, credentials |
| Filesystem | Output directory, format |

## WebRTC Streaming

The backend uses aiortc for low-latency video streaming:

```python
# WebRTC flow
1. Client requests offer
   └── POST /api/webrtc/offer

2. Server creates peer connection
   └── Generates SDP answer

3. ICE candidates exchanged
   └── POST /api/webrtc/candidate

4. Video track established
   └── Frames streamed at configured FPS

5. Connection closed
   └── DELETE /api/webrtc/connection
```

## Testing

### Running Tests

```bash
cd backend

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/unit/services/test_model_service.py

# Run with verbose output
uv run pytest -v
```

### Test Structure

```
tests/
├── unit/
│   ├── api/           # Endpoint tests
│   ├── services/      # Service tests
│   └── repositories/  # Repository tests
└── conftest.py        # Shared fixtures
```

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_repository():
    return AsyncMock(spec=ProjectRepository)

@pytest.fixture
def service(mock_repository):
    return ProjectService(mock_repository)

async def test_get_project(service, mock_repository):
    # Arrange
    project = ProjectDB(id=uuid4(), name="Test")
    mock_repository.get_by_id.return_value = project
    
    # Act
    result = await service.get_project(project.id)
    
    # Assert
    assert result.name == "Test"
    mock_repository.get_by_id.assert_called_once_with(project.id)
```

## Logging

Logging is configured using [Loguru](https://github.com/Delgan/loguru):

```python
from loguru import logger

logger.info("Starting training job", job_id=job_id)
logger.error("Training failed", error=str(e))
```

Log files are stored in the `logs/` directory:

| Log File | Content |
|----------|---------|
| `app.log` | Main application logs |
| `training.log` | Training job logs |
| `inference.log` | Inference pipeline logs |
| `dispatching.log` | Output dispatch logs |

## Development Tips

### Hot Reload

The backend supports hot reload during development:

```bash
uv run uvicorn main:app --reload
```

### Debug Mode

Enable debug mode for detailed error responses:

```bash
DEBUG=true ./run.sh
```

### Database Inspection

Use SQLite CLI to inspect the database:

```bash
sqlite3 data/geti_inspect.db
.tables
.schema projects
SELECT * FROM projects;
```

### OpenAPI Specification

Generate the OpenAPI spec for API documentation:

```bash
uv run src/cli.py gen-api --target-path docs/openapi.json
```

