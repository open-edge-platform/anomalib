# API Reference

The Geti Inspect API is a RESTful API that provides endpoints for managing anomaly detection projects, models, and inference pipelines.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the API does not require authentication for local development. For production deployments, configure appropriate authentication middleware.

## Response Format

All responses are JSON formatted. Successful responses return data directly, while errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Endpoints

---

## Projects

### List All Projects

```http
GET /projects
```

**Response** `200 OK`
```json
[
  {
    "id": "6ee0c080-c7d9-4438-a7d2-067fd395eecf",
    "name": "Manufacturing QC",
    "created_at": "2025-01-08T10:30:00Z",
    "updated_at": "2025-01-08T14:20:00Z"
  }
]
```

### Create Project

```http
POST /projects
```

**Request Body**
```json
{
  "name": "New Project"
}
```

**Response** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "New Project",
  "created_at": "2025-01-08T15:00:00Z",
  "updated_at": "2025-01-08T15:00:00Z"
}
```

### Get Project

```http
GET /projects/{project_id}
```

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `project_id` | UUID | Project identifier |

**Response** `200 OK`
```json
{
  "id": "6ee0c080-c7d9-4438-a7d2-067fd395eecf",
  "name": "Manufacturing QC",
  "created_at": "2025-01-08T10:30:00Z",
  "updated_at": "2025-01-08T14:20:00Z"
}
```

### Update Project

```http
PATCH /projects/{project_id}
```

**Request Body**
```json
{
  "name": "Updated Project Name"
}
```

**Response** `200 OK`

### Delete Project

```http
DELETE /projects/{project_id}
```

**Response** `204 No Content`

---

## Media

### List Project Media

```http
GET /projects/{project_id}/media
```

**Query Parameters**
| Name | Type | Description |
|------|------|-------------|
| `label` | string | Filter by label (`normal`, `anomalous`) |
| `limit` | integer | Maximum items to return |
| `offset` | integer | Number of items to skip |

**Response** `200 OK`
```json
{
  "items": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "filename": "sample_001.png",
      "label": "normal",
      "thumbnail_url": "/media/thumbnails/a1b2c3d4.png",
      "created_at": "2025-01-08T10:00:00Z"
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

### Upload Media

```http
POST /projects/{project_id}/media
```

**Request** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| `files` | file[] | Image files to upload |
| `label` | string | Label for all images (`normal` or `anomalous`) |

**Response** `201 Created`
```json
{
  "uploaded": 5,
  "failed": 0,
  "items": [...]
}
```

### Get Media Item

```http
GET /media/{media_id}
```

**Response** `200 OK` with image binary

### Delete Media

```http
DELETE /media/{media_id}
```

**Response** `204 No Content`

---

## Models

### List Project Models

```http
GET /projects/{project_id}/models
```

**Response** `200 OK`
```json
[
  {
    "id": "977eeb18-eaac-449d-bc80-e340fbe052ad",
    "name": "PatchCore Model v1",
    "format": "openvino",
    "architecture": "patchcore",
    "status": "ready",
    "metrics": {
      "image_auroc": 0.95,
      "pixel_auroc": 0.92
    },
    "created_at": "2025-01-08T12:00:00Z"
  }
]
```

### Get Model

```http
GET /models/{model_id}
```

**Response** `200 OK`

### Delete Model

```http
DELETE /models/{model_id}
```

**Response** `204 No Content`

### Export Model

```http
GET /models/{model_id}/export
```

**Query Parameters**
| Name | Type | Description |
|------|------|-------------|
| `format` | string | Export format (`openvino`, `onnx`, `pytorch`) |

**Response** `200 OK` with model archive

---

## Training

### List Trainable Models

```http
GET /trainable-models
```

**Response** `200 OK`
```json
[
  {
    "name": "patchcore",
    "display_name": "PatchCore",
    "description": "Memory bank-based anomaly detection",
    "parameters": {
      "backbone": {
        "type": "string",
        "default": "wide_resnet50_2",
        "options": ["resnet18", "wide_resnet50_2"]
      },
      "layers": {
        "type": "array",
        "default": ["layer2", "layer3"]
      }
    }
  },
  {
    "name": "efficientad",
    "display_name": "EfficientAD",
    "description": "Efficient student-teacher network"
  }
]
```

### Start Training

```http
POST /projects/{project_id}/train
```

**Request Body**
```json
{
  "model_name": "Model v1",
  "architecture": "patchcore",
  "parameters": {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"]
  },
  "snapshot_id": "optional-snapshot-uuid"
}
```

**Response** `202 Accepted`
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending"
}
```

---

## Jobs

### List Jobs

```http
GET /projects/{project_id}/jobs
```

**Query Parameters**
| Name | Type | Description |
|------|------|-------------|
| `type` | string | Filter by job type (`training`, `export`) |
| `status` | string | Filter by status (`pending`, `running`, `completed`, `failed`) |

**Response** `200 OK`
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "type": "training",
    "status": "running",
    "progress": 0.45,
    "created_at": "2025-01-08T14:00:00Z",
    "started_at": "2025-01-08T14:01:00Z"
  }
]
```

### Get Job Status

```http
GET /jobs/{job_id}
```

**Response** `200 OK`
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "type": "training",
  "status": "running",
  "progress": 0.65,
  "metrics": {
    "epoch": 15,
    "loss": 0.023
  },
  "created_at": "2025-01-08T14:00:00Z",
  "started_at": "2025-01-08T14:01:00Z"
}
```

### Cancel Job

```http
DELETE /jobs/{job_id}
```

**Response** `204 No Content`

---

## Pipelines

### List Pipelines

```http
GET /projects/{project_id}/pipelines
```

**Response** `200 OK`
```json
[
  {
    "id": "abc12345-6789-0def-ghij-klmnopqrstuv",
    "name": "Production Line Monitor",
    "model_id": "977eeb18-eaac-449d-bc80-e340fbe052ad",
    "source_id": "src-uuid",
    "sink_id": "sink-uuid",
    "config": {
      "threshold": 0.5,
      "fps": 30
    },
    "created_at": "2025-01-08T15:00:00Z"
  }
]
```

### Create Pipeline

```http
POST /projects/{project_id}/pipelines
```

**Request Body**
```json
{
  "name": "New Pipeline",
  "model_id": "977eeb18-eaac-449d-bc80-e340fbe052ad",
  "source_id": "source-uuid",
  "sink_id": "sink-uuid",
  "config": {
    "threshold": 0.5,
    "fps": 30
  }
}
```

**Response** `201 Created`

### Update Pipeline

```http
PATCH /pipelines/{pipeline_id}
```

### Delete Pipeline

```http
DELETE /pipelines/{pipeline_id}
```

**Response** `204 No Content`

---

## Active Pipelines

### List Active Pipelines

```http
GET /active-pipelines
```

**Response** `200 OK`
```json
[
  {
    "id": "active-123",
    "pipeline_id": "abc12345-6789-0def-ghij-klmnopqrstuv",
    "status": "running",
    "started_at": "2025-01-08T16:00:00Z",
    "metrics": {
      "frames_processed": 1500,
      "anomalies_detected": 3,
      "avg_inference_time_ms": 25
    }
  }
]
```

### Start Pipeline

```http
POST /active-pipelines
```

**Request Body**
```json
{
  "pipeline_id": "abc12345-6789-0def-ghij-klmnopqrstuv"
}
```

**Response** `201 Created`
```json
{
  "id": "active-123",
  "pipeline_id": "abc12345-6789-0def-ghij-klmnopqrstuv",
  "status": "starting"
}
```

### Stop Pipeline

```http
DELETE /active-pipelines/{active_pipeline_id}
```

**Response** `204 No Content`

---

## Sources

### List Sources

```http
GET /projects/{project_id}/sources
```

**Response** `200 OK`
```json
[
  {
    "id": "src-webcam-1",
    "type": "webcam",
    "name": "Built-in Camera",
    "config": {
      "device_index": 0
    }
  },
  {
    "id": "src-ip-1",
    "type": "ip_camera",
    "name": "Factory Camera",
    "config": {
      "url": "rtsp://192.168.1.100:554/stream"
    }
  }
]
```

### Create Source

```http
POST /projects/{project_id}/sources
```

**Request Body (Webcam)**
```json
{
  "type": "webcam",
  "name": "USB Camera",
  "config": {
    "device_index": 1
  }
}
```

**Request Body (IP Camera)**
```json
{
  "type": "ip_camera",
  "name": "RTSP Stream",
  "config": {
    "url": "rtsp://user:pass@192.168.1.100:554/stream"
  }
}
```

**Request Body (Video File)**
```json
{
  "type": "video_file",
  "name": "Test Video",
  "config": {
    "path": "/path/to/video.mp4",
    "loop": true
  }
}
```

### Delete Source

```http
DELETE /sources/{source_id}
```

---

## Sinks

### List Sinks

```http
GET /projects/{project_id}/sinks
```

**Response** `200 OK`
```json
[
  {
    "id": "sink-webhook-1",
    "type": "webhook",
    "name": "Alert Webhook",
    "config": {
      "url": "https://alerts.example.com/anomaly",
      "method": "POST"
    }
  }
]
```

### Create Sink

```http
POST /projects/{project_id}/sinks
```

**Request Body (Webhook)**
```json
{
  "type": "webhook",
  "name": "Alert System",
  "config": {
    "url": "https://alerts.example.com/anomaly",
    "method": "POST",
    "headers": {
      "Authorization": "Bearer token"
    }
  }
}
```

**Request Body (MQTT)**
```json
{
  "type": "mqtt",
  "name": "MQTT Broker",
  "config": {
    "broker": "mqtt://broker.example.com:1883",
    "topic": "anomaly/alerts",
    "username": "user",
    "password": "pass"
  }
}
```

**Request Body (Filesystem)**
```json
{
  "type": "filesystem",
  "name": "Local Storage",
  "config": {
    "output_dir": "/data/anomalies",
    "save_images": true
  }
}
```

### Delete Sink

```http
DELETE /sinks/{sink_id}
```

---

## Devices

### List Available Devices

```http
GET /devices
```

**Response** `200 OK`
```json
{
  "cameras": [
    {
      "index": 0,
      "name": "FaceTime HD Camera",
      "type": "webcam"
    },
    {
      "index": 1,
      "name": "USB Camera",
      "type": "webcam"
    }
  ],
  "compute": [
    {
      "name": "CPU",
      "type": "cpu"
    },
    {
      "name": "Intel GPU",
      "type": "gpu"
    }
  ]
}
```

---

## Captures

### Capture Frame

```http
POST /active-pipelines/{active_pipeline_id}/capture
```

**Response** `201 Created`
```json
{
  "id": "capture-uuid",
  "timestamp": "2025-01-08T16:30:00Z",
  "anomaly_score": 0.85,
  "image_url": "/captures/capture-uuid.png"
}
```

---

## Snapshots

### List Snapshots

```http
GET /projects/{project_id}/snapshots
```

**Response** `200 OK`
```json
[
  {
    "id": "snapshot-uuid",
    "name": "Training Set v1",
    "media_count": 500,
    "created_at": "2025-01-08T10:00:00Z"
  }
]
```

### Create Snapshot

```http
POST /projects/{project_id}/snapshots
```

**Request Body**
```json
{
  "name": "Training Set v2"
}
```

**Response** `201 Created`

---

## WebRTC

### Create Offer

```http
POST /webrtc/offer
```

**Request Body**
```json
{
  "pipeline_id": "active-pipeline-id",
  "sdp": "v=0\r\no=- ..."
}
```

**Response** `200 OK`
```json
{
  "sdp": "v=0\r\no=- ...",
  "type": "answer"
}
```

### Send ICE Candidate

```http
POST /webrtc/candidate
```

**Request Body**
```json
{
  "pipeline_id": "active-pipeline-id",
  "candidate": "candidate:..."
}
```

**Response** `200 OK`

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| `400` | Bad Request - Invalid input |
| `404` | Not Found - Resource doesn't exist |
| `409` | Conflict - Resource already exists or operation not allowed |
| `422` | Unprocessable Entity - Validation error |
| `500` | Internal Server Error |

## Rate Limiting

The API currently does not implement rate limiting for local development. Production deployments should configure appropriate rate limits.

## OpenAPI Specification

The complete OpenAPI specification is available at:

```
GET /api/openapi.json
```

Import this into tools like Postman or Swagger UI for interactive API exploration.

