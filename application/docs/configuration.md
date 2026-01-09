# Configuration Guide

This guide covers all configuration options for the Geti Inspect application, including environment variables, settings files, and deployment configurations.

## Environment Variables

### Backend Configuration

The backend uses environment variables for configuration. Create a `.env` file in the `backend/` directory or set variables in your shell.

#### Application Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode with verbose logging | `false` |
| `ENVIRONMENT` | Environment type (`dev` or `prod`) | `dev` |
| `DATA_DIR` | Directory for storing application data | `data` |
| `LOG_DIR` | Directory for log files | `logs` |

#### Server Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `HTTP_SERVER_PORT` | Alternative port setting | `8000` |

#### Database Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_FILE` | SQLite database filename | `geti_inspect.db` |
| `DB_ECHO` | Enable SQL query logging | `false` |

#### Proxy Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `no_proxy` | Hosts to exclude from proxy | `localhost,127.0.0.1,::1` |

### Example `.env` File

```bash
# Backend .env file
DEBUG=true
ENVIRONMENT=dev
DATA_DIR=./data
LOG_DIR=./logs

HOST=0.0.0.0
PORT=8000

DATABASE_FILE=geti_inspect.db
DB_ECHO=false
```

### Frontend Configuration

Frontend configuration is handled through build-time environment variables.

| Variable | Description | Default |
|----------|-------------|---------|
| `RSBUILD_API_URL` | Backend API base URL | `http://localhost:8000` |

## Directory Structure

### Data Directory

The `DATA_DIR` contains all persistent application data:

```
data/
├── geti_inspect.db      # SQLite database
├── images/              # Uploaded media files
│   └── projects/
│       └── {project_id}/
│           ├── normal/
│           └── anomalous/
├── models/              # Trained model artifacts
│   └── projects/
│       └── {project_id}/
│           └── {model_id}/
│               ├── model.xml    # OpenVINO IR
│               ├── model.bin
│               └── metadata.json
└── snapshots/           # Dataset snapshots
    └── projects/
        └── {project_id}/
            └── {snapshot_id}/
```

### Log Directory

The `LOG_DIR` contains application logs:

```
logs/
├── app.log              # Main application log
├── training.log         # Training job logs
├── inference.log        # Inference pipeline logs
├── dispatching.log      # Output dispatch logs
├── stream_loader.log    # Video stream logs
├── jobs/                # Per-job log files
│   └── {job_id}.log
└── tensorboard/         # TensorBoard logs
    └── {job_id}/
```

## Logging Configuration

### Log Levels

The application uses [Loguru](https://github.com/Delgan/loguru) for logging. Set the log level via environment:

```bash
# Verbose logging
DEBUG=true ./run.sh

# Production logging (less verbose)
DEBUG=false ENVIRONMENT=prod ./run.sh
```

### Log Rotation

Logs are automatically rotated:

- **Size limit**: 10 MB per file
- **Retention**: 7 days
- **Compression**: Older logs are compressed

### Structured Logging

Logs include structured metadata:

```
2025-01-08 15:30:00.123 | INFO | services.training_service:start_training:45 - Starting training job | job_id=abc123 | project_id=xyz789
```

## Model Configuration

### Pre-trained Weights

Some models require pre-trained weights. Place them in the `pre_trained/` directory:

```
backend/
└── pre_trained/
    ├── dinov2_vitb14_reg4_pretrain.pth    # DINOv2 backbone
    └── efficientad_pretrained_weights/
        ├── pretrained_teacher_medium.pth
        └── pretrained_teacher_small.pth
```

### OpenVINO Cache

OpenVINO model compilation cache improves inference startup time:

```
backend/
└── openvino_cache/
    └── {model_hash}/
        └── compiled_model.blob
```

Set custom cache directory:

```bash
export OPENVINO_CACHE_DIR=/custom/path/openvino_cache
```

## Training Configuration

### Default Training Parameters

Each model architecture has default parameters that can be overridden via the API:

```json
{
  "patchcore": {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "coreset_sampling_ratio": 0.1
  },
  "efficientad": {
    "teacher_out_channels": 384,
    "model_size": "small"
  },
  "padim": {
    "backbone": "resnet18",
    "layers": ["layer1", "layer2", "layer3"]
  }
}
```

### Hardware Acceleration

#### OpenVINO Inference

By default, inference uses OpenVINO on CPU. To use GPU:

```python
# In model loading code
model.to("GPU")  # Intel GPU
model.to("AUTO")  # Automatic device selection
```

#### Training Device

Training automatically uses available GPU (CUDA/MPS) or falls back to CPU.

## Pipeline Configuration

### Source Configuration

#### Webcam

```json
{
  "type": "webcam",
  "config": {
    "device_index": 0,
    "resolution": [1920, 1080],
    "fps": 30
  }
}
```

#### IP Camera

```json
{
  "type": "ip_camera",
  "config": {
    "url": "rtsp://user:pass@192.168.1.100:554/stream",
    "timeout_ms": 5000,
    "reconnect_attempts": 3
  }
}
```

#### Video File

```json
{
  "type": "video_file",
  "config": {
    "path": "/path/to/video.mp4",
    "loop": true,
    "start_frame": 0
  }
}
```

### Sink Configuration

#### Webhook

```json
{
  "type": "webhook",
  "config": {
    "url": "https://api.example.com/alerts",
    "method": "POST",
    "headers": {
      "Authorization": "Bearer token",
      "Content-Type": "application/json"
    },
    "timeout_ms": 5000,
    "retry_attempts": 3
  }
}
```

#### MQTT

```json
{
  "type": "mqtt",
  "config": {
    "broker": "mqtt://broker.example.com:1883",
    "topic": "anomaly/alerts",
    "qos": 1,
    "username": "user",
    "password": "secret"
  }
}
```

Enable MQTT support by installing the optional dependency:

```bash
uv sync --extra mqtt
```

#### Filesystem

```json
{
  "type": "filesystem",
  "config": {
    "output_dir": "/data/anomalies",
    "save_images": true,
    "image_format": "png",
    "filename_template": "{timestamp}_{score:.2f}"
  }
}
```

### Pipeline Options

```json
{
  "threshold": 0.5,
  "fps": 30,
  "overlay_heatmap": true,
  "heatmap_opacity": 0.5,
  "save_anomalies": true,
  "min_anomaly_score": 0.3
}
```

## Database Configuration

### SQLite Settings

The default SQLite configuration uses WAL mode for better concurrency:

```python
# Automatic in settings.py
database_url = f"sqlite+aiosqlite:///./{data_dir}/{database_file}?journal_mode=WAL"
```

### Database Migrations

Run migrations when upgrading:

```bash
# Check current status
uv run src/cli.py check-db

# Run pending migrations
uv run src/cli.py migrate
```

### Backup

Backup the database file regularly:

```bash
# Simple file copy (stop server first for consistency)
cp data/geti_inspect.db data/geti_inspect.db.backup

# With timestamp
cp data/geti_inspect.db "data/geti_inspect_$(date +%Y%m%d_%H%M%S).db.backup"
```

## CORS Configuration

For development, CORS is configured to allow local origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict origins appropriately.

## Performance Tuning

### Inference Optimization

| Setting | Description | Recommendation |
|---------|-------------|----------------|
| Batch size | Frames processed together | 1 for real-time, higher for batch |
| Image size | Input resolution | Match training resolution |
| FPS | Frames per second | 15-30 for real-time |

### Memory Management

| Setting | Description | Recommendation |
|---------|-------------|----------------|
| Model caching | Keep models in memory | Enable for frequently used models |
| Image caching | Cache processed images | Disable for large datasets |
| Worker processes | Parallel workers | 1-2 per CPU core |

### Database Optimization

For large datasets:

```bash
# Vacuum database to reclaim space
sqlite3 data/geti_inspect.db "VACUUM;"

# Analyze for query optimization
sqlite3 data/geti_inspect.db "ANALYZE;"
```

## Security Considerations

### Production Checklist

- [ ] Disable debug mode (`DEBUG=false`)
- [ ] Set `ENVIRONMENT=prod`
- [ ] Configure proper CORS origins
- [ ] Use HTTPS with reverse proxy
- [ ] Implement authentication
- [ ] Set restrictive file permissions
- [ ] Enable request logging
- [ ] Configure rate limiting

### File Permissions

```bash
# Restrict data directory access
chmod 700 data/
chmod 600 data/geti_inspect.db

# Restrict log directory
chmod 700 logs/
```

## Troubleshooting Configuration Issues

### Common Problems

**Database connection errors:**
```bash
# Check database file permissions
ls -la data/geti_inspect.db

# Verify database is valid
sqlite3 data/geti_inspect.db ".tables"
```

**Port already in use:**
```bash
# Find process using port
lsof -i :8000

# Use alternative port
PORT=8001 ./run.sh
```

**Missing environment variables:**
```bash
# Debug environment
env | grep -E "(DEBUG|PORT|DATA_DIR)"
```

**Model loading failures:**
```bash
# Check pre-trained weights
ls -la pre_trained/

# Verify model files
ls -la data/models/projects/
```

