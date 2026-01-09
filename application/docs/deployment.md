# Deployment Guide

This guide covers deploying the Geti Inspect application in production environments.

## Deployment Options

### 1. Standalone Deployment

Run the application directly on a server or edge device.

### 2. Docker Deployment

Containerized deployment for consistent environments.

### 3. Tauri Desktop Application

Cross-platform desktop application using Tauri.

---

## Standalone Deployment

### Prerequisites

- Python 3.13+
- Node.js 24.2.0+ (for building frontend)
- [uv](https://docs.astral.sh/uv/) package manager

### Backend Deployment

1. **Clone and setup:**
   ```bash
   git clone https://github.com/open-edge-platform/anomalib.git
   cd anomalib/application/backend
   uv sync --no-dev  # Install production dependencies only
   ```

2. **Configure environment:**
   ```bash
   cat > .env << EOF
   DEBUG=false
   ENVIRONMENT=prod
   HOST=0.0.0.0
   PORT=8000
   DATA_DIR=/var/lib/geti-inspect/data
   LOG_DIR=/var/log/geti-inspect
   EOF
   ```

3. **Initialize database:**
   ```bash
   mkdir -p /var/lib/geti-inspect/data
   mkdir -p /var/log/geti-inspect
   uv run src/cli.py init-db
   ```

4. **Start the server:**
   ```bash
   uv run src/main.py
   ```

### Frontend Deployment

1. **Build the frontend:**
   ```bash
   cd ../ui
   npm ci  # Clean install
   npm run build
   ```

2. **Serve static files:**
   
   The built files are in `dist/`. Serve them with a web server:
   
   ```bash
   # Using Python's built-in server (development only)
   cd dist && python -m http.server 3000
   
   # Or copy to your web server root
   cp -r dist/* /var/www/geti-inspect/
   ```

### Using Nginx as Reverse Proxy

```nginx
# /etc/nginx/sites-available/geti-inspect

upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name inspect.example.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name inspect.example.com;
    
    ssl_certificate /etc/ssl/certs/inspect.example.com.crt;
    ssl_certificate_key /etc/ssl/private/inspect.example.com.key;
    
    # Frontend static files
    location / {
        root /var/www/geti-inspect;
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy
    location /api/ {
        proxy_pass http://backend/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for WebRTC signaling
        proxy_read_timeout 86400;
    }
    
    # Media files (optional, for direct serving)
    location /media/ {
        alias /var/lib/geti-inspect/data/images/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Systemd Service

Create a systemd service for automatic startup:

```ini
# /etc/systemd/system/geti-inspect.service

[Unit]
Description=Geti Inspect Anomaly Detection Server
After=network.target

[Service]
Type=simple
User=geti-inspect
Group=geti-inspect
WorkingDirectory=/opt/geti-inspect/backend
Environment="PATH=/opt/geti-inspect/backend/.venv/bin"
ExecStart=/usr/local/bin/uv run src/main.py
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/geti-inspect /var/log/geti-inspect

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable geti-inspect
sudo systemctl start geti-inspect
```

---

## Docker Deployment

### Dockerfile (Backend)

```dockerfile
# backend/Dockerfile

FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Copy application code
COPY src/ ./src/

# Create data directories
RUN mkdir -p data logs

# Set environment
ENV PYTHONPATH=/app/src
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "src/main.py"]
```

### Dockerfile (Frontend)

```dockerfile
# ui/Dockerfile

FROM node:24-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Build
RUN npm run build

# Production image
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - geti-data:/app/data
      - geti-logs:/app/logs
    environment:
      - DEBUG=false
      - ENVIRONMENT=prod
    restart: unless-stopped

  frontend:
    build:
      context: ./ui
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  geti-data:
  geti-logs:
```

### Running with Docker Compose

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose down -v
```

---

## Tauri Desktop Application

The application can be packaged as a desktop application using Tauri.

### Prerequisites

- Rust (latest stable)
- Platform-specific build tools (see [Tauri prerequisites](https://tauri.app/v1/guides/getting-started/prerequisites))

### Building the Desktop App

```bash
cd application/binary/tauri

# Install dependencies
npm install

# Development mode
npm run tauri dev

# Build for production
npm run tauri build
```

### Platform Builds

| Platform | Output |
|----------|--------|
| Windows | `.msi`, `.exe` installers |
| macOS | `.dmg`, `.app` bundle |
| Linux | `.deb`, `.AppImage` |

---

## Edge Deployment

### Intel NUC / Edge Devices

For deployment on Intel edge devices:

1. **Install Intel OpenVINO Runtime:**
   ```bash
   # Add Intel repository
   wget https://apt.repos.intel.com/openvino/2025/GPG-PUB-KEY-INTEL-OPENVINO-2025
   sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2025
   echo "deb https://apt.repos.intel.com/openvino/2025 focal main" | \
       sudo tee /etc/apt/sources.list.d/intel-openvino-2025.list
   
   sudo apt update
   sudo apt install openvino-2025
   ```

2. **Enable GPU acceleration (Intel integrated GPU):**
   ```bash
   sudo usermod -aG render $USER
   sudo usermod -aG video $USER
   ```

3. **Deploy application:**
   Follow the standalone deployment instructions above.

### Resource Optimization

For resource-constrained devices:

```bash
# Limit memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use smaller models
# Configure training to use EfficientAD-small or PaDiM with resnet18

# Reduce inference resolution
# Set lower FPS in pipeline config
```

---

## Monitoring

### Health Checks

The API provides a health endpoint:

```bash
curl http://localhost:8000/health
```

### Logging

Configure log aggregation:

```yaml
# docker-compose with log driver
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Metrics

For production monitoring, consider adding:

- Prometheus metrics endpoint
- Grafana dashboards
- Alert rules for anomalies

---

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/geti-inspect"
DATA_DIR="/var/lib/geti-inspect/data"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup database
sqlite3 "$DATA_DIR/geti_inspect.db" ".backup $BACKUP_DIR/geti_inspect_$DATE.db"

# Backup media files
tar -czf "$BACKUP_DIR/media_$DATE.tar.gz" -C "$DATA_DIR" images/

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" -C "$DATA_DIR" models/

# Cleanup old backups (keep 7 days)
find "$BACKUP_DIR" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR="/backup/geti-inspect"
DATA_DIR="/var/lib/geti-inspect/data"
BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: restore.sh YYYYMMDD_HHMMSS"
    exit 1
fi

# Stop service
sudo systemctl stop geti-inspect

# Restore database
cp "$BACKUP_DIR/geti_inspect_$BACKUP_DATE.db" "$DATA_DIR/geti_inspect.db"

# Restore media
tar -xzf "$BACKUP_DIR/media_$BACKUP_DATE.tar.gz" -C "$DATA_DIR"

# Restore models
tar -xzf "$BACKUP_DIR/models_$BACKUP_DATE.tar.gz" -C "$DATA_DIR"

# Start service
sudo systemctl start geti-inspect

echo "Restore completed from: $BACKUP_DATE"
```

---

## Scaling

### Horizontal Scaling

For high-traffic deployments:

```yaml
# docker-compose-scaled.yml
version: '3.8'

services:
  backend:
    build: ./backend
    deploy:
      replicas: 3
    environment:
      - DATABASE_URL=postgresql://...  # Use PostgreSQL for multi-instance

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
```

### Database Scaling

For large deployments, consider migrating from SQLite to PostgreSQL:

```python
# settings.py modification
database_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:pass@localhost/geti_inspect"
)
```

---

## Security Hardening

### Production Checklist

- [ ] Enable HTTPS with valid SSL certificate
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Implement authentication
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Vulnerability scanning

### Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 22/tcp   # SSH (restrict to specific IPs)
sudo ufw enable
```

### SSL/TLS Configuration

Use strong TLS settings:

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
ssl_prefer_server_ciphers off;
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;
```

---

## Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check logs
journalctl -u geti-inspect -n 50

# Verify permissions
ls -la /var/lib/geti-inspect/
```

**Database locked:**
```bash
# Check for stuck processes
fuser /var/lib/geti-inspect/data/geti_inspect.db

# Force unlock (use with caution)
sqlite3 /var/lib/geti-inspect/data/geti_inspect.db "PRAGMA busy_timeout = 5000;"
```

**Out of memory:**
```bash
# Monitor memory
htop

# Reduce worker processes
# Limit model cache size
```

**WebRTC connection failures:**
```bash
# Check TURN/STUN server connectivity
# Verify firewall allows UDP traffic
# Check WebSocket proxy configuration
```

