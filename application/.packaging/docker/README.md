# Docker Deployment for Geti Inspect


## CPU build

```bash
docker compose build
```

## GPU build

```bash
docker compose build --build-arg AI_DEVICE=cuda
```