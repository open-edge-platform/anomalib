# Docker distribution for Geti Inspect

## To create CPU build

### Build the container

```bash
cd application/docker
docker compose build
```

### Start the container

```bash
cd application/docker
docker compose up
```

## To create XPU build

### Build the container

```bash
cd application/docker
AI_DEVICE=xpu docker compose build
```

### Start the container

```bash
cd application/docker
AI_DEVICE=xpu docker compose up
```

## To create CUDA build

### Build the container

```bash
cd application/docker
docker compose -f docker-compose.cuda.yaml build
```
### Start the container

```bash
cd application/docker
docker compose -f docker-compose.cuda.yaml up
```