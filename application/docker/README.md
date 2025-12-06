# Docker distribution for Geti Inspect

## To create CPU build

```bash
cd application/.packaging/docker
docker compose up
```

## To create XPU build

> [!NOTE]
> You need to first set the `RENDER_GROUP_ID` environment variable to match your host system's render group ID.

```bash
cd application/.packaging/docker
RENDER_GROUP_ID=$(getent group render | cut -d: -f3) AI_DEVICE=xpu docker compose up
```

## To create CUDA build

> [!NOTE]
> You need to uncomment the `deploy:` section in the `docker-compose.yml` file to enable GPU support.

```bash
cd application/.packaging/docker
AI_DEVICE=cuda docker compose up
```