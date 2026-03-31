# Docker Distribution for Anomalib Studio

This guide covers how to build and run Anomalib Studio using Docker. Three build targets are supported depending on your hardware: CPU, Intel XPU, and NVIDIA CUDA GPU.

---

## Prerequisites

Ensure the following are installed before proceeding:

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) — Docker Compose v2 is recommended (`docker compose`, not `docker-compose`)
- For XPU builds: Intel GPU drivers and Intel oneAPI runtime must be installed on the host
- For CUDA builds: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be configured so Docker can access the GPU

---

## Setup

Start by cloning the repository and navigating to the Docker directory where the Compose files are located:

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib/application/docker
```

All subsequent commands should be run from this directory.

---

## Building and Starting the Container

Choose the build that matches your hardware. The `AI_DEVICE` environment variable tells the build which inference backend to target.

> [!NOTE]
> The `HOST` defaults to `0.0.0.0` and thus can be omitted.

### CPU Build

Use this if you don't have a GPU or want a hardware-agnostic setup. This is the default and works on any machine with Docker installed.

```bash
docker compose build
HOST=0.0.0.0 docker compose up
```

### XPU Build

Use this for Intel discrete or integrated GPUs. Requires Intel GPU drivers and the oneAPI runtime on the host.

```bash
AI_DEVICE=xpu docker compose build
HOST=0.0.0.0 AI_DEVICE=xpu docker compose up
```

### CUDA Build

Use this for NVIDIA GPUs. Uses a separate Compose file configured for the CUDA runtime.

```bash
AI_DEVICE=gpu docker compose -f docker-compose.cuda.yaml build
HOST=0.0.0.0 AI_DEVICE=gpu docker compose -f docker-compose.cuda.yaml up
```

---

## Accessing the Application

Once the containers are running, open your browser and navigate to:

```
http://localhost:8000
```

Allow a few seconds for all services to initialize before the UI becomes available. To run the containers in the background without occupying the terminal, append `-d` to the `docker compose up` command. The `HOST=0.0.0.0` variable ensures the application binds to all network interfaces, making it accessible from outside the container.

---

## Stopping the Application

To stop and remove the running containers:

```bash
docker compose down
```

---

## Troubleshooting

If the application does not start or the UI is unreachable, check the container logs for errors:

```bash
docker compose logs -f
```

Look for lines indicating which port the service is listening on, or any startup failures related to missing drivers or permissions.
