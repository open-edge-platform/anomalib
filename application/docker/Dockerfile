#------------------------------------------
# Web UI Build Stage
#------------------------------------------
FROM node:24-alpine3.22@sha256:d28696cabe6a72c5addbb608b344818e5a158d849174abd4b1ae85ab48536280 AS web-ui-base

WORKDIR /home/app/web_ui/

# Install git for degit
RUN apk add --no-cache git


COPY --link application/ui/package.json ./package.json
COPY --link application/ui/package-lock.json ./package-lock.json
RUN npm ci --audit=false


COPY --link application/ui/tsconfig.json ./tsconfig.json
COPY --link application/ui/rsbuild.config.ts ./rsbuild.config.ts
COPY --link application/ui/src ./src

FROM web-ui-base AS web-ui

RUN npm run build


#------------------------------------------
# Anomalib Studio Base with UI built
#------------------------------------------
FROM python:3.13-slim@sha256:3de9a8d7aedbb7984dc18f2dff178a7850f16c1ae7c34ba9d7ecc23d0755e35f AS anomalib-studio-base
COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0


RUN apt-get update \
&& apt-get install -y --no-install-recommends \
libgl1 \
libglib2.0-0 \
g++ \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

ENV PATH=/app/.venv/bin:$PATH
WORKDIR /app

ARG STATIC_FILES_DIR="/app/application/ui/"
ENV STATIC_FILES_DIR=${STATIC_FILES_DIR}
ENV DATA_DIR="/app/data"
ENV LOG_DIR="/app/logs"
ENV PYTHONPATH=/app/application/backend

COPY --link --from=web-ui /home/app/web_ui/dist ${STATIC_FILES_DIR}
COPY  application/backend/run.sh /app/application/backend/run.sh

# Copy anomalib source code (pyproject.toml depends on README and LICENSE as well)
COPY --link src ./src
COPY --link pyproject.toml uv.lock ./
COPY --link LICENSE ./LICENSE
COPY --link README.md ./README.md

# Copy backend application
COPY --link application/backend /app/application/backend

# Create data and logs directories with write permissions for mounted volumes
RUN mkdir -p /app/data /app/logs


#------------------------------------------
# Anomalib Studio CPU version
#------------------------------------------
FROM anomalib-studio-base AS anomalib-studio-cpu

WORKDIR /app/application/backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cpu

EXPOSE 8000

# nosemgrep: dockerfile.security.missing-user.missing-user # will fix later
CMD ["sh", "-c", "exec ./run.sh"]

#------------------------------------------
# Anomalib Studio CUDA version
#------------------------------------------
FROM anomalib-studio-base AS anomalib-studio-cuda

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-13-1 \
    libcublas-13-1 \
    libcufft-13-1 \
    libcurand-13-1 \
    libcusolver-13-1 \
    libcusparse-13-1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app/application/backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cu124

EXPOSE 8000

# nosemgrep: dockerfile.security.missing-user.missing-user # will fix later
CMD ["sh", "-c", "exec ./run.sh"]

#------------------------------------------
# Anomalib Studio XPU version
#------------------------------------------
FROM anomalib-studio-base AS anomalib-studio-xpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends wget gpg \
 && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
 && echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
    tee /etc/apt/sources.list.d/intel-gpu-jammy.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    libze-intel-gpu1 \
    libze1 \
    intel-opencl-icd \
    clinfo \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get remove -y wget gpg \
 && apt-get clean



WORKDIR /app/application/backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra xpu

EXPOSE 8000

# nosemgrep: dockerfile.security.missing-user.missing-user # will fix later
CMD ["sh", "-c", "exec ./run.sh"]