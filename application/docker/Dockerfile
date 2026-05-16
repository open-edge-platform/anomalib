# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#------------------------------------------
# Web UI Build Stage
#------------------------------------------
FROM node:24-alpine3.22@sha256:b689d4005875ae167178471a7a622ec2909459a3bbb32277260be1971af7a99f AS web-ui-base

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
FROM python:3.13-slim@sha256:dc1546eefcbe8caaa1f004f16ab76b204b5e1dbd58ff81b899f21cd40541232f AS anomalib-studio-base
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

# Remove pip because uv is used as the package manager.
# This reduces attack surface and image size (a little bit).
# python:3.13-slim only ships pip (not setuptools/wheel).
ENV PIP_ROOT_USER_ACTION=ignore
RUN python -m pip uninstall -y pip \ 
    && rm -rf /root/.cache/pip
ENV PATH=/app/.venv/bin:$PATH
WORKDIR /app

ARG STATIC_FILES_DIR="/app/application/ui/"
ENV STATIC_FILES_DIR=${STATIC_FILES_DIR}
ENV DATA_DIR="/app/data"
ENV LOG_DIR="/app/logs"
ENV PYTHONPATH=/app/application/backend

COPY --link --from=web-ui /home/app/web_ui/dist ${STATIC_FILES_DIR}
COPY --link application/backend/run.sh /app/application/backend/run.sh

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
# Anomalib Studio CUDA intermediate stage to install CUDA dependencies
#------------------------------------------
FROM anomalib-studio-base AS anomalib-studio-cuda-base

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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
    uv sync --frozen --extra cu130

#------------------------------------------
# Anomalib Studio CUDA intermediate stage to cleanup CUDA non-distributable dependencies
#------------------------------------------
FROM anomalib-studio-cuda-base AS anomalib-studio-cuda-cleanup

# CUDA EULA Compliance Cleanup
RUN echo "Removing non-redistributable CUDA components per NVIDIA EULA..." && \
    # Define search paths including venv
    SEARCH_PATHS="/usr /opt" && \
     # Include PYTHONPATH in search paths if it is set
    if [ -n "${PYTHONPATH}" ]; then \
    SEARCH_PATHS="$SEARCH_PATHS ${PYTHONPATH}"; \
    fi && \
    # Add any detected venvs from PYTHONPATH (if set)
    for venv in $(find ${PYTHONPATH:+${PYTHONPATH}} -name "site-packages" -type d 2>/dev/null | sed 's|/lib/python[0-9.]*/site-packages||' | sort -u); do \
    if [ -d "$venv" ]; then \
    SEARCH_PATHS="$SEARCH_PATHS $venv"; \
    fi; \
    done && \
    echo "Cleaning CUDA components in: $SEARCH_PATHS" && \
    # Remove development tools (non-redistributable)
    find $SEARCH_PATHS -type f \( \
    -name "nvcc" -o \
    -name "cuda-gdb" -o \
    -name "nvprof" -o \
    -name "nvdisasm" -o \
    -name "cuobjdump" -o \
    -name "cuda-memcheck" -o \
    -name "nsight*" -o \
    -name "compute-sanitizer" -o \
    -name "ncu" -o \
    -name "nv-nsight-cu*" \
    \) -delete 2>/dev/null || true && \
    # Remove headers and include directories (non-redistributable)
    find $SEARCH_PATHS -type d \( \
    -name "include" -o \
    -name "headers" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove NVIDIA package includes specifically
    find $SEARCH_PATHS -type d -path "*/nvidia/*/include" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove static libraries (typically non-redistributable)
    find $SEARCH_PATHS -name "*.a" -path "*cuda*" -delete 2>/dev/null || true && \
    find $SEARCH_PATHS -name "*.a" -path "*nvidia*" -delete 2>/dev/null || true && \
    # Remove documentation and samples (non-redistributable)
    find $SEARCH_PATHS -type d \( \
    -name "doc" -o \
    -name "docs" -o \
    -name "documentation" -o \
    -name "samples" -o \
    -name "sample" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove specific non-redistributable CUPTI components
    find $SEARCH_PATHS -type f \( \
    -name "libcheckpoint.so*" -o \
    -name "libnvperf_host.so*" -o \
    -name "libnvperf_target.so*" -o \
    -name "libpcsamplingutil.so*" \
    \) -path "*cupti*" -delete 2>/dev/null || true && \
    # Remove cuSolver MG libraries (enterprise feature)
    find $SEARCH_PATHS -name "libcusolverMg.so*" -delete 2>/dev/null || true && \
    # Remove Triton NVIDIA backends (development components)
    find $SEARCH_PATHS -type d -path "*/triton/backends/nvidia" -exec rm -rf {} + 2>/dev/null || true && \
    # System package cleanup
    apt-get remove -y \
    cuda-compiler-* \
    cuda-cuobjdump-* \
    cuda-gdb-* \
    cuda-memcheck-* \
    cuda-nvcc-* \
    cuda-nvdisasm-* \
    cuda-nvprof-* \
    cuda-sanitizer-* \
    cuda-documentation-* \
    cuda-samples-* \
    nsight-* \
    2>/dev/null || true && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

#------------------------------------------
# Anomalib Studio CUDA version
#------------------------------------------
FROM anomalib-studio-base AS anomalib-studio-cuda

COPY --link --from=anomalib-studio-cuda-cleanup /usr/local /usr/local
COPY --link --from=anomalib-studio-cuda-cleanup ${PYTHONPATH} ${PYTHONPATH}

WORKDIR /app/application/backend

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