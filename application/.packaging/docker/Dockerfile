#------------------------------------------
# Web UI Build Stage
#------------------------------------------
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS web-ui-base

WORKDIR /home/app/web_ui/

COPY --link application/ui/package.json ./package.json
COPY --link application/ui/package-lock.json ./package-lock.json
COPY --link application/ui/packages ./packages
RUN npm ci --audit=false --ignore-scripts


COPY --link application/ui/tsconfig.json ./tsconfig.json
COPY --link application/ui/rsbuild.config.ts ./rsbuild.config.ts
COPY --link application/ui/src ./src

ENV PUBLIC_API_BASE_URL=""
ENV PUBLIC_API_BASE_URL=${PUBLIC_API_BASE_URL}

FROM web-ui-base AS web-ui


RUN npm run build


#------------------------------------------
# Geti Inspect Base with UI built
#------------------------------------------
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS geti-inspect-base
COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0


# Create non-root user and group
RUN groupadd --gid 10001 non-root && \
useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash non-root

RUN apt-get update \
&& apt-get install -y --no-install-recommends \
libgl1 \
libglib2.0-0 \
g++ \
nginx=1.26.* \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

ENV PATH=/app/.venv/bin:$PATH
WORKDIR /app
ENV PYTHONPATH=/app/application/backend

COPY --chown=non-root:non-root application/.packaging/docker/nginx.conf /etc/nginx/nginx.conf
COPY --chown=non-root:non-root --from=web-ui /home/app/web_ui/dist /usr/share/nginx/html

# Copy anomalib source code (pyproject.toml depends on README and LICENSE as well)
COPY --chown=non-root:non-root  src ./src
COPY --chown=non-root:non-root  pyproject.toml uv.lock ./
COPY --chown=non-root:non-root LICENSE ./LICENSE
COPY --chown=non-root:non-root README.md ./README.md

# Copy backend application
COPY --chown=non-root:non-root application/backend /app/application/backend

# Create nginx directories and set ownership for non-root user
RUN mkdir -p /var/lib/nginx/body /var/lib/nginx/proxy /var/lib/nginx/fastcgi \
/var/lib/nginx/uwsgi /var/lib/nginx/scgi /var/run/nginx \
&& chown -R non-root:non-root /var/lib/nginx /var/run/nginx /var/log/nginx \
&& chmod -R 755 /var/lib/nginx /var/run/nginx /var/log/nginx

# Create data and logs directories with write permissions for mounted volumes
RUN mkdir -p /app/data /app/logs \
 && chown -R non-root:non-root /app


#------------------------------------------
# Geti Inspect CPU version
#------------------------------------------
FROM geti-inspect-base AS geti-inspect-cpu
USER non-root
WORKDIR /app/application/backend

RUN --mount=type=cache,target=/home/non-root/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --extra cpu

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run src/main.py"]

#------------------------------------------
# Geti Inspect CUDA version
#------------------------------------------
FROM geti-inspect-base AS geti-inspect-cuda

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-cudart-12-4 \
    libcublas-12-4 \
    libcufft-12-4 \
    libcurand-12-4 \
    libcusolver-12-4 \
    libcusparse-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER non-root
WORKDIR /app/application/backend

RUN --mount=type=cache,target=/home/non-root/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --extra cu124

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run src/main.py"]

#------------------------------------------
# Geti Inspect XPU version
#------------------------------------------
FROM geti-inspect-base AS geti-inspect-xpu

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

RUN groupadd -f render && usermod -aG render non-root

USER non-root
WORKDIR /app/application/backend

RUN --mount=type=cache,target=/home/non-root/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --extra xpu

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run src/main.py"]
