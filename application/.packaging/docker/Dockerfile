# Install UI Packages
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS web-ui-base

WORKDIR /home/app/web_ui/

COPY --link application/ui/package.json ./package.json
COPY --link application/ui/package-lock.json ./package-lock.json
COPY --link application/ui/packages ./packages
RUN npm ci --audit=false --ignore-scripts


COPY --link application/ui/tsconfig.json ./tsconfig.json
COPY --link application/ui/rsbuild.config.ts ./rsbuild.config.ts
COPY --link application/ui/src ./src


FROM web-ui-base AS web-ui

RUN npm run build


# Geti Inspect Base with UI built
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS geti-inspect-base
COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/

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

WORKDIR /app
ENV PYTHONPATH=/app

COPY --chown=non-root:non-root application/.packaging/docker/nginx.conf /etc/nginx/nginx.conf
COPY --chown=non-root:non-root --from=web-ui /home/app/web_ui/dist /usr/share/nginx/html

# Copy anomalib source code (required for build)
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
RUN mkdir -p /app/data /app/logs 
RUN chown -R non-root:non-root /app

# Geti Inspect CPU version
FROM geti-inspect-base AS geti-inspect-cpu
USER non-root
WORKDIR /app

RUN --mount=type=cache,target=/home/non-root/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --extra cpu --extra application

EXPOSE 80

CMD ["sh", "-c", "nginx && uv run geti-inspect"]
