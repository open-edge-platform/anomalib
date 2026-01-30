# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.endpoints.active_pipeline_endpoints import router as active_pipeline_router
from api.endpoints.capture_endpoints import router as capture_router
from api.endpoints.job_endpoints import job_router
from api.endpoints.media_endpoints import media_router
from api.endpoints.model_endpoints import model_router
from api.endpoints.pipeline_endpoints import router as pipeline_router
from api.endpoints.project_endpoints import project_router
from api.endpoints.sink_endpoints import router as sink_router
from api.endpoints.snapshot_endpoints import router as snapshot_router
from api.endpoints.source_endpoints import router as source_router
from api.endpoints.stream_endpoints import router as stream_router
from api.endpoints.system_endpoints import system_router
from api.endpoints.trainable_models_endpoints import router as trainable_model_router
from api.endpoints.video_endpoints import router as video_router
from api.endpoints.webui_endpoints import webui_router
from core.lifecycle import lifespan
from settings import get_settings

app = FastAPI(
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    redoc_url=None,
    docs_url=None,
)

import exception_handlers  # noqa: E402

_ = exception_handlers  # to avoid import being removed by linters

settings = get_settings()
# TODO: check if middleware is required
# Enable CORS for local test UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(project_router)
app.include_router(job_router)
app.include_router(media_router)
app.include_router(model_router)
app.include_router(pipeline_router)
app.include_router(active_pipeline_router)
app.include_router(source_router)
app.include_router(sink_router)

# TODO: WebRTC disabled due to connectivity issues: revisit implementation
# app.include_router(webrtc_router)

app.include_router(trainable_model_router)
app.include_router(capture_router)
app.include_router(snapshot_router)
app.include_router(system_router)
app.include_router(video_router)
app.include_router(stream_router)

settings = get_settings()

# In docker deployment, the UI is built and served statically
if (
    settings.static_files_dir
    and Path(settings.static_files_dir).is_dir()
    and (Path(settings.static_files_dir) / "index.html").exists()
):
    static_dir = Path(settings.static_files_dir)
    app.mount("/static", StaticFiles(directory=static_dir / "static"), name="static")
    app.include_router(webui_router)


def main() -> None:
    """Main function to run the Geti Inspect server"""
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", settings.port))
    uvicorn.run("main:app", loop="uvloop", host=settings.host, port=uvicorn_port, log_config=None)


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    main()
