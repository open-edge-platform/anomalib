# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints.active_pipeline_endpoints import router as active_pipeline_router
from api.endpoints.capture_endpoints import router as capture_router
from api.endpoints.devices_endpoints import device_router
from api.endpoints.job_endpoints import job_router
from api.endpoints.media_endpoints import media_router
from api.endpoints.model_endpoints import model_router
from api.endpoints.pipeline_endpoints import router as pipeline_router
from api.endpoints.project_endpoints import project_router
from api.endpoints.sink_endpoints import router as sink_router
from api.endpoints.snapshot_endpoints import router as snapshot_router
from api.endpoints.source_endpoints import router as source_router
from api.endpoints.trainable_models_endpoints import router as trainable_model_router
from api.endpoints.webrtc import router as webrtc_router
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

# TODO: check if middleware is required
# Enable CORS for local test UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ],
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
app.include_router(webrtc_router)
app.include_router(trainable_model_router)
app.include_router(device_router)
app.include_router(capture_router)
app.include_router(snapshot_router)


if __name__ == "__main__":
    settings = get_settings()
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", settings.port))
    uvicorn.run("main:app", loop="uvloop", host=settings.host, port=uvicorn_port, log_config=None)
