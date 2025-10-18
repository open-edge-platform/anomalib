# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from core.logging import setup_logging
from core.scheduler import Scheduler
from db import init_models
from settings import get_settings
from webrtc.manager import WebRTCManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    setup_logging()

    settings = get_settings()
    logger.info(f"Starting {settings.app_name} application...")

    # Initialize database
    # TODO add migration
    # if not migration_manager.initialize_database():
    #     logger.error("Failed to initialize database. Application cannot start.")
    #     raise RuntimeError("Database initialization failed")
    await init_models()

    # Initialize Scheduler
    app_scheduler = Scheduler()
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler
    app.state.active_models = {}

    webrtc_manager = WebRTCManager(app_scheduler.rtc_stream_queue)
    app.state.webrtc_manager = webrtc_manager
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app_name} application...")
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
