# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.scheduler import Scheduler
from db import init_models
from settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    settings = get_settings()
    logger.info("Starting %s application...", settings.app_name)

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

    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
