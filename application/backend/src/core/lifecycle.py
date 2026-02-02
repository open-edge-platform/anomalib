# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from aiortc import RTCConfiguration, RTCIceServer
from fastapi import FastAPI
from loguru import logger

from core.logging import setup_logging, setup_uvicorn_logging
from core.scheduler import Scheduler
from db import MigrationManager
from settings import get_settings
from webrtc.manager import WebRTCManager, WebRTCSettings
from webrtc.sdp_handler import SDPHandler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    setup_logging()
    setup_uvicorn_logging()

    settings = get_settings()
    app.state.settings = settings
    logger.info(f"Starting {settings.app_name} application...")

    # Initialize database with migrations
    migration_manager = MigrationManager(settings)
    if not migration_manager.initialize_database():
        logger.error("Failed to initialize database. Application cannot start.")
        raise RuntimeError("Database initialization failed")

    # Initialize Scheduler
    app_scheduler = Scheduler()
    app_scheduler.initialize_broadcaster(asyncio.get_running_loop())
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler
    app.state.active_models = {}

    webrtc_settings = WebRTCSettings(
        config=RTCConfiguration(iceServers=[RTCIceServer(**server) for server in settings.ice_servers]),
        advertise_ip=settings.webrtc_advertise_ip,
    )
    sdp_handler = SDPHandler()
    webrtc_manager = WebRTCManager(app_scheduler.rtc_stream_queue, webrtc_settings, sdp_handler)
    app.state.webrtc_manager = webrtc_manager
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app_name} application...")
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
