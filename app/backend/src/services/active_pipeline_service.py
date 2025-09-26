# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread

from db import get_async_db_session_ctx
from pydantic_models import DisconnectedSinkConfig, DisconnectedSourceConfig, Sink, Source
from repositories import PipelineRepository

logger = logging.getLogger(__name__)


class ActivePipelineService:
    """
    A service used in workers for loading pipeline-based application configuration from SQLite database.

    This service handles loading and monitoring configuration changes based on the active pipeline.
    The configuration is built from Source -> Pipeline -> Sinks relationships.

    The service must be created using the `create()` class method for proper async initialization.
    In child processes, it automatically starts a daemon thread to monitor configuration changes.

    Example:
        # Create service in main process
        service = await ActivePipelineService.create()

        # Create service in child process with condition monitoring
        service = await ActivePipelineService.create(config_changed_condition)
    """

    @classmethod
    async def create(
        cls, config_changed_condition: ConditionClass | None = None, start_daemon: bool = False
    ) -> "ActivePipelineService":
        """
        Factory method to create and initialize the service asynchronously.

        Args:
            config_changed_condition: Multiprocessing Condition object for getting configuration
                                    updates in child processes. Required for child processes.
            start_daemon: whether or not to start daemon thread to monitor configuration changes.

        Returns:
            ActivePipelineService: Fully initialized service instance.

        Raises:
            ValueError: When config_changed_condition is None in a child process.
        """
        instance = cls()
        await instance._initialize(config_changed_condition, start_daemon)
        return instance

    async def _initialize(
        self, config_changed_condition: ConditionClass | None = None, start_daemon: bool = False
    ) -> None:
        """
        Initialize the service asynchronously.

        Args:
            config_changed_condition: Multiprocessing Condition object for getting configuration
                                    updates in child processes. Required for child processes.
            start_daemon: whether or not to start daemon thread to monitor configuration changes.

        Raises:
            ValueError: When config_changed_condition is None in a child process.
        """
        self.config_changed_condition = config_changed_condition
        self._source: Source = DisconnectedSourceConfig()
        self._sink: Sink = DisconnectedSinkConfig()
        await self._load_app_config()

        # For child processes with config_changed_condition, start a daemon to monitor configuration changes
        if start_daemon is not None and self.config_changed_condition is not None:
            # Store the current event loop for the daemon thread to use
            self._event_loop = asyncio.get_running_loop()

            self._config_reload_daemon = Thread(
                target=self._reload_config_daemon_routine, name="Config reloader", daemon=True
            )
            self._config_reload_daemon.start()
        elif start_daemon is not None and self.config_changed_condition is None:
            # This is a child process but no condition provided - this is likely an API process
            # that doesn't need the daemon thread, so we just log and continue
            logger.debug("Child process detected but no config_changed_condition provided - skipping daemon thread")

    async def reload(self) -> None:
        """
        Reload the application configuration from the database.

        This method must be called from an async context and will await
        the configuration reload operation.
        """
        await self._load_app_config()

    async def _load_app_config(self) -> None:
        """
        Load application configuration from the database.

        This method loads the active pipeline configuration and updates the
        internal source and sink configurations accordingly.
        """
        logger.info("Loading configuration from database")
        async with get_async_db_session_ctx() as db:
            repo = PipelineRepository(db)

            # Loads the first active pipeline
            pipeline = await repo.get_active_pipeline()
            if pipeline is None:
                self._source = DisconnectedSourceConfig()
                self._sink = DisconnectedSinkConfig()
                return

            source = pipeline.source
            if source is not None:
                self._source = source

            sink = pipeline.sink
            if sink is not None:
                self._sink = sink

    def _reload_config_daemon_routine(self) -> None:
        """
        Daemon thread routine to monitor configuration changes and reload when necessary.

        This method runs in a separate thread and waits for configuration change
        notifications. When changes are detected, it schedules the async reload
        operation in the main event loop.

        Raises:
            RuntimeError: When config_changed_condition is None.
        """
        if self.config_changed_condition is None:
            raise RuntimeError("daemon thread initialized without config_changed_condition")
        while True:
            with self.config_changed_condition:
                notified = self.config_changed_condition.wait(timeout=3)
                if not notified:  # awakened before of timeout
                    continue
                logger.debug("Configuration changes detected. Process: %s", mp.current_process().name)
                # Schedule the async reload in the event loop using the stored loop reference
                asyncio.run(self.reload())

    def get_source_config(self) -> Source:
        """
        Get the current source configuration.

        Returns:
            Source: The current source configuration.
        """
        return self._source

    def get_sink_config(self) -> Sink:
        """
        Get the current sink configuration.

        Returns:
            Sink: The current sink configuration.
        """
        return self._sink
