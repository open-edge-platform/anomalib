# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from multiprocessing.synchronize import Condition as ConditionClass

from pydantic_models import DisconnectedSinkConfig, DisconnectedSourceConfig, Sink, Source

logger = logging.getLogger(__name__)


# TODO: add implementation
class ActivePipelineService:
    """
    A service used in workers for loading pipeline-based application configuration from SQLite database.

    This service handles loading and monitoring configuration changes based on the active pipeline.
    The configuration is built from Source -> Pipeline -> Sinks relationships.

    Args:
        config_changed_condition: Multiprocessing Condition object for getting configuration updates in child
                                processes. Required for child processes.

    Raises:
        ValueError: When config_changed_condition is None in a child process.
    """

    def __init__(self, config_changed_condition: ConditionClass | None = None) -> None:
        self.config_changed_condition = config_changed_condition
        self._source: Source = DisconnectedSourceConfig()
        self._sink: Sink = DisconnectedSinkConfig()

    def get_source_config(self) -> Source:
        return self._source

    def get_sink_config(self) -> Sink:
        return self._sink
