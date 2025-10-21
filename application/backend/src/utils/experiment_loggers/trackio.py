# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import trackio
from lightning.pytorch.loggers import Logger as LightningLogger
from loguru import logger


class TrackioLogger(LightningLogger):
    def __init__(
        self,
        project: str,
        name: str | None = None,
    ):
        super().__init__()
        self._project = project
        self._name = name

        self._logger = logger.bind(
            project=self._project,
            name=self._name,
        )
        # Initialize run
        self.run = trackio.init(
            project=self._project,
            name=self._name,
        )

    @property
    def name(self) -> str:
        return self._name or self._project

    @property
    def version(self) -> str:
        # You might use some attribute of trackio.Run if exists
        return f"run_{id(self.run)}"

    def log_hyperparams(self, params: dict[str, Any] | Any) -> None:
        # Pass hyperparams via config if possible, or log as metrics
        params_dict = dict(params) if not isinstance(params, dict) else params
        # One option: include in config during init; if already initialized,
        # log them as metrics at step=0 or custom logic
        trackio.log({f"hyperparams/{k}": v for k, v in params_dict.items()}, step=0)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        trackio.log(metrics, step=step)

    def finalize(self, _status: str) -> None:
        # PyTorch Lightning calls logger.finalize("success" / "failed" etc.)
        trackio.finish()