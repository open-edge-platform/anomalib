# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import BaseModel, Field


class LatencyMetrics(BaseModel):
    """Latency metrics for inference operations"""

    avg_ms: float | None = Field(..., description="Average latency in milliseconds")
    min_ms: float | None = Field(..., description="Minimum latency in milliseconds")
    max_ms: float | None = Field(..., description="Maximum latency in milliseconds")
    p95_ms: float | None = Field(..., description="95th percentile latency in milliseconds")
    latest_ms: float | None = Field(..., description="Latest recorded latency in milliseconds")


class InferenceMetrics(BaseModel):
    """Inference-related metrics"""

    latency: LatencyMetrics


class TimeWindow(BaseModel):
    """Time window for metrics calculation"""

    start: datetime = Field(..., description="Start timestamp of the time window")
    end: datetime = Field(..., description="End timestamp of the time window")
    time_window: int = Field(..., description="Duration of the time window in seconds")


class PipelineMetrics(BaseModel):
    """Pipeline metrics response"""

    time_window: TimeWindow
    inference: InferenceMetrics

    model_config = {
        "json_schema_extra": {
            "example": {
                "time_window": {"start": "2025-08-25T10:00:00Z", "end": "2025-08-25T10:01:00Z", "time_window": 60},
                "inference": {
                    "latency": {"avg_ms": 15.1, "min_ms": 12.3, "max_ms": 30.4, "p95_ms": 25.4, "latest_ms": 15.6}
                },
            }
        }
    }
