# Copyright (C) 2025-2026 Intel Corporation
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


class ThroughputMetrics(BaseModel):
    """Throughput metrics for inference operations"""

    avg_requests_per_second: float | None = Field(..., description="Average requests per second")
    total_requests: int | None = Field(..., description="Total number of requests in the time window")
    max_requests_per_second: float | None = Field(..., description="Max requests per second")


class InferenceMetrics(BaseModel):
    """Inference-related metrics"""

    latency: LatencyMetrics
    throughput: ThroughputMetrics


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
                    "latency": {"avg_ms": 15.1, "min_ms": 12.3, "max_ms": 30.4, "p95_ms": 25.4, "latest_ms": 15.6},
                    "throughput": {
                        "avg_requests_per_second": 66.7,
                        "total_requests": 4000,
                        "max_requests_per_second": 85.2,
                    },
                },
            },
        },
    }
