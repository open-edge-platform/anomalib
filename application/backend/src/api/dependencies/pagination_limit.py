# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from fastapi import HTTPException, Query, status

DEFAULT_PAGINATION_LIMIT = 20
MAX_PAGINATION_LIMIT = 50


class PaginationLimit:
    """Dependency factory for pagination limit with configurable bounds."""

    def __init__(self, default: int = DEFAULT_PAGINATION_LIMIT, max_limit: int = MAX_PAGINATION_LIMIT):
        self.default = default
        self.max_limit = max_limit

    def __call__(
        self,
        limit: Annotated[int | None, Query(ge=1, description="Number of items to return")] = None,
    ) -> int:
        actual_limit = limit if limit is not None else self.default

        if actual_limit > self.max_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Limit must be between 1 and {self.max_limit}",
            )
        return actual_limit
