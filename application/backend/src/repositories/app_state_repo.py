# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sqlalchemy as sa
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import AppStateDB

APP_STATE_ID = 1


class AppStateRepository:
    """Repository for persisted application-wide state."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_last_used_project_id(self) -> str | None:
        result = await self.db.execute(
            sa.select(AppStateDB.last_used_project_id).where(AppStateDB.id == APP_STATE_ID),
        )
        return result.scalar_one_or_none()

    async def set_last_used_project_id(self, project_id: str) -> None:
        state = await self.db.get(AppStateDB, APP_STATE_ID)

        if state is None:
            self.db.add(AppStateDB(id=APP_STATE_ID, last_used_project_id=project_id))
        else:
            state.last_used_project_id = project_id

        await self.db.commit()

    async def clear_last_used_project_id(self) -> None:
        state = await self.db.get(AppStateDB, APP_STATE_ID)
        if state is None:
            return

        state.last_used_project_id = None
        await self.db.commit()
