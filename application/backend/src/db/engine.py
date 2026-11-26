# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager, contextmanager
from sqlite3 import Connection
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.engine.create import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm.session import Session, sessionmaker
from sqlalchemy.pool.impl import NullPool

from db.schema import Base
from settings import get_settings

settings = get_settings()

async_engine = create_async_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30},
    # Using NullPool to disable connection pooling, which is necessary for SQLite when using multiprocessing
    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    poolclass=NullPool,
    echo=settings.db_echo,
)
async_session = async_sessionmaker(async_engine, expire_on_commit=False)

sync_engine = create_engine(
    settings.sync_database_url,
    connect_args={"check_same_thread": False, "timeout": 30},
    # Using NullPool to disable connection pooling, which is necessary for SQLite when using multiprocessing
    # https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    poolclass=NullPool,
    echo=settings.db_echo,
)
sync_session = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Connection, _: Any) -> None:
    """Enable foreign key support for SQLite."""
    # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support
    # SQLAlchemy's event system does not directly expose async version
    # https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#using-events-with-the-asyncio-extension
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")  # avoids database being locked
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


async def get_async_db_session() -> AsyncGenerator[AsyncSession, Any]:
    """Get a database session.

    To be used for dependency injection.
    """
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_async_db_session_ctx() -> AsyncGenerator[AsyncSession, Any]:
    """Get a database session for direct async context manager usage."""
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@contextmanager
def get_sync_db_session() -> Iterator[Session]:
    """Context manager to get a database session."""
    db = sync_session()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def init_models() -> None:
    """Create tables if they don't already exist.

    In a real-life example we would use Alembic to manage migrations.
    """
    async with async_engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
