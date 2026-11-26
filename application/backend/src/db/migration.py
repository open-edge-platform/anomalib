# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from alembic.config import Config
from alembic.runtime import migration
from alembic.script import ScriptDirectory
from loguru import logger
from sqlalchemy import text

from alembic import command
from db import sync_engine as db_engine
from settings import Settings


class RevisionNotFoundError(Exception):
    """Raised when the current revision is not found in Alembic history."""


class MigrationManager:
    """Manages database connections and migrations"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.__ensure_data_directory()

    def __ensure_data_directory(self) -> None:
        """Ensure the data directory exists"""
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_connection() -> bool:
        """Check if database connection is working"""
        try:
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    def get_alembic_config(self) -> Config:
        """Get Alembic configuration"""
        alembic_cfg = Config(self.settings.alembic_config_path)
        alembic_cfg.set_main_option("script_location", self.settings.alembic_script_location)
        alembic_cfg.set_main_option("sqlalchemy.url", self.settings.sync_database_url)
        return alembic_cfg

    def run_migrations(self) -> bool:
        """Run database migrations"""
        try:
            logger.info("Running database migrations...")
            alembic_cfg = self.get_alembic_config()
            command.upgrade(alembic_cfg, "head")
            logger.success("✓ Database migrations completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Database migration failed: {e}")
            return False

    def check_migration_status(self) -> tuple[bool, str]:
        """Check if database needs migration"""
        try:
            alembic_cfg = self.get_alembic_config()
            script = ScriptDirectory.from_config(alembic_cfg)
            current_head = script.get_current_head()

            with db_engine.connect() as conn:
                context = migration.MigrationContext.configure(conn)
                current_rev = context.get_current_revision()

            # Check if current_rev is in Alembic's tracked revisions
            if current_rev and current_rev not in script.get_heads() + script.get_bases():
                raise RevisionNotFoundError(
                    f"Current revision '{current_rev}' not found in Alembic history. Please, recreate the database.",
                )

            needs_migration = current_rev != current_head
            status = f"Current: {current_rev or 'None'}, Head: {current_head or 'None'}"

            return needs_migration, status

        except RevisionNotFoundError:
            raise
        except Exception as e:
            logger.warning(f"Could not check migration status: {e}")
            return True, "Unknown - assuming migration needed"

    def initialize_database(self) -> bool:
        """Initialize database with migrations if needed"""
        try:
            # Ensure data directory exists
            self.__ensure_data_directory()

            # Check if we can connect
            if not self.check_connection():
                logger.error("Cannot connect to database")
                return False

            # Check migration status
            needs_migration, status = self.check_migration_status()
            logger.info(f"Migration status: {status}")

            if needs_migration:
                logger.info("Database needs migration")
                return self.run_migrations()
            logger.success("Database is up to date")
            return True

        except RevisionNotFoundError as e:
            logger.error(f"Revision not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False

    def make_revision(self, message: str, autogenerate: bool = True) -> None:
        """Create a new Alembic revision.

        Returns: path to the new revision file if successful, else None.
        """
        try:
            alembic_cfg = self.get_alembic_config()
            script = command.revision(alembic_cfg, message=message, autogenerate=autogenerate)

            if isinstance(script, list):
                for s in [s for s in script if s]:
                    logger.success(f"✓ Created migration {s.revision}: {s.path}")
            elif script is not None:
                # script is an alembic.script.revision.Revision object
                logger.success(f"✓ Created migration {script.revision}: {script.path}")
            else:
                raise RuntimeError("Alembic did not return a revision script.")
        except Exception as e:
            logger.error(f"✗ Failed to create migration: {e}")
            raise e
