# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Application
    app_name: str = "Geti Inspect"
    version: str = "0.1.0"
    summary: str = "Geti Inspect server"
    description: str = "Geti Inspect allows to fine-tune anomaly models at the edge."
    openapi_url: str = "/rest_api/openapi.json"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: Literal["dev", "prod"] = "dev"

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
    port: int = Field(default=7860, alias="PORT")

    # Database
    database_url: str = Field(
        default="sqlite:///./data/geti_inspect.db", alias="DATABASE_URL", description="Database connection URL"
    )
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic TODO: add alembic support
    # alembic_config_path: str = "app/alembic.ini"
    # alembic_script_location: str = "app/alembic"

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    @property
    def database_dir(self) -> Path:
        """Get database directory path"""
        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url.replace("sqlite:///", ""))
            return db_path.parent
        return Path("./data")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
