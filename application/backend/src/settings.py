# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory where this settings module is located
_MODULE_DIR = Path(__file__).parent


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
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    static_files_dir: str | None = Field(default=None, alias="STATIC_FILES_DIR")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
    port: int = Field(default=8000, alias="PORT")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000",
        alias="CORS_ORIGINS",
    )

    # Database
    database_file: str = Field(default="geti_inspect.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = str(_MODULE_DIR / "alembic.ini")
    alembic_script_location: str = str(_MODULE_DIR / "alembic")

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    # WebRTC
    ice_servers: list[dict] = Field(default=[], alias="ICE_SERVERS")
    webrtc_advertise_ip: str | None = Field(default=None, alias="WEBRTC_ADVERTISE_IP")

    # Stream
    stream_max_clients: int = Field(default=5, alias="STREAM_MAX_CLIENTS")

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Get CORS allowed origins as a list"""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"sqlite+aiosqlite:///{self.data_dir / self.database_file}?journal_mode=WAL"

    @property
    def sync_database_url(self) -> str:
        """Get synchronous database URL"""
        return f"sqlite:///{self.data_dir / self.database_file}"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
