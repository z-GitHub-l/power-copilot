from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "power-copilot"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_api_base_url: str = "http://127.0.0.1:8000"

    upload_dir: str = "data/uploads"
    chroma_dir: str = "data/chroma"
    collection_name: str = "power_copilot_documents"

    chunk_size: int = 800
    chunk_overlap: int = 120

    llm_api_key: str = Field(default="", validation_alias=AliasChoices("LLM_API_KEY"))
    llm_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL", "LLM_API_BASE_URL"),
    )
    llm_model: str = Field(default="", validation_alias=AliasChoices("LLM_MODEL"))
    llm_timeout: int = Field(default=60, validation_alias=AliasChoices("LLM_TIMEOUT"))
    llm_temperature: float = Field(
        default=0.2,
        validation_alias=AliasChoices("LLM_TEMPERATURE"),
    )

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def llm_api_base_url(self) -> str:
        return self.llm_base_url

    def ensure_directories(self) -> None:
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
