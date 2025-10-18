"""Configuration management using Pydantic Settings."""
import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings from .env file."""

    # App
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Server
    fastapi_host: str = Field(default="0.0.0.0", alias="FASTAPI_HOST")
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")
    fastapi_reload: bool = Field(default=True, alias="FASTAPI_RELOAD")

    # PostgreSQL Database
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/rag_db", alias="DATABASE_URL")
    db_name: str = Field(default="rag_db", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="postgres", alias="DB_PASSWORD")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_min_conn: int = Field(default=1, alias="DB_MIN_CONN")
    db_max_conn: int = Field(default=10, alias="DB_MAX_CONN")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4-turbo", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")

    # Processing
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    default_top_k: int = Field(default=4, alias="DEFAULT_TOP_K")
    max_top_k: int = Field(default=20, alias="MAX_TOP_K")
    vector_dim: int = Field(default=1536, alias="VECTOR_DIM")

    # Web Scraping
    web_timeout: int = Field(default=10, alias="WEB_TIMEOUT")
    max_urls_per_request: int = Field(default=10, alias="MAX_URLS_PER_REQUEST")

    class Config:
        # Load from .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields

# ✅ Create global settings instance
try:
    settings = Settings()
    print("✅ Settings loaded successfully")
except Exception as e:
    print(f"❌ Error loading settings: {e}")
    # Use defaults if .env doesn't exist
    settings = Settings()