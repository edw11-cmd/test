"""
Configuration management for HyperCortex-AI
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(4000, env="OPENAI_MAX_TOKENS")
    
    # Opik Configuration
    opik_api_key: Optional[str] = Field(None, env="OPIK_API_KEY")
    opik_workspace: str = Field("hypercortex-ai", env="OPIK_WORKSPACE")
    opik_project_name: str = Field("hypercortex-main", env="OPIK_PROJECT_NAME")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./hypercortex.db", env="DATABASE_URL")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Application Configuration
    app_name: str = Field("HyperCortex-AI", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(12000, env="PORT")
    
    # Memory Configuration
    vector_store_type: str = Field("faiss", env="VECTOR_STORE_TYPE")
    max_memory_size: int = Field(10000, env="MAX_MEMORY_SIZE")
    memory_similarity_threshold: float = Field(0.7, env="MEMORY_SIMILARITY_THRESHOLD")
    
    # Agent Configuration
    max_iterations: int = Field(10, env="MAX_ITERATIONS")
    reflection_enabled: bool = Field(True, env="REFLECTION_ENABLED")
    tool_timeout: int = Field(30, env="TOOL_TIMEOUT")
    
    # Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings