"""
Configuration management for Felix Framework deployment.
Handles environment variables, secrets, and feature flags with validation.
"""

import os
import secrets
from typing import List, Optional, Any, Dict
from functools import lru_cache

from pydantic import BaseSettings, validator, Field
from pydantic_settings import SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Security-related configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # JWT Configuration
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # API Keys
    felix_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:7860"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_allow_headers: List[str] = ["*"]

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10

    # Input Validation
    max_input_size: int = 10000
    max_file_size_mb: int = 50

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class LLMSettings(BaseSettings):
    """LLM integration configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Primary LLM Configuration
    llm_endpoint: str = "http://localhost:1234"
    llm_api_key: Optional[str] = None
    llm_provider: str = "lm_studio"  # lm_studio, openai, huggingface

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    openai_model: str = "gpt-4"

    # Hugging Face Configuration
    hf_inference_api_key: Optional[str] = None
    hf_model_endpoint: Optional[str] = None

    # Multi-model Configuration
    enable_multi_model: bool = True
    research_model: str = "qwen/qwen3-4b-2507"
    analysis_model: str = "qwen/qwen3-4b-thinking-2507"
    synthesis_model: str = "google/gemma-3-12b"
    critic_model: str = "qwen/qwen3-4b-thinking-2507"

    # Token Management
    default_token_budget: int = 100000
    max_tokens_per_request: int = 4096
    cost_per_token: float = 0.0001

    # Request Configuration
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3
    llm_retry_delay: float = 1.0

    @validator("llm_provider")
    def validate_llm_provider(cls, v):
        valid_providers = ["lm_studio", "openai", "huggingface"]
        if v not in valid_providers:
            raise ValueError(f"Invalid LLM provider. Must be one of: {valid_providers}")
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring, logging, and metrics configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"  # json, text
    log_file: str = "logs/felix.log"
    log_rotation: str = "1d"
    log_retention: str = "30d"

    # Structured Logging
    enable_structured_logging: bool = True
    log_correlation_id: bool = True
    log_performance_metrics: bool = True

    # Metrics Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    prometheus_metrics_path: str = "/metrics"

    # Health Checks
    health_check_timeout: int = 10
    health_check_interval: int = 30

    # Performance Monitoring
    enable_performance_tracking: bool = True
    track_response_times: bool = True
    track_memory_usage: bool = True
    track_cpu_usage: bool = True

    # Error Tracking (Sentry)
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "development"
    sentry_traces_sample_rate: float = 0.1
    enable_error_reporting: bool = True

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class DatabaseSettings(BaseSettings):
    """Database and caching configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Database Configuration
    database_url: str = "sqlite:///./felix_framework.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: int = 30
    auto_migrate: bool = False
    backup_before_migrate: bool = True

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_max_connections: int = 10
    redis_timeout: int = 5

    # Cache Settings
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    enable_cache: bool = True

    # Session Management
    session_timeout_minutes: int = 60
    session_cleanup_interval: int = 300


class FeatureFlags(BaseSettings):
    """Feature flags for experimental and optional functionality."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Experimental Features
    enable_experimental_features: bool = False
    enable_dynamic_spawning: bool = True
    enable_adaptive_temperature: bool = True
    enable_helix_visualization: bool = True

    # Research Mode
    enable_research_mode: bool = False
    collect_anonymous_metrics: bool = True
    experimental_algorithms: bool = False

    # Core Features
    feature_multi_agent_coordination: bool = True
    feature_geometric_optimization: bool = True
    feature_statistical_validation: bool = True
    feature_real_time_updates: bool = True

    # Development Features
    enable_debug_endpoints: bool = False
    mock_llm_responses: bool = False
    enable_load_testing: bool = False


class ResourceSettings(BaseSettings):
    """Resource limits and optimization settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Memory Management
    max_memory_usage_mb: int = 1024
    memory_cleanup_threshold: float = 0.8
    enable_memory_monitoring: bool = True

    # CPU Optimization
    max_cpu_usage_percent: int = 80
    cpu_affinity: str = "auto"
    thread_pool_size: int = 4

    # Disk Usage
    max_disk_usage_gb: int = 10
    cleanup_old_logs: bool = True
    cleanup_old_metrics: bool = True

    # Hugging Face Spaces Limits
    max_memory_gb: int = 16
    max_cpu_cores: int = 8
    timeout_seconds: int = 300

    @validator("memory_cleanup_threshold")
    def validate_memory_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Memory cleanup threshold must be between 0.0 and 1.0")
        return v


class AppSettings(BaseSettings):
    """Main application configuration combining all settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Application Metadata
    app_name: str = "Felix Framework"
    app_version: str = "0.5.0"
    app_description: str = "Helix-based Multi-Agent Cognitive Architecture"

    # Deployment Configuration
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 7860
    workers: int = 1
    reload: bool = False
    debug: bool = False

    # Hugging Face Spaces
    hf_space_id: Optional[str] = None
    hf_repo_type: str = "space"

    # Component Settings
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)

    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_database_url(self, test: bool = False) -> str:
        """Get database URL with optional test database."""
        if test:
            return os.getenv("TEST_DATABASE_URL", "sqlite:///./test_felix.db")
        return self.database.database_url


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached application settings instance."""
    return AppSettings()


# Convenience functions for accessing specific setting groups
def get_security_settings() -> SecuritySettings:
    """Get security settings."""
    return get_settings().security


def get_llm_settings() -> LLMSettings:
    """Get LLM settings."""
    return get_settings().llm


def get_monitoring_settings() -> MonitoringSettings:
    """Get monitoring settings."""
    return get_settings().monitoring


def get_database_settings() -> DatabaseSettings:
    """Get database settings."""
    return get_settings().database


def get_feature_flags() -> FeatureFlags:
    """Get feature flags."""
    return get_settings().features


def get_resource_settings() -> ResourceSettings:
    """Get resource settings."""
    return get_settings().resources


# Environment-specific configuration overrides
def configure_for_hugging_face_spaces():
    """Apply Hugging Face Spaces specific configurations."""
    settings = get_settings()

    # Override for HF Spaces constraints
    settings.resources.max_memory_usage_mb = min(
        settings.resources.max_memory_usage_mb,
        settings.resources.max_memory_gb * 1024
    )

    # Disable features that might not work in HF Spaces
    if settings.environment == "production":
        settings.features.enable_debug_endpoints = False
        settings.monitoring.enable_performance_tracking = True
        settings.security.rate_limit_requests_per_minute = 30  # More conservative

    return settings


def validate_configuration() -> Dict[str, Any]:
    """Validate configuration and return status report."""
    try:
        settings = get_settings()
        issues = []
        warnings = []

        # Check required settings for different environments
        if settings.is_production:
            if not settings.security.secret_key or len(settings.security.secret_key) < 32:
                issues.append("Production requires strong SECRET_KEY (32+ characters)")

            if settings.debug:
                warnings.append("Debug mode should be disabled in production")

            if not settings.monitoring.sentry_dsn:
                warnings.append("Consider setting up Sentry for error tracking in production")

        # Check LLM configuration
        if settings.llm.llm_provider == "openai" and not settings.llm.openai_api_key:
            issues.append("OpenAI API key required when using OpenAI provider")

        if settings.llm.llm_provider == "huggingface" and not settings.llm.hf_inference_api_key:
            issues.append("Hugging Face API key required when using HF provider")

        # Resource validation
        if settings.resources.max_memory_usage_mb > settings.resources.max_memory_gb * 1024:
            warnings.append("Memory limit exceeds available memory")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "environment": settings.environment,
            "features_enabled": {
                "multi_model": settings.llm.enable_multi_model,
                "metrics": settings.monitoring.enable_metrics,
                "cache": settings.database.enable_cache,
                "experimental": settings.features.enable_experimental_features
            }
        }

    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Configuration validation failed: {str(e)}"],
            "warnings": [],
            "environment": "unknown"
        }


if __name__ == "__main__":
    # CLI tool for configuration validation
    import json

    result = validate_configuration()
    print(json.dumps(result, indent=2))

    if not result["valid"]:
        exit(1)