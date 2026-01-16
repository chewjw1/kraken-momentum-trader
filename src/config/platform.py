"""
Platform detection and environment abstraction.
Auto-detects Replit vs AWS and provides appropriate configurations.
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Platform(Enum):
    """Supported deployment platforms."""
    REPLIT = "replit"
    AWS = "aws"
    LOCAL = "local"


@dataclass
class PlatformConfig:
    """Platform-specific configuration."""
    platform: Platform
    secrets_backend: str  # "env", "secrets_manager"
    storage_backend: str  # "file", "dynamodb"
    logging_backend: str  # "console", "cloudwatch"


def detect_platform() -> Platform:
    """
    Detect the current deployment platform.

    Returns:
        Platform enum indicating the detected environment.
    """
    # Check for Replit environment
    if os.environ.get("REPL_ID") or os.environ.get("REPLIT_DB_URL"):
        return Platform.REPLIT

    # Check for AWS environment
    if os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return Platform.AWS

    # Check for ECS
    if os.environ.get("ECS_CONTAINER_METADATA_URI"):
        return Platform.AWS

    return Platform.LOCAL


def get_platform_config() -> PlatformConfig:
    """
    Get configuration appropriate for the current platform.

    Returns:
        PlatformConfig with platform-specific settings.
    """
    platform = detect_platform()

    if platform == Platform.REPLIT:
        return PlatformConfig(
            platform=platform,
            secrets_backend="env",
            storage_backend="file",
            logging_backend="console"
        )
    elif platform == Platform.AWS:
        return PlatformConfig(
            platform=platform,
            secrets_backend="secrets_manager",
            storage_backend="dynamodb",
            logging_backend="cloudwatch"
        )
    else:
        return PlatformConfig(
            platform=platform,
            secrets_backend="env",
            storage_backend="file",
            logging_backend="console"
        )


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value from the appropriate backend.

    Args:
        key: The secret key name.
        default: Default value if secret not found.

    Returns:
        The secret value or default.
    """
    platform = detect_platform()

    if platform == Platform.AWS:
        return _get_aws_secret(key, default)
    else:
        return os.environ.get(key, default)


def _get_aws_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve secret from AWS Secrets Manager.

    Args:
        key: The secret key name.
        default: Default value if secret not found.

    Returns:
        The secret value or default.
    """
    try:
        import boto3
        import json

        client = boto3.client("secretsmanager")
        secret_name = os.environ.get("SECRET_NAME", "kraken-momentum-trader/secrets")

        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])

        return secrets.get(key, default)
    except Exception:
        # Fall back to environment variable
        return os.environ.get(key, default)


def get_kraken_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Get Kraken API credentials from the appropriate backend.

    Returns:
        Tuple of (api_key, api_secret).
    """
    api_key = get_secret("KRAKEN_API_KEY")
    api_secret = get_secret("KRAKEN_API_SECRET")

    return api_key, api_secret
