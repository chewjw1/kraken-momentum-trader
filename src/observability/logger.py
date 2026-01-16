"""
Structured logging for the trading application.
Supports JSON and text formats with platform-specific backends.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from ..config.platform import detect_platform, Platform


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        base = f"[{timestamp}] {record.levelname:8} {record.name}: {record.getMessage()}"

        # Add extra fields if present
        if hasattr(record, "extra_fields") and record.extra_fields:
            extras = " | ".join(f"{k}={v}" for k, v in record.extra_fields.items())
            base = f"{base} | {extras}"

        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


class StructuredLogger:
    """
    Wrapper around standard logger with structured logging support.
    """

    def __init__(self, name: str, logger: logging.Logger):
        self._name = name
        self._logger = logger

    def _log(self, level: int, msg: str, **kwargs) -> None:
        """Log with optional extra fields."""
        record = self._logger.makeRecord(
            self._name, level, "", 0, msg, (), None
        )
        if kwargs:
            record.extra_fields = kwargs
        self._logger.handle(record)

    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, **kwargs)

    def trade(
        self,
        action: str,
        pair: str,
        side: str,
        price: float,
        amount: float,
        **kwargs
    ) -> None:
        """Log a trade event."""
        self.info(
            f"TRADE: {action} {side} {amount} {pair} @ {price}",
            action=action,
            pair=pair,
            side=side,
            price=price,
            amount=amount,
            **kwargs
        )

    def signal(
        self,
        signal_type: str,
        pair: str,
        value: float,
        threshold: float,
        **kwargs
    ) -> None:
        """Log a signal event."""
        self.debug(
            f"SIGNAL: {signal_type} for {pair}: {value} (threshold: {threshold})",
            signal_type=signal_type,
            pair=pair,
            value=value,
            threshold=threshold,
            **kwargs
        )

    def risk_event(self, event_type: str, details: str, **kwargs) -> None:
        """Log a risk management event."""
        self.warning(
            f"RISK: {event_type} - {details}",
            event_type=event_type,
            **kwargs
        )


# Logger registry
_loggers: dict[str, StructuredLogger] = {}
_configured = False


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    include_timestamps: bool = True
) -> None:
    """
    Configure the logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: "json" or "text".
        include_timestamps: Whether to include timestamps.
    """
    global _configured

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter based on format type
    if format_type.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # Add CloudWatch handler for AWS
    platform = detect_platform()
    if platform == Platform.AWS:
        _add_cloudwatch_handler(root_logger, level)

    _configured = True


def _add_cloudwatch_handler(logger: logging.Logger, level: str) -> None:
    """
    Add CloudWatch Logs handler for AWS deployment.

    Args:
        logger: Logger to add handler to.
        level: Log level.
    """
    try:
        import watchtower
        import boto3

        client = boto3.client("logs")
        handler = watchtower.CloudWatchLogHandler(
            log_group="kraken-momentum-trader",
            stream_name="trading-logs",
            boto3_client=client
        )
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    except ImportError:
        # watchtower not installed, skip CloudWatch
        pass
    except Exception as e:
        # Log error but continue without CloudWatch
        print(f"Failed to configure CloudWatch logging: {e}", file=sys.stderr)


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (typically module name).

    Returns:
        StructuredLogger instance.
    """
    global _configured

    if not _configured:
        configure_logging()

    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, logging.getLogger(name))

    return _loggers[name]
