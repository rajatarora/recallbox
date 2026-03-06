"""Configuration loader for recallbox.

Loads a single-source-of-truth `config.yaml` from the current working
directory and an optional `.env` file. Validates the merged configuration
with pydantic and exposes a singleton `get_config()` accessor.

Public API: `Config`, `get_config`
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, SecretStr, ValidationError
from dotenv import load_dotenv

import importlib
import importlib.util


# Dynamically import yaml so mypy doesn't require stubs in the environment.
def _dynamic_import(name: str) -> Any | None:
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


yaml = _dynamic_import("yaml")

logger = logging.getLogger(__name__)

__all__ = ["Config", "get_config"]


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


class RetrievalSettings(BaseModel):
    top_k: int = Field(default=10, description="Number of results to retrieve")


class BackupSettings(BaseModel):
    enabled: bool = Field(default=False)
    # Stored as SecretStr when read from env; YAML may supply empty/default
    passphrase: SecretStr | None = Field(default=None)


class Config(BaseModel):
    # Ingestion settings
    max_chunk_size: int = Field(default=1024, description="Maximum characters per chunk")
    chunk_overlap: int = Field(default=200, description="Number of overlapping characters between chunks")
    # Project metadata
    project_name: str = Field(default="recallbox")
    debug: bool = Field(default=False)
    embedding_model: str
    chat_model: str
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter API",
    )
    retrieval: RetrievalSettings = RetrievalSettings()
    backup: BackupSettings = BackupSettings()


# Module-level singleton
_CONFIG_INSTANCE: Config | None = None
# Path of the config file used to produce the current _CONFIG_INSTANCE
_LOADED_CONFIG_PATH: Path | None = None

# Private secrets cache (read once)
_SECRETS: Dict[str, str | None] = {}

# Secrets we care about (ENV names)
_SECRET_NAMES = [
    "OPENROUTER_API_KEY",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "BACKUP_PASSPHRASE",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from `path` and return a dict.

    Raises ConfigError if file is missing or cannot be parsed.
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    if yaml is None:
        raise ConfigError("pyyaml not installed; cannot load config.yaml")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ConfigError("config.yaml must contain a mapping at top level")
            return data
    except Exception as exc:
        # yaml may raise YAMLError, but keep a generic except to avoid type errors
        raise ConfigError(f"failed to parse YAML: {exc}") from exc


def load_env(env_path: Path | None = None) -> None:
    """Load environment from `.env` (if present) and cache secret values.

    This function reads `.env` once and stores required secrets into
    a private module-level dict. Missing `.env` is not an error.
    """
    # If env_path is supplied, attempt to load it; otherwise load default
    try:
        if env_path is not None and env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)
        else:
            # load default .env if present
            load_dotenv(override=False)
    except Exception:
        # Do not expose failure details; raise a ConfigError
        raise ConfigError("failed to load .env file")

    # Read secrets once and keep them private
    for name in _SECRET_NAMES:
        # Distinguish between unset and empty: os.environ.get returns None if missing
        val = os.environ.get(name)
        _SECRETS[name] = val


def _ensure_secure_permissions(path: Path) -> None:
    """Ensure `path` has mode 0o600. If not, attempt chmod; on failure log a warning.

    The function logs a JSON-structured warning if the file is more permissive
    and chmod cannot be applied. It does not raise for permissive modes.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return

    mode = st.st_mode & 0o777
    desired = 0o600
    if mode != desired:
        # Log a structured warning about permissive mode (do not include secrets)
        logger.warning(
            json.dumps({"component": "config", "path": str(path), "issue": "insecure_permissions", "mode": oct(mode)})
        )
        # Try to set secure permissions, but failure is non-fatal
        try:
            os.chmod(path, desired)
        except PermissionError:
            logger.warning(
                json.dumps(
                    {"component": "config", "path": str(path), "issue": "chmod_permission_error", "mode": oct(mode)}
                )
            )
        except Exception:
            logger.warning(
                json.dumps({"component": "config", "path": str(path), "issue": "chmod_failed", "mode": oct(mode)})
            )


def _read_secrets_copy() -> Dict[str, str]:
    """Return a copy of the secrets mapping where missing values become empty string.

    This helper is intended for code that needs secrets without exposing whether
    they were set or not; tests may inspect the Behavior.
    """
    return {k: (v if v is not None else "") for k, v in _SECRETS.items()}


def get_config() -> Config:
    """Return the singleton Config instance, loading and validating on first call."""
    global _CONFIG_INSTANCE
    global _LOADED_CONFIG_PATH
    cwd = Path(os.getcwd())
    config_path = cwd / "config.yaml"

    # If we've already loaded a config from the same path, return it
    if _CONFIG_INSTANCE is not None and _LOADED_CONFIG_PATH == config_path:
        return _CONFIG_INSTANCE

    env_path = cwd / ".env"

    # Ensure config.yaml exists (ACCEPTANCE: missing -> readable error)
    if not config_path.exists():
        raise ConfigError(f"config.yaml not found in working directory: {cwd}")

    # Load env and secrets
    load_env(env_path if env_path.exists() else None)

    # Permission checks
    _ensure_secure_permissions(config_path)
    # .env is optional, but if present check permissions
    if env_path.exists():
        _ensure_secure_permissions(env_path)

    # Load YAML
    raw = load_yaml(config_path)

    # Validate via pydantic
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as exc:
        # Re-raise a ConfigError with readable message
        raise ConfigError(f"configuration validation error: {exc}") from exc

    # If backup passphrase is present in secrets, put it into model
    # Do not log secrets
    secrets = _read_secrets_copy()
    bp = secrets.get("BACKUP_PASSPHRASE") or None
    if bp:
        cfg.backup.passphrase = SecretStr(bp)

    _CONFIG_INSTANCE = cfg
    _LOADED_CONFIG_PATH = config_path
    return _CONFIG_INSTANCE
