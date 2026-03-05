import os
from pathlib import Path
import pytest

from recallbox import config


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_successful_load(tmp_path, monkeypatch):
    cwd = tmp_path
    yaml = cwd / "config.yaml"
    write_yaml(
        yaml,
        """
embedding_model: "embed-1"
chat_model: "gpt-chat"
retrieval:
  top_k: 5
backup:
  enabled: true
""",
    )

    # Ensure working directory
    monkeypatch.chdir(cwd)

    # Ensure .env doesn't exist
    env = cwd / ".env"
    if env.exists():
        env.unlink()

    cfg = config.get_config()
    assert cfg.embedding_model == "embed-1"
    assert cfg.chat_model == "gpt-chat"
    assert cfg.retrieval.top_k == 5
    assert cfg.backup.enabled is True


def test_missing_required_key(tmp_path, monkeypatch):
    cwd = tmp_path
    yaml = cwd / "config.yaml"
    # missing chat_model
    write_yaml(
        yaml,
        """
embedding_model: "embed-1"
retrieval:
  top_k: 5
""",
    )

    monkeypatch.chdir(cwd)

    with pytest.raises(config.ConfigError):
        config.get_config()


def test_invalid_type_top_k(tmp_path, monkeypatch):
    cwd = tmp_path
    yaml = cwd / "config.yaml"
    write_yaml(
        yaml,
        """
embedding_model: "embed-1"
chat_model: "gpt-chat"
retrieval:
  top_k: "five"
""",
    )

    monkeypatch.chdir(cwd)

    with pytest.raises(config.ConfigError):
        config.get_config()


def test_secrets_missing_return_empty(tmp_path, monkeypatch):
    cwd = tmp_path
    yaml = cwd / "config.yaml"
    write_yaml(
        yaml,
        """
embedding_model: "embed-1"
chat_model: "gpt-chat"
""",
    )

    # Create .env but do not set secrets
    env = cwd / ".env"
    env.write_text("# empty env\n", encoding="utf-8")

    monkeypatch.chdir(cwd)
    # Ensure no secrets set in env
    for k in config._SECRET_NAMES:
        os.environ.pop(k, None)

    config.get_config()
    secrets = config._read_secrets_copy()
    # Missing secrets should be empty strings
    for name in config._SECRET_NAMES:
        assert secrets[name] == ""


def test_permissive_file_mode_logs_warning(tmp_path, monkeypatch, caplog):
    cwd = tmp_path
    yaml = cwd / "config.yaml"
    write_yaml(
        yaml,
        """
embedding_model: "embed-1"
chat_model: "gpt-chat"
""",
    )

    # Make file permissive
    yaml.chmod(0o644)

    monkeypatch.chdir(cwd)

    caplog.set_level("WARNING")
    # Ensure .env not present
    env = cwd / ".env"
    if env.exists():
        env.unlink()

    # Load config; should not raise but should emit a warning about insecure permissions
    config.get_config()
    assert any(
        "insecure_permissions" in rec.getMessage() or "insecure_permissions" in rec.message for rec in caplog.records
    )


def test_missing_env_not_error(tmp_path, monkeypatch):
    cwd = tmp_path
    yaml = cwd / "config.yaml"
    write_yaml(
        yaml,
        """
embedding_model: "embed-1"
chat_model: "gpt-chat"
""",
    )

    # ensure .env does not exist
    env = cwd / ".env"
    if env.exists():
        env.unlink()

    monkeypatch.chdir(cwd)

    cfg = config.get_config()
    assert cfg.embedding_model == "embed-1"
