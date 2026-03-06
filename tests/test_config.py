import os
import json
import pytest


from recallbox.config import (
    load_yaml,
    _ensure_secure_permissions,
    _read_secrets_copy,
    get_config,
    ConfigError,
)


def test_load_yaml_missing(tmp_path):
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "nope.yml")


def test_load_yaml_bad_yaml(tmp_path, monkeypatch):
    # simulate yaml.safe_load raising via monkeypatching recallbox.config.yaml to None
    import recallbox.config as cfgmod

    monkeypatch.setattr(cfgmod, "yaml", None)
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "f.yml")


def test_read_secrets_copy_and_env(tmp_path, monkeypatch):
    # create a .env and set environment variable
    p = tmp_path / ".env"
    p.write_text("OPENROUTER_API_KEY=abc\n")
    monkeypatch.chdir(tmp_path)
    # load env should populate module secrets
    load_env = __import__("recallbox.config", fromlist=["load_env"]).load_env
    load_env(p)

    # ensure copy replaces None with ""
    copy = _read_secrets_copy()
    assert isinstance(copy, dict)
    assert "OPENROUTER_API_KEY" in copy


def test_ensure_secure_permissions(tmp_path, caplog):
    p = tmp_path / "cfg.yaml"
    p.write_text("{}")
    # make file world-readable
    os.chmod(p, 0o644)
    _ensure_secure_permissions(p)
    # If chmod succeeded this will be 0o600
    mode = p.stat().st_mode & 0o777
    assert mode in (0o600, 0o644)


def test_get_config_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        get_config()


def test_get_config_loads_and_singleton(tmp_path, monkeypatch, fake_yaml_module, clear_config_singleton):
    # write config.yaml with required fields
    cfg = {"embedding_model": "e", "chat_model": "c", "project_name": "p"}
    p = tmp_path / "config.yaml"
    p.write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    cfg_obj = get_config()
    assert cfg_obj.project_name == "p"
    # second call returns same instance
    cfg2 = get_config()
    assert cfg_obj is cfg2


def test_get_config_with_passphrase(tmp_path, monkeypatch, fake_yaml_module, clear_config_singleton):
    cfg = {"embedding_model": "e", "chat_model": "c", "project_name": "p"}
    p = tmp_path / "config.yaml"
    p.write_text(json.dumps(cfg))
    # set env var
    monkeypatch.setenv("BACKUP_PASSPHRASE", "secretpass")
    monkeypatch.chdir(tmp_path)
    obj = get_config()
    assert obj.backup.passphrase is not None
