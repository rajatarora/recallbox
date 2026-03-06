import types

import pytest
from fastapi.testclient import TestClient
from recallbox.www import app as fastapi_app
from recallbox.services.db import get_session as get_session_depends


@pytest.fixture
def no_sleep(monkeypatch):
    async def _noop_sleep(_):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    yield


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Change cwd to a temporary directory for tests working with files."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def fastapi_client(monkeypatch, tmp_path):
    """Provide a TestClient for the FastAPI app, overriding DB dependency to avoid real DB."""
    # simple TestClient; don't try to start backend services
    client = TestClient(fastapi_app)

    async def fake_session_dep():
        # Minimal async generator to satisfy dependency override
        if False:
            yield None

    fastapi_app.dependency_overrides[get_session_depends] = fake_session_dep
    return client


@pytest.fixture
def fake_yaml_module(monkeypatch):
    """Provide a fake yaml-like module with safe_load. Tests can adjust behavior by
    setting attributes on the returned types.SimpleNamespace.safe_load.
    """
    ns = types.SimpleNamespace()

    def safe_load(f):
        # Accept either a file handle or string content
        if hasattr(f, "read"):
            text = f.read()
        else:
            text = str(f)
        # Very small YAML-ish loader for our tests: use JSON for simplicity
        import json

        return json.loads(text)

    ns.safe_load = safe_load
    monkeypatch.setattr("recallbox.config.yaml", ns)
    return ns


@pytest.fixture
def clear_config_singleton(monkeypatch):
    import recallbox.config as configmod

    # Reset module-level singletons
    monkeypatch.setattr(configmod, "_CONFIG_INSTANCE", None)
    monkeypatch.setattr(configmod, "_LOADED_CONFIG_PATH", None)
    # Clear secrets mapping
    configmod._SECRETS.clear()
    yield configmod
