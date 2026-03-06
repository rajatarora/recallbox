import pytest


from recallbox import cli


@pytest.mark.asyncio
async def test_test_data_calls_services(monkeypatch, capsys):
    # Provide a fake config with project_name
    class C:
        project_name = "proj"

    monkeypatch.setattr("recallbox.cli.get_config", lambda: C())

    called = {}

    async def fake_test_data(session):
        called["session"] = session

    def fake_get_session():
        class CM:
            async def __aenter__(self):
                return "sess"

            async def __aexit__(self, exc_type, exc, tb):
                return False

        # return an async context manager instance
        return CM()

    # Patch the services functions where cli imports them from inside the function
    import recallbox.services.db as dbmod

    # get_session should be a callable that returns an async context manager
    def fake_get_session_wrapper():
        return fake_get_session()

    monkeypatch.setattr(dbmod, "get_session", fake_get_session_wrapper)
    monkeypatch.setattr(dbmod, "test_data", fake_test_data)

    # Call the underlying async function (async wrapper preserved as __wrapped__)
    await cli.test_data.__wrapped__()

    # Ensure our fake was called
    assert called.get("session") == "sess"


@pytest.mark.asyncio
async def test_test_data_config_error_fallback(monkeypatch):
    # Make get_config raise ConfigError so CLI falls back to pydantic settings
    import recallbox.config as cfgmod

    monkeypatch.setattr(cfgmod, "get_config", lambda: (_ for _ in ()).throw(cfgmod.ConfigError()))

    async def fake_test_data(session):
        return None

    def fake_get_session():
        class CM:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return CM()

    import recallbox.services.db as dbmod

    def fake_get_session_wrapper2():
        return fake_get_session()

    monkeypatch.setattr(dbmod, "get_session", fake_get_session_wrapper2)
    monkeypatch.setattr(dbmod, "test_data", fake_test_data)

    await cli.test_data.__wrapped__()
