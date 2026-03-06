import pytest

import recallbox.services.db as dbmod


@pytest.mark.asyncio
async def test_get_session_depends_uses_get_session(monkeypatch):
    # Provide a fake get_session returning an async context manager
    def fake_get_session():
        class CM:
            async def __aenter__(self):
                return "sess"

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return CM()

    monkeypatch.setattr(dbmod, "get_session", fake_get_session)

    # Use get_session_depends as an async generator and get the yielded session
    agen = dbmod.get_session_depends()
    session = await agen.__anext__()
    assert session == "sess"
    # finalize generator
    try:
        await agen.__anext__()
    except StopAsyncIteration:
        pass


@pytest.mark.asyncio
async def test_test_data_raises_when_is_dev_set(monkeypatch):
    monkeypatch.setenv("IS_DEV", "1")
    with pytest.raises(ValueError):
        await dbmod.test_data(None)
