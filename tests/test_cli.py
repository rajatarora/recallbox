from recallbox.cli import syncify


def test_syncify_runs():
    async def afunc(x):
        return x + 1

    sf = syncify(afunc)
    assert sf(1) == 2


def test_version_and_hello(monkeypatch, capsys):
    # Monkeypatch get_config to raise ConfigError so fallback to pydantic settings
    import recallbox.config as configmod

    def fake_get_config():
        raise configmod.ConfigError()

    monkeypatch.setattr("recallbox.cli.get_config", fake_get_config)
    # Capture typer output by invoking the functions directly
    from recallbox.cli import version, hello

    version()
    hello()
    captured = capsys.readouterr()
    assert "-" in captured.out
