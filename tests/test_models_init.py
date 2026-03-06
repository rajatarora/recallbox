from recallbox.models import __all__


def test_models_all_is_list():
    assert isinstance(__all__, list)


def test_settings_get_config_reflects_settings():
    from recallbox.settings import get_config, settings

    assert get_config() is settings
