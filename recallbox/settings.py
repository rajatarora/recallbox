from .conf.settings import Settings

# Backwards-compatible settings accessor kept for previous code.
settings = Settings()


def get_config() -> Settings:
    """Return the shared Settings instance (backwards-compatible).

    New code should use `recallbox.config.get_config()` for the application
    configuration defined in YAML; this function preserves the existing
    `settings` object used across the project.
    """
    return settings
