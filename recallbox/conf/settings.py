from .db import DatabaseSettings


class Settings(DatabaseSettings):
    project_name: str = "recallbox"
    debug: bool = False
