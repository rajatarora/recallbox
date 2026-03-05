import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import typer

from .settings import settings as _pydantic_settings
from .config import get_config, ConfigError

app = typer.Typer()


def syncify(f: Callable[..., Any]) -> Callable[..., Any]:
    """This simple decorator converts an async function into a sync function,
    allowing it to work with Typer.
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@app.command(help="Install testing data for local development.")
@syncify
async def test_data() -> None:
    from . import __version__
    from .services.db import get_session, test_data

    # Try to use YAML config as single source of truth; fall back to pydantic settings
    try:
        project = get_config().project_name
    except ConfigError:
        project = _pydantic_settings.project_name

    typer.echo(f"{project} - {__version__}")

    async with get_session() as session:
        await test_data(session)

    typer.echo("Development data installed successfully.")


@app.command(help=f"Display the current installed version of {_pydantic_settings.project_name}.")
def version() -> None:
    from . import __version__

    try:
        project = get_config().project_name
    except ConfigError:
        project = _pydantic_settings.project_name

    typer.echo(f"{project} - {__version__}")


@app.command(help="Display a friendly greeting.")
def hello() -> None:
    try:
        project = get_config().project_name
    except ConfigError:
        project = _pydantic_settings.project_name

    typer.echo(f"Hello from {project}!")


if __name__ == "__main__":
    app()
