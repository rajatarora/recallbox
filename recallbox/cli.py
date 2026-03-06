import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import typer

from .settings import settings as _pydantic_settings
from .config import get_config, ConfigError
import logging

logger = logging.getLogger(__name__)

app = typer.Typer()


# Typer doesn't support async commands directly. This decorator allows us to
# run an async function within a Typer command. It also handles basic error
# logging for the command.
def syncify(f: Callable[..., Any]) -> Callable[..., Any]:
    """This simple decorator converts an async function into a sync function,
    allowing it to work with Typer.
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # If the function is already running in an asyncio event loop,
        # we can just await it directly. Otherwise, create a new loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If loop is running, we cannot use asyncio.run()
            # instead, create a task and wait for it.
            # This is a simplified approach and might block the event loop
            # if the task takes too long. For production, consider more robust
            # async task management.
            future = asyncio.ensure_future(f(*args, **kwargs))
            # If we are in an existing loop, we must block until completion.
            # This is not ideal but necessary for Typer integration.
            while not future.done():
                loop.run_until_complete(asyncio.sleep(0.1))
            return future.result()
        else:
            # If no loop is running, create a new one.
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


# Command to enable the file watcher service
@app.command(help="Start the file watcher service.")
@syncify
async def watch() -> None:
    """Starts the file watcher service."""
    from .config import ConfigError, get_config
    from .store.chromadb import MemoryStore
    from .services.watcher import FileWatcher

    try:
        cfg = get_config()
    except ConfigError as e:
        typer.echo(f"Error loading configuration: {e}", err=True)
        raise typer.Exit(code=1)

    if not cfg.enable_file_watcher:
        typer.echo("File watcher is not enabled. Set 'enable_file_watcher = true' in config.yaml.")
        raise typer.Exit(code=0)

    # Initialize dependencies needed for MemoryStore and FileWatcher
    # This part might need adjustment based on how MemoryStore is truly initialized
    # For now, assuming a simple init is sufficient for the watcher's context.
    try:
        # A basic session might be needed, or MemoryStore might be self-contained.
        # If MemoryStore requires a session, it should be managed here.
        # For simplicity, let's assume MemoryStore can be initialized directly.
        # If not, we'd need to adjust this part.

        # Mocking dependencies for now, as their actual initialization is complex
        # and depends on other parts of the application not fully defined here.
        # Replace these with actual initializations if MemoryStore and OpenRouterClient
        # have specific requirements.

        # Placeholder for a real session if needed:
        # async with get_session() as session:
        #    store = MemoryStore(cfg.memories_folder, OpenRouterClient(), session=session)

        # Assuming MemoryStore and OpenRouterClient can be initialized directly
        # This might need adjustment based on actual implementation details.

        # Create a lightweight embed client stub implementing `embed` for MemoryStore
        # Use typing.cast to satisfy the MemoryStore type expectation for mypy
        # Construct a real OpenRouterClient using secure env-configured API key.
        # The API key is read from the environment via config's secrets cache.
        from recallbox.llm import OpenRouterClient
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            typer.echo("OPENROUTER_API_KEY is not set in the environment. Set it to run the watcher.", err=True)
            raise typer.Exit(code=1)

        # Use the memory_prompt_path from config for prompt-based evaluations.
        openrouter = OpenRouterClient(
            api_key=api_key,
            embedding_model=cfg.embedding_model,
            chat_model=cfg.chat_model,
            memory_prompt_path=str(cfg.memory_prompt_path),
            base_url=cfg.openrouter_base_url,
            config=cfg,
        )

        store = MemoryStore(cfg.memories_folder, openrouter)

        watcher = FileWatcher(
            folder=cfg.memories_folder,
            store=store,
            cfg=cfg,
        )
    except Exception as e:
        logger.exception("Failed to initialize FileWatcher dependencies.")
        typer.echo(f"Error initializing watcher: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Starting file watcher on '{cfg.memories_folder}'...")
    watcher.start()

    # Keep the main thread alive to allow the watcher thread to run
    try:
        while True:
            # Check if watcher thread is alive; exit if it died unexpectedly
            if not watcher.is_alive():
                typer.echo("File watcher thread terminated unexpectedly.", err=True)
                raise typer.Exit(code=1)
            await asyncio.sleep(1)  # Sleep to prevent busy-waiting
    except KeyboardInterrupt:
        typer.echo("\nStopping file watcher...")
        watcher.stop()
        watcher.join()  # Wait for the watcher thread to finish
        typer.echo("File watcher stopped.")
    except Exception as e:
        logger.exception("An error occurred during watcher execution.")
        watcher.stop()
        watcher.join()
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
