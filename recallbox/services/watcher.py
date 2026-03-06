from __future__ import annotations
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any
from datetime import datetime

import watchdog.events
import watchdog.observers

from recallbox.config import Config
from recallbox.store.chromadb import Document, MemoryStore

logger = logging.getLogger(__name__)


class FileParseError(RuntimeError):
    """Raised when a file cannot be parsed into text.

    The original exception is stored as ``__cause__`` so callers can inspect the
    underlying problem.
    """

    def __init__(self, path: Path, original: Exception) -> None:
        super().__init__(f"Failed to parse file {path}: {original}")
        self.path = path
        self.original = original


# Import parsers lazily in _process_file to avoid hard import-time dependency
# errors when optional libs (pdfminer/bs4/etc.) are missing in the test env.


class FileWatcherEventHandler(watchdog.events.FileSystemEventHandler):
    """A watchdog event handler that processes file events with debouncing."""

    def __init__(
        self,
        folder: Path,
        store: MemoryStore,
        cfg: Config,
        debounce_interval: float = 2.0,
    ):
        super().__init__()
        self._folder = folder
        self._store = store
        self._cfg = cfg
        self._debounce_interval = debounce_interval
        self._timers: dict[str, threading.Timer] = {}
        self._processing_lock = threading.Lock()

    def _process_file(self, file_path: Path) -> None:
        """Process a single file: parse and store."""
        if not file_path.is_file():
            return

        try:
            # Use a lock to prevent concurrent processing of the same file if events are rapid
            with self._processing_lock:
                # Import parse_file here to avoid optional dependency errors at import time
                try:
                    from recallbox.utils.parsers import parse_file, FileParseError

                    documents = parse_file(file_path)
                except Exception as exc:
                    # If optional parser dependencies are missing (pdfminer/bs4/etc.)
                    # fall back to a simple text-only parser so the watcher still
                    # ingests plain text files. Log the import error and proceed.
                    logger.debug("Parser import or parse failed (%s), falling back to plain-text parser", exc)
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        # Could not read file; nothing to do
                        logger.exception("Failed to read file during fallback parsing: %s", file_path)
                        return

                    now = datetime.utcnow()
                    metadata = {
                        "source": "file_watcher",
                        "file_path": str(file_path),
                        "timestamp": now.isoformat(),
                        "importance": 3,
                        "chunk_index": 0,
                    }
                    documents = [Document(content=text, metadata=metadata)]
                if not documents:
                    logger.warning("No documents parsed from file: %s", file_path, extra={"component": "file_watcher"})
                    return

                # Use asyncio.run to execute the async store method in a sync context
                # This is generally safe for standalone threads like this watcher.
                asyncio.run(self._store.add_documents(documents))

                logger.info(
                    "Ingested %d chunks from file: %s",
                    len(documents),
                    file_path,
                    extra={"component": "file_watcher"},
                )
        except FileParseError as e:
            logger.exception(
                "Failed to parse file %s: %s",
                file_path,
                e.original,
                extra={"component": "file_watcher"},
            )
        except Exception:
            logger.exception(
                "Unexpected error processing file %s",
                file_path,
                extra={"component": "file_watcher"},
            )

    def _schedule_processing(self, event: watchdog.events.FileModifiedEvent | watchdog.events.FileCreatedEvent) -> None:
        """Schedule file processing with debouncing."""
        file_path = Path(str(event.src_path))

        # Ignore directories and files not within the watched folder
        # Use os.path.realpath to resolve any symlinks and ensure accurate path comparison
        real_file_path = Path(os.path.realpath(str(file_path)))
        real_folder_path = Path(os.path.realpath(str(self._folder)))

        if not file_path.is_file() or not str(real_file_path).startswith(str(real_folder_path)):
            return

        # Cancel any existing timer for this path
        timer_key = str(file_path)  # Use string representation as dict key
        if timer_key in self._timers:
            try:
                self._timers[timer_key].cancel()
            except Exception:
                logger.debug("Existing debounce timer cancel failed for %s", timer_key)

        # Create and schedule a new timer
        timer = threading.Timer(self._debounce_interval, lambda: self._process_file(file_path))
        self._timers[timer_key] = timer
        timer.daemon = True
        timer.start()

    # Adjusting event handler methods to accept broader event types as per watchdog's base class
    # and then filtering/casting internally if needed.

    def on_created(self, event: watchdog.events.FileSystemEvent) -> None:
        logger.debug("File system event (created): %s", event.src_path)
        if isinstance(event, watchdog.events.FileCreatedEvent):
            self._schedule_processing(event)

    def on_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        logger.debug("File system event (modified): %s", event.src_path)
        if isinstance(event, watchdog.events.FileModifiedEvent):
            self._schedule_processing(event)

    def on_deleted(self, event: watchdog.events.FileSystemEvent) -> None:
        # We don't need to handle deletions for ingestion
        logger.debug("File system event (deleted): %s", event.src_path)

    def on_moved(self, event: watchdog.events.FileSystemEvent) -> None:
        logger.debug("File system event (moved): from %s to %s", event.src_path, event.dest_path)
        # Create a FileCreatedEvent for the destination path to trigger processing
        if isinstance(event, watchdog.events.FileMovedEvent):
            # Ensure the destination is treated as a creation event
            self._schedule_processing(watchdog.events.FileCreatedEvent(event.dest_path))


class FileWatcher(threading.Thread):
    """A threaded service that watches a folder for new/modified files and ingests them."""

    def __init__(
        self,
        folder: Path,
        store: MemoryStore,
        cfg: Config,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._store = store
        self._cfg = cfg
        self._stop_event = threading.Event()
        # Observer type is dynamic; use Any to satisfy static type checkers
        self._observer: Any = None

    def run(self) -> None:
        """Start the file watcher observer."""
        logger.info("Starting file watcher on folder: %s", self._folder)
        self._observer = watchdog.observers.Observer()
        # Accessing watcher_debounce_seconds directly from cfg
        event_handler = FileWatcherEventHandler(
            self._folder, self._store, self._cfg, self._cfg.watcher_debounce_seconds
        )
        # Schedule the observer on the target folder, recursively
        # Ensure the path is correctly converted to a string for the observer
        self._observer.schedule(event_handler, str(self._folder), recursive=True)
        self._observer.start()

        # Keep the thread alive until stop_event is set
        while not self._stop_event.is_set():
            try:
                # Use wait with a timeout to periodically check the stop event
                # and allow other threads/asyncio tasks to run.
                self._stop_event.wait(timeout=1.0)
            except asyncio.CancelledError:
                # This might happen if the watcher is run within an asyncio loop
                logger.warning("File watcher run loop was cancelled.")
                break
            except Exception:
                logger.exception("Unexpected error in file watcher run loop.")
                break  # Exit loop on unexpected error

        logger.info("File watcher stopping.")
        if self._observer:
            self._observer.stop()
            # Respect the timeout for observer.join
            join_timeout = 5.0
            try:
                self._observer.join(timeout=join_timeout)
                if self._observer.is_alive():
                    logger.warning(
                        "File watcher observer did not shut down gracefully within %s seconds.", join_timeout
                    )
            except Exception:
                logger.exception("Error during observer join.")

    def stop(self) -> None:
        """Signal the watcher to stop and join the observer thread."""
        logger.info("Stopping file watcher...")
        self._stop_event.set()
