import time
from pathlib import Path

from recallbox.services.watcher import FileWatcher
from recallbox.config import Config
from typing import cast
from recallbox.store.chromadb import MemoryStore
from recallbox.services.watcher import FileWatcherEventHandler
import watchdog.events


class DummyStore:
    def __init__(self):
        self.calls = []

    async def add_documents(self, docs):
        # record number of docs and first chunk content for assertions
        self.calls.append((len(docs), docs[0].content if docs else None))


def test_file_watcher_debounce(tmp_path: Path) -> None:
    """Create and rapidly modify a file and assert it's ingested only once."""
    # Prepare dummy store and config
    store = DummyStore()

    cfg = Config(
        embedding_model="embed-test",
        chat_model="chat-test",
        watcher_debounce_seconds=0.5,
    )

    # Ensure folder exists
    folder = tmp_path / "memories"
    folder.mkdir()

    watcher = FileWatcher(folder=folder, store=cast(MemoryStore, store), cfg=cfg)
    watcher.start()

    try:
        fpath = folder / "note.txt"
        # Create file
        fpath.write_text("Hello world\nThis is a test.")

        # Immediately modify the file (simulate editor saving twice)
        time.sleep(0.1)
        fpath.write_text("Hello world\nThis is a modified test.")

        # Wait longer than debounce to allow processing
        time.sleep(1.0)

        # The DummyStore should have been called exactly once due to debounce
        assert len(store.calls) == 1, f"Expected 1 ingest call, got {len(store.calls)}"
        num_chunks, first_chunk = store.calls[0]
        assert num_chunks >= 1
        assert isinstance(first_chunk, str)
    finally:
        watcher.stop()
        watcher.join(timeout=5)


def test_process_file_fallback_on_parse_exception(tmp_path: Path, monkeypatch) -> None:
    """If the parser raises, the watcher should fall back to plain-text parsing."""
    store = DummyStore()

    cfg = Config(
        embedding_model="embed-test",
        chat_model="chat-test",
        watcher_debounce_seconds=0.1,
    )

    folder = tmp_path / "memories"
    folder.mkdir()

    fpath = folder / "note.txt"
    content = "Fallback content here"
    fpath.write_text(content)

    # Monkeypatch the parser module to avoid importing the real parsers (which
    # may bring optional dependencies like pdfminer). Insert a fake module into
    # sys.modules with the expected attributes.
    import types
    import sys

    class _FP(Exception):
        pass

    def _raise_parse(path):
        raise RuntimeError("parser failed")

    fake_mod = types.SimpleNamespace(parse_file=_raise_parse, FileParseError=_FP)
    monkeypatch.setitem(sys.modules, "recallbox.utils.parsers", fake_mod)

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)

    # Call processing directly to avoid observer timing issues
    handler._process_file(fpath)

    assert len(store.calls) == 1
    assert store.calls[0][1].startswith("Fallback content")


def test_process_file_with_empty_documents_logs_and_skips(tmp_path: Path, monkeypatch) -> None:
    """If parser returns empty list, no ingestion occurs."""
    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.1)
    folder = tmp_path / "memories"
    folder.mkdir()
    fpath = folder / "empty.txt"
    fpath.write_text("nothing")

    # Insert a fake parser module returning an empty list
    import types
    import sys

    fake_mod = types.SimpleNamespace(parse_file=(lambda p: []), FileParseError=Exception)
    monkeypatch.setitem(sys.modules, "recallbox.utils.parsers", fake_mod)

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)
    handler._process_file(fpath)

    assert len(store.calls) == 0


def test_schedule_ignores_outside_folder(tmp_path: Path) -> None:
    """Files outside the watched folder are ignored by schedule_processing."""
    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.1)

    folder = tmp_path / "memories"
    folder.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    outside = other / "outside.txt"
    outside.write_text("hi")

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)

    ev = watchdog.events.FileCreatedEvent(str(outside))
    handler._schedule_processing(ev)

    # No timer should have been scheduled for the outside path
    assert str(outside) not in handler._timers


def test_on_moved_triggers_processing(tmp_path: Path, monkeypatch) -> None:
    """A moved event should be treated as a creation for the destination path."""
    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.01)

    folder = tmp_path / "memories"
    folder.mkdir()
    src = folder / "src.txt"
    dest = folder / "dest.txt"
    dest.write_text("moved content")

    # Ensure parser returns a document so store gets called
    def _parse(path):
        from recallbox.store.chromadb import Document

        return [Document(content=path.read_text(), metadata={})]

    import types
    import sys

    fake_mod = types.SimpleNamespace(parse_file=_parse, FileParseError=Exception)
    monkeypatch.setitem(sys.modules, "recallbox.utils.parsers", fake_mod)

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)

    ev = watchdog.events.FileMovedEvent(str(src), str(dest))
    handler.on_moved(ev)

    # Wait a short moment for the debounce timer to fire
    time.sleep(0.05)

    assert len(store.calls) == 1


def test_process_non_file_ignored(tmp_path: Path) -> None:
    """Calling _process_file on a non-file path does nothing."""
    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.1)
    folder = tmp_path / "memories"
    folder.mkdir()

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)

    non_file = folder / "nofile"
    # Path doesn't exist
    handler._process_file(non_file)

    assert len(store.calls) == 0


def test_store_add_documents_raises_fileparseerror_caught(tmp_path: Path, monkeypatch, caplog) -> None:
    """If store.add_documents raises FileParseError it's caught and logged."""
    import sys
    import types
    import logging
    import recallbox.services.watcher as watcher_mod

    caplog.set_level(logging.ERROR)
    store = DummyStore()

    async def _raise(docs):
        raise watcher_mod.FileParseError(Path("/no/such"), RuntimeError("boom"))

    # Replace add_documents with raising coroutine
    store.add_documents = _raise  # type: ignore

    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.1)
    folder = tmp_path / "memories"
    folder.mkdir()
    fpath = folder / "file.txt"
    fpath.write_text("ok")

    # Provide a fake parser module that returns a document
    fake_mod = types.SimpleNamespace(
        parse_file=(lambda p: [watcher_mod.Document(content=p.read_text(), metadata={})]), FileParseError=Exception
    )
    monkeypatch.setitem(sys.modules, "recallbox.utils.parsers", fake_mod)

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)
    handler._process_file(fpath)

    # Ensure the FileParseError was logged
    assert any("Failed to parse file" in rec.getMessage() for rec in caplog.records)


def test_schedule_cancel_exception_logs(tmp_path: Path, caplog) -> None:
    """If cancelling an existing debounce timer raises, we log a debug message but continue."""
    import logging

    caplog.set_level(logging.DEBUG)
    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.1)
    folder = tmp_path / "memories"
    folder.mkdir()
    fpath = folder / "foo.txt"
    fpath.write_text("x")

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)

    # Insert a fake timer that raises on cancel
    class BadTimer:
        def cancel(self):
            raise RuntimeError("cancel failed")

    handler._timers[str(fpath)] = BadTimer()

    ev = watchdog.events.FileModifiedEvent(str(fpath))
    handler._schedule_processing(ev)

    assert any("Existing debounce timer cancel failed" in rec.getMessage() for rec in caplog.records)


def test_on_created_modified_filters(tmp_path: Path) -> None:
    """Ensure on_created/on_modified only schedule for the specific event subtypes."""
    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=1.0)
    folder = tmp_path / "memories"
    folder.mkdir()
    fpath = folder / "a.txt"
    fpath.write_text("1")

    handler = FileWatcherEventHandler(folder=folder, store=cast(MemoryStore, store), cfg=cfg, debounce_interval=0.01)

    # Generic FileSystemEvent should not schedule
    base_ev = watchdog.events.FileSystemEvent(str(fpath))
    handler.on_created(base_ev)
    handler.on_modified(base_ev)
    assert len(handler._timers) == 0

    # Specific events should schedule
    handler.on_created(watchdog.events.FileCreatedEvent(str(fpath)))
    handler.on_modified(watchdog.events.FileModifiedEvent(str(fpath)))
    # Timer should be present for the file path
    assert str(fpath) in handler._timers


def test_filewatcher_run_observer_join_warning(tmp_path: Path, monkeypatch, caplog) -> None:
    """Start FileWatcher.run with a fake Observer whose is_alive stays True so we log a warning on shutdown."""
    import recallbox.services.watcher as watcher_mod

    caplog.set_level(30)  # WARNING

    # Fake observer class
    class FakeObserver:
        def __init__(self):
            self._scheduled = False

        def schedule(self, handler, path, recursive=False):
            self._scheduled = True

        def start(self):
            return None

        def stop(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return True

    monkeypatch.setattr(watcher_mod.watchdog.observers, "Observer", lambda: FakeObserver())

    store = DummyStore()
    cfg = Config(embedding_model="e", chat_model="c", watcher_debounce_seconds=0.01)
    folder = tmp_path / "memories"
    folder.mkdir()

    watcher = FileWatcher(folder=folder, store=cast(MemoryStore, store), cfg=cfg)
    watcher.start()
    # Give thread time to start
    time.sleep(0.05)
    watcher.stop()
    watcher.join(timeout=2)

    assert any("did not shut down gracefully" in rec.getMessage() for rec in caplog.records)
