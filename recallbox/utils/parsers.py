"""Utilities for parsing various file types into semantic chunks.

This module provides a thin wrapper around common file‑type libraries and a
sentence‑aware chunker that respects a maximum character length and an
overlap between consecutive chunks.

The public API consists of:
- :class:`FileParseError`
- :func:`parse_file`
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Callable, cast
import importlib
import importlib.util
from recallbox.store.chromadb import Document
from recallbox.config import get_config


# Dynamically import optional dependencies so mypy doesn't require stubs.
def _dynamic_import(name: str) -> Any | None:
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


magic = _dynamic_import("magic")
nltk = _dynamic_import("nltk")

# bs4 provides BeautifulSoup in package 'bs4'
bs4_mod = _dynamic_import("bs4")
BeautifulSoup = getattr(bs4_mod, "BeautifulSoup", None) if bs4_mod is not None else None

md_mod = _dynamic_import("markdown")
markdown: Callable[[str], str] | None = getattr(md_mod, "markdown", None) if md_mod is not None else None

pdfmod = _dynamic_import("pdfminer.high_level")
pdf_extract_text: Callable[[str], str] | None = getattr(pdfmod, "extract_text", None) if pdfmod is not None else None


# Ensure the NLTK Punkt tokenizer data is available when nltk is present.
# If nltk is not installed at type-check or runtime we fall back to a simple
# sentence splitter.
if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover – exercised in CI when missing
        nltk.download("punkt", quiet=True)

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


def _detect_mime(path: Path) -> Optional[str]:
    """Return a supported MIME type string or ``None``.

    Supported returns are ``"text/plain"``, ``"text/html"``, ``"text/markdown"`` and
    ``"application/pdf"``.  Any other type results in ``None``.
    """
    if magic is None:
        logger.debug("python-magic not available; cannot detect MIME for %s", path)
        return None
    try:
        mime = magic.from_file(str(path), mime=True)
    except Exception:  # pragma: no cover – unlikely but defensive
        logger.exception("MIME detection failed for %s", path)
        return None

    if mime.startswith("text/plain"):
        return "text/plain"
    if mime.startswith("text/html"):
        return "text/html"
    if mime.startswith("text/markdown") or mime.startswith("text/x-markdown"):
        return "text/markdown"
    if mime == "application/pdf":
        return "application/pdf"
    logger.warning("Unsupported MIME type %s for %s", mime, path)
    return None


def _pdf_to_text(path: Path) -> str:
    """Extract plain text from a PDF file.

    ``pdfminer.six`` keeps line breaks, which is suitable for chunking.
    """
    if pdf_extract_text is None:
        raise FileParseError(path, RuntimeError("pdfminer.six not installed"))
    try:
        # pdf_extract_text returns a str on success
        result = pdf_extract_text(str(path))
        if not isinstance(result, str):
            raise FileParseError(path, RuntimeError("pdf extraction returned non-string"))
        return result
    except Exception as exc:
        raise FileParseError(path, exc) from exc


def _strip_markup(text: str, is_markdown: bool = False) -> str:
    """Strip HTML or Markdown markup, returning plain text.

    For Markdown we first convert it to HTML using :pypi:`markdown` and then strip
    the HTML with ``BeautifulSoup``.  ``separator="\n"`` preserves line breaks.
    """
    if is_markdown and markdown is None:
        raise RuntimeError("markdown library not installed")
    if BeautifulSoup is None:
        raise RuntimeError("bs4 (BeautifulSoup) not installed")
    try:
        if is_markdown:
            # Convert markdown to HTML first. Use a local ref so mypy can narrow None.
            md = markdown
            if md is None:
                raise RuntimeError("markdown library not installed")
            html = md(text)
        else:
            html = text
        soup = BeautifulSoup(html, "html.parser")
        # get_text with newline separator keeps paragraph boundaries.
        return cast(str, soup.get_text(separator="\n"))
    except Exception as exc:
        raise RuntimeError("Markup stripping failed") from exc


def _chunk_text(text: str, max_len: int, overlap: int) -> List[str]:
    """Split *text* into chunks respecting *max_len* and *overlap*.

    The function tokenises the input into sentences using NLTK's
    ``PunktSentenceTokenizer`` (English).  Sentences are accumulated until the
    length would exceed ``max_len``.  When a chunk reaches the limit, the last
    ``overlap`` characters of that chunk are prefixed to the next chunk.

    Edge cases:
    * If a single sentence exceeds ``max_len`` it is split on a character basis.
    * ``overlap`` is clamped to ``max_len`` to avoid infinite loops.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    # Ensure overlap never exceeds max_len.
    overlap = min(overlap, max_len)

    # Use NLTK sentence tokenizer when available; otherwise fall back to
    # a simple newline/sentence split which is less accurate but safe.
    if nltk is not None:
        # mypy can't see nltk's attributes; use getattr to access tokenizer
        punkt_cls = getattr(nltk.tokenize, "PunktSentenceTokenizer", None)
        if punkt_cls is not None:
            tokenizer = punkt_cls(lang="english")
            sentences = tokenizer.tokenize(text)
        else:
            # Unexpected: fallback to simple split
            import re

            sentences = [s.strip() for s in re.split(r"\n+|(?<=\.)\s+", text) if s.strip()]
    else:
        # Fallback: split on newlines and periods followed by space.
        import re

        sentences = [s.strip() for s in re.split(r"\n+|(?<=\.)\s+", text) if s.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_chunk(extra_prefix: str = "") -> None:
        """Combine *current* into a chunk, apply *extra_prefix* and store it."""
        if not current:
            return
        chunk = "".join(current)
        if extra_prefix:
            chunk = extra_prefix + chunk
        chunks.append(chunk)

    for sent in sentences:
        sent_len = len(sent)
        # If a single sentence is longer than max_len we need to split it.
        if sent_len > max_len:
            # Flush any accumulated content first.
            if current:
                # Add overlap from previous chunk if any.
                prev_overlap = chunks[-1][-overlap:] if chunks else ""
                flush_chunk(prev_overlap)
                current = []
                current_len = 0
            # Slice the long sentence.
            start = 0
            while start < sent_len:
                end = min(start + max_len, sent_len)
                slice_part = sent[start:end]
                # Determine overlap prefix based on previous chunk.
                prefix = ""
                if chunks:
                    prefix = chunks[-1][-overlap:]
                chunks.append(prefix + slice_part)
                start = end
            continue

        # Normal sentence handling.
        if current_len + sent_len > max_len:
            # Flush the current buffer.
            prev_overlap = chunks[-1][-overlap:] if chunks else ""
            flush_chunk(prev_overlap)
            current = []
            current_len = 0
        current.append(sent)
        current_len += sent_len

    # Flush any remaining content.
    if current:
        prev_overlap = chunks[-1][-overlap:] if chunks else ""
        flush_chunk(prev_overlap)

    # Ensure no empty chunks are produced.
    return [c for c in chunks if c]


def parse_file(path: Path) -> List[Document]:
    """Parse *path* into a list of :class:`~recallbox.store.chromadb.Document`.

    The function determines the file type, extracts raw text, strips markup if
    needed, chunks the text according to the global configuration and returns a
    ``Document`` for each chunk.  Metadata fields follow the acceptance
    criteria.
    """
    if not path.is_file():
        raise FileParseError(path, FileNotFoundError("Path does not exist or is not a file"))

    mime = _detect_mime(path)
    if mime is None:
        # Unsupported type – caller decides how to handle; we return empty list.
        logger.warning("Unsupported file type for %s", path)
        return []

    try:
        if mime == "application/pdf":
            raw_text = _pdf_to_text(path)
        else:
            # For text, html, markdown read as UTF‑8 with replacement errors.
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                raw_text = fh.read()

        # Strip markup for html / markdown.
        if mime == "text/html":
            raw_text = _strip_markup(raw_text, is_markdown=False)
        elif mime == "text/markdown":
            raw_text = _strip_markup(raw_text, is_markdown=True)
        # plain text needs no stripping.

    except Exception as exc:
        raise FileParseError(path, exc) from exc

    # Configuration for chunking.
    cfg = get_config()
    max_len = getattr(cfg, "max_chunk_size", 1024)
    overlap = getattr(cfg, "chunk_overlap", 200)

    chunks = _chunk_text(raw_text, max_len, overlap)

    now = datetime.utcnow()
    documents: List[Document] = []
    for idx, chunk in enumerate(chunks):
        metadata = {
            "source": "file_watcher",
            "file_path": str(path),
            "timestamp": now.isoformat(),
            "importance": 3,
            "chunk_index": idx,
        }
        documents.append(Document(content=chunk, metadata=metadata))
    return documents


__all__ = ["parse_file", "FileParseError"]
