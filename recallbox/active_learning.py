"""Active learning helpers for evaluating and storing memories.

Provides an async helper that evaluates a user-assistant exchange using the
configured OpenRouter client and stores a concise summary in the long-term
memory store when appropriate.

The public API in this module is intentionally small:
- `evaluate_and_store(user, assistant, store, llm)` — fire-and-forget style
  async helper that will not raise to callers (all errors are caught and
  logged).
"""

from __future__ import annotations

import asyncio
import difflib
from datetime import datetime, timezone
from typing import Optional

import structlog

from recallbox.llm.client import MemoryEvaluationError, OpenRouterClient
from recallbox.store.chromadb import Document, MemoryStore

__all__ = ["ActiveLearningError", "evaluate_and_store"]

logger = structlog.get_logger()
log = logger.bind(component="active_learning")


class ActiveLearningError(RuntimeError):
    """Raised for unrecoverable active-learning internal failures.

    NOTE: evaluate_and_store itself will not raise; this exception is defined
    for callers who may reuse lower-level utilities.
    """


def _heuristic_extract(user: str, assistant: str) -> Optional[str]:
    """Fallback heuristic to extract a memory summary.

    Criteria:
    - Similarity ratio (difflib.SequenceMatcher) > 0.8
    - Assistant contains a sentence that starts with "I remember" (case
      insensitive). The matched sentence (stripped) is returned.

    Returns the extracted sentence or None.
    """
    # First find candidate sentences that start with "I remember"
    normalized = assistant.replace("!", ".").replace("?", ".")
    candidates = [p.strip() for p in normalized.split(".") if p.strip()]

    for cand in candidates:
        if cand.lower().startswith("i remember"):
            # Compute similarity between the user text and the candidate sentence
            ratio = difflib.SequenceMatcher(None, user, cand).ratio()
            if ratio > 0.8:
                return cand

    # No candidate matched
    return None


async def evaluate_and_store(user: str, assistant: str, store: MemoryStore, llm: OpenRouterClient) -> None:
    """Evaluate a conversation and persist a concise memory if warranted.

    The function captures all exceptions and logs them; it never raises to the
    caller to avoid impacting the REPL. The LLM evaluation is performed using
    `llm.evaluate_memory(user, assistant)` and on JSON/parse failures a simple
    heuristic is attempted in a thread via `asyncio.to_thread`.
    """
    try:
        should_store = False
        summary: Optional[str] = None

        try:
            # Exact LLM call required by spec
            should_store, explanation = await llm.evaluate_memory(user, assistant)
            if should_store:
                summary = explanation
        except MemoryEvaluationError as e:
            # JSON parsing (or prompt template) failed inside the LLM client
            log.warning("LLM memory evaluation failed, falling back to heuristic", error=str(e))
            summary = await asyncio.to_thread(_heuristic_extract, user, assistant)
            if summary:
                should_store = True
                log.info("Heuristic recovered memory", summary=summary)
            else:
                should_store = False
                log.debug("Heuristic did not find a memory")
        except Exception as e:
            # Any unexpected errors from LLM should also fall back to heuristic
            log.warning("Unexpected LLM error, attempting heuristic", error=str(e))
            summary = await asyncio.to_thread(_heuristic_extract, user, assistant)
            if summary:
                should_store = True
                log.info("Heuristic recovered memory after unexpected error", summary=summary)
            else:
                should_store = False
                log.debug("Heuristic did not find a memory after unexpected error")

        if should_store and summary:
            utc_iso = datetime.now(timezone.utc).isoformat()
            doc = Document(
                content=summary,
                metadata={"source": "ai_extracted", "timestamp": utc_iso, "importance": 4},
            )

            try:
                await store.add_documents([doc])
                log.info("Memory saved", event="memory_saved", summary=summary)
            except Exception as e:
                log.exception("Failed to save memory to store", error=str(e))
        else:
            log.debug("No memory to store from evaluation")

    except Exception as e:
        # Catch-all: never bubble errors to the REPL
        log.exception("Unhandled error in evaluate_and_store", error=str(e))
