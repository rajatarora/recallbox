from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Tuple

import httpx
import numpy as np
from pydantic import BaseModel
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    pass


class EmbeddingError(OpenRouterError):
    pass


class ChatError(OpenRouterError):
    pass


class MemoryEvaluationError(OpenRouterError):
    pass


class _MemoryEvaluationResponse(BaseModel):
    ok: bool
    explanation: str


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        chat_model: str,
        memory_prompt_path: str,
        base_url: str | None = None,
        config: object | None = None,
        max_retries: int = 3,
        base_retry_wait: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        # Prefer explicitly provided base_url; otherwise use provided config or
        # fall back to reading the global project config. This allows callers
        # (and tests) to provide a Config object to avoid importing the global
        # config singleton and for easier dependency injection.
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif config is not None:
            # Accept any object with an `openrouter_base_url` attribute
            try:
                self.base_url = config.openrouter_base_url.rstrip("/")
            except Exception:
                self.base_url = str(config).rstrip("/")
        else:
            try:
                from recallbox.config import get_config

                cfg = get_config()
                self.base_url = cfg.openrouter_base_url.rstrip("/")
            except Exception:
                # Fallback to sensible default if config is unavailable
                self.base_url = "https://openrouter.ai/api/v1"
        self.max_retries = max_retries
        self.base_retry_wait = base_retry_wait
        # Path to a prompt template file used by evaluate_memory. The file
        # must contain placeholders {user} and {assistant} which will be
        # replaced using str.format(). This is required (no fallback).
        self.memory_prompt_path = memory_prompt_path

        # httpx client will be created per request to keep things simple in tests

    async def _request_with_retry(self, method: str, path: str, json_data: dict) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                headers = {"Authorization": f"Bearer {self._api_key}"}
                async with httpx.AsyncClient(http2=False, timeout=10.0) as client:
                    resp = await client.request(method, url, json=json_data, headers=headers)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            wait = int(retry_after)
                        except ValueError:
                            wait = float(retry_after)
                        logger.warning("Rate limited, sleeping %s seconds", wait, extra={"component": "openrouter"})
                        await asyncio.sleep(wait)
                        continue
                    # no Retry-After, fall through to raise

                if 500 <= resp.status_code < 600:
                    # server error -> retry
                    if attempt < self.max_retries:
                        wait = self.base_retry_wait * (2**attempt)
                        logger.warning(
                            "Server error %s, retrying in %s seconds (attempt %s)",
                            resp.status_code,
                            wait,
                            attempt + 1,
                            extra={"component": "openrouter"},
                        )
                        await asyncio.sleep(wait)
                        continue
                    else:
                        raise OpenRouterError(f"Server error: {resp.status_code}")

                if resp.status_code >= 400:
                    raise OpenRouterError(f"HTTP error: {resp.status_code} {resp.text}")

                return resp.json()
            except httpx.HTTPError as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = self.base_retry_wait * (2**attempt)
                    logger.warning(
                        "HTTP error %s, retrying in %s seconds (attempt %s)",
                        e,
                        wait,
                        attempt + 1,
                        extra={"component": "openrouter"},
                    )
                    await asyncio.sleep(wait)
                    continue
                raise OpenRouterError("HTTP request failed") from e

        if last_exc:
            raise OpenRouterError("Request failed") from last_exc
        raise OpenRouterError("Unknown error")

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        if len(texts) == 0:
            return []
        if len(texts) > 64:
            raise EmbeddingError("Batch size must be <= 64")

        body = {"model": self.embedding_model, "input": texts}
        try:
            resp = await self._request_with_retry("POST", "/embeddings", body)
        except OpenRouterError as e:
            raise EmbeddingError("Failed to get embeddings") from e

        # Expecting {'data': [{'embedding': [...]}, ...]}
        data = resp.get("data")
        if not isinstance(data, list):
            raise EmbeddingError("Unexpected response format for embeddings")

        vectors: List[np.ndarray] = []
        for item in data:
            emb = item.get("embedding")
            if emb is None:
                raise EmbeddingError("Missing embedding vector in response")
            arr = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(arr) + 1e-10
            arr = arr / norm
            vectors.append(arr)

        return vectors

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        body = {"model": self.chat_model, "messages": messages}
        try:
            resp = await self._request_with_retry("POST", "/chat/completions", body)
        except OpenRouterError as e:
            raise ChatError("Chat request failed") from e

        # Expecting {'choices': [{'message': {'role': 'assistant', 'content': '...'}}]}
        choices = resp.get("choices")
        if not choices or not isinstance(choices, list):
            raise ChatError("Unexpected chat response")

        first = choices[0]
        message = first.get("message") or first.get("text")
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = message

        if content is None:
            raise ChatError("No assistant content in response")

        return content

    async def evaluate_memory(self, user: str, assistant: str) -> Tuple[bool, str]:
        # Always load the prompt template from the configured file. No
        # fallback is provided; failure to read or format the template will
        # raise MemoryEvaluationError.
        try:
            tmpl = Path(self.memory_prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            raise MemoryEvaluationError("Failed to read memory prompt file") from e
        try:
            # Use simple placeholder replacement for {user} and {assistant}
            # so templates can include other braces (for JSON) without
            # needing to escape them for str.format().
            prompt = tmpl.replace("{user}", user).replace("{assistant}", assistant)
        except Exception as e:
            raise MemoryEvaluationError("Failed to format memory prompt template") from e
        messages = [{"role": "user", "content": prompt}]
        resp_text = await self.chat(messages)

        try:
            parsed = json.loads(resp_text)
            model = _MemoryEvaluationResponse.model_validate(parsed)
            return model.ok, model.explanation
        except Exception as e:
            raise MemoryEvaluationError("Failed to parse memory evaluation response") from e
