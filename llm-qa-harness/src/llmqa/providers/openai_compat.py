"""OpenAI-compatible chat completions provider.

Deliberately generic: the /v1/chat/completions shape is spoken by
OpenAI, Ollama, vLLM, llama.cpp, LM Studio, Together, Groq and most
self-hosted inference servers. One provider therefore covers both
"call a frontier model" and "call a 7B model on my laptop", which is
what makes this harness usable with no budget.

Spec form:  openai:<model>[@<base_url>]
    openai:gpt-4o-mini
    openai:qwen2.5:7b@http://localhost:11434/v1
"""

from __future__ import annotations

import os
import time

from .base import (
    CompletionRequest,
    CompletionResponse,
    HTTPProvider,
    ProviderError,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatProvider(HTTPProvider):
    """Calls any endpoint implementing /chat/completions."""

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        # Local servers usually ignore auth entirely; hosted ones do not.
        if not self.api_key and self.base_url == DEFAULT_BASE_URL:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Export it, point --provider at a "
                "local endpoint (openai:<model>@http://localhost:11434/v1), "
                "or use --provider mock."
            )

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def describe(self) -> str:
        if self.base_url != DEFAULT_BASE_URL:
            return f"openai:{self.model}@{self.base_url}"
        return f"openai:{self.model}"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body, error, attempts = self._post_json(self.url, payload, headers)
        latency = time.perf_counter() - started

        if error is not None or body is None:
            return CompletionResponse(
                text="", latency_s=latency, error=error, attempts=attempts
            )

        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            return CompletionResponse(
                text="",
                latency_s=latency,
                error=f"unexpected response shape: {exc}",
                attempts=attempts,
            )

        return CompletionResponse(text=text, latency_s=latency, attempts=attempts)
