"""Anthropic Messages API provider.

Stdlib HTTP only - no SDK. That keeps the dependency surface at one
package (PyYAML) and makes the request shape explicit rather than
hidden behind a client library, which is useful when the thing you are
demonstrating is the harness rather than the API.
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

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(HTTPProvider):
    """Calls the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Export it, or use "
                "--provider mock to run offline."
            )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()

        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            payload["system"] = request.system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        body, error, attempts = self._post_json(API_URL, payload, headers)
        latency = time.perf_counter() - started

        if error is not None or body is None:
            return CompletionResponse(
                text="", latency_s=latency, error=error, attempts=attempts
            )

        try:
            blocks = body.get("content") or []
            text = "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )
        except (AttributeError, TypeError) as exc:
            return CompletionResponse(
                text="",
                latency_s=latency,
                error=f"unexpected response shape: {exc}",
                attempts=attempts,
            )

        return CompletionResponse(text=text, latency_s=latency, attempts=attempts)
