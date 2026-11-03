"""The provider seam.

Everything above this layer asks a Provider for a completion and gets
back text, a latency and maybe an error. Nothing above it knows whether
that came from a hosted API, a model running on localhost, or a fixture
in a YAML file.

That seam is what makes the harness testable: the full pipeline runs in
CI against the mock provider with no API key, no cost and no network,
and the same code paths run against a real model when you point it at
one.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionRequest:
    """What the runner asks a provider for."""

    prompt: str
    system: str | None = None
    case_id: str = ""
    # Carried so the mock provider can serve the fixture declared in the
    # suite. Real providers ignore it.
    mock_response: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass(frozen=True)
class CompletionResponse:
    """What comes back. An error is data, not an exception.

    A provider that raised on every failure would abort a run on the
    first flaky request. A run that reports 3 errored cases out of 60
    is far more useful than a traceback.
    """

    text: str
    latency_s: float
    error: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None


class ProviderError(RuntimeError):
    """Configuration problem - bad spec, missing key. Fail fast."""


class Provider(ABC):
    """A thing that turns a prompt into text."""

    name: str = "provider"
    model: str = ""

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return a completion. Must not raise for transport failures."""

    def describe(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name


# --------------------------------------------------------------------
# Shared HTTP plumbing
# --------------------------------------------------------------------

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class HTTPProvider(Provider):
    """Base for providers that POST JSON to an HTTP endpoint.

    Retries on 429 and 5xx with exponential backoff and jitter. Jitter
    matters when the runner is parallel: without it, N workers that hit
    a rate limit together retry together and hit it again.
    """

    def __init__(
        self,
        *,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
        timeout_s: float = 120.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.timeout_s = timeout_s

    def _backoff_for(self, attempt: int, retry_after: str | None = None) -> float:
        """Seconds to wait before attempt N+1."""
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_cap_s)
            except ValueError:
                pass
        raw = self.backoff_base_s * (2 ** (attempt - 1))
        return min(raw, self.backoff_cap_s) * (0.5 + random.random() * 0.5)

    def _post_json(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str],
    ) -> tuple[dict | None, str | None, int]:
        """POST and return (parsed_body, error, attempts_used)."""
        body = json.dumps(payload).encode("utf-8")
        last_error = "no attempt made"

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                return json.loads(raw), None, attempt

            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:300]
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
                last_error = f"HTTP {exc.code}: {detail or exc.reason}"
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    return None, last_error, attempt
                time.sleep(self._backoff_for(attempt, exc.headers.get("Retry-After")))

            except urllib.error.URLError as exc:
                last_error = f"network error: {exc.reason}"
                if attempt == self.max_retries:
                    return None, last_error, attempt
                time.sleep(self._backoff_for(attempt))

            except json.JSONDecodeError as exc:
                return None, f"response was not JSON: {exc}", attempt

            except TimeoutError:
                last_error = f"timed out after {self.timeout_s}s"
                if attempt == self.max_retries:
                    return None, last_error, attempt
                time.sleep(self._backoff_for(attempt))

        return None, last_error, self.max_retries
