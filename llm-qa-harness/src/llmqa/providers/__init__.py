"""Provider registry and spec parsing.

One string on the command line picks the backend:

    mock
    mock:anything
    anthropic:claude-sonnet-4-5
    openai:gpt-4o-mini
    openai:qwen2.5:7b@http://localhost:11434/v1

Parsing lives here rather than in the CLI so that tests can exercise
it without going through argparse.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import (
    CompletionRequest,
    CompletionResponse,
    HTTPProvider,
    Provider,
    ProviderError,
)
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider

__all__ = [
    "AnthropicProvider",
    "CompletionRequest",
    "CompletionResponse",
    "HTTPProvider",
    "MockProvider",
    "OpenAICompatProvider",
    "Provider",
    "ProviderError",
    "build_provider",
    "parse_spec",
]


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """Split a provider spec into (kind, model, base_url).

    The model part may itself contain colons (ollama tags look like
    'qwen2.5:7b'), so only the first colon separates kind from model,
    and '@' separates an optional base URL.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ProviderError("empty provider spec")

    kind, _, remainder = spec.partition(":")
    kind = kind.lower()

    base_url: str | None = None
    model = remainder

    # Split on the last '@' so URLs containing '@' still work.
    if "@" in remainder:
        model, _, base_url = remainder.rpartition("@")
        base_url = base_url or None

    return kind, model, base_url


def build_provider(spec: str, **kwargs: object) -> Provider:
    """Construct a provider from its spec string."""
    kind, model, base_url = parse_spec(spec)

    if kind == "mock":
        return MockProvider(model=model or "fixture")

    if kind == "anthropic":
        if not model:
            raise ProviderError("anthropic provider needs a model: anthropic:<model>")
        return AnthropicProvider(model=model, **kwargs)

    if kind == "openai":
        if not model:
            raise ProviderError(
                "openai provider needs a model: openai:<model>[@<base_url>]"
            )
        return OpenAICompatProvider(model=model, base_url=base_url, **kwargs)

    raise ProviderError(
        f"unknown provider kind {kind!r}; known: mock, anthropic, openai"
    )
