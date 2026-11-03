"""A deterministic provider for CI and offline development.

This is not a pretend model. It is a fixture: each case declares the
response it should be scored against, and the mock hands that back.

That distinction is the point. These runs do not tell you anything
about a model - they tell you whether the harness itself still works:
that suites parse, scorers fire, severities propagate, the gate exits
with the right code. Those are the regressions that silently break an
evaluation pipeline, and they are exactly the ones a model-dependent
test cannot catch.
"""

from __future__ import annotations

import hashlib
import time

from .base import CompletionRequest, CompletionResponse, Provider


class MockProvider(Provider):
    """Returns each case's declared mock_response.

    Cases with no mock_response get a stable stub derived from the
    prompt hash, so a suite without fixtures still runs end to end
    (and will mostly fail its checks, which is the honest outcome).
    """

    name = "mock"

    def __init__(self, model: str = "fixture", latency_s: float = 0.0) -> None:
        self.model = model
        self.latency_s = latency_s

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)

        if request.mock_response is not None:
            text = request.mock_response
        else:
            digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:12]
            text = (
                f"[mock] no fixture declared for case {request.case_id or 'unknown'} "
                f"(prompt digest {digest})"
            )

        return CompletionResponse(
            text=text,
            latency_s=time.perf_counter() - started,
        )
