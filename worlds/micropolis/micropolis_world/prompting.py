"""Concurrent LLM prompting: per-provider rate limiting and async fan-out.

The scripts that prompt many models build a list of PromptJobs and consume
run_prompts() with `async for`, so the network calls overlap while every disk
write and print happens serially in the consumer — no locks, no interleaved
output. The per-provider semaphore pattern is adapted from
worlds/freeciv/freeciv_world/evaluation/rate_limiter.py (copied, not imported:
freeciv is a sibling world, not a dependency).

Kept litellm-free at import time, like usage.py: the one function that needs
litellm imports it lazily, so scoring-only code can import this module's
dataclasses without pulling in litellm.

prompt_model_async takes a full message list rather than a prompt string, so a
future multi-turn caller just appends assistant/user turns and calls it again;
run_prompts is only the single-shot fan-out convenience built on top of it and
ProviderRateLimiter, both of which are public for exactly that reason.
"""

import asyncio
import random
import sys
import time
from collections.abc import AsyncIterator, Hashable
from dataclasses import dataclass
from typing import Any

from .usage import LLMResponse, usage_from_response

# Max concurrent in-flight calls per provider. A little above freeciv's
# deliberately conservative numbers: 429s are retried with backoff (see
# DEFAULT_NUM_RETRIES), so these only need to keep the retry loop from being
# the common case, and a config's "provider_concurrency" map can override any
# of them. Keyed by LiteLLMModel.provider_cls strings.
DEFAULT_PROVIDER_LIMITS: dict[str, int] = {
    "OpenAIProvider": 16,
    "AnthropicProvider": 16,
    "GoogleProvider": 16,
    "TogetherProvider": 4,
    "MistralProvider": 2,
    "XAIProvider": 2,
}
DEFAULT_UNKNOWN_LIMIT = 2

# Retries after the first attempt. prompt_model_async owns the retry loop:
# litellm's own retry paths (its tenacity wrapper and the provider SDK's
# max_retries) are completely silent, so a run deep in 429 territory just
# looked slow. We disable both layers and back off ourselves, printing a
# warning to stderr before each retry.
DEFAULT_NUM_RETRIES = 3
# Backoff schedule: exponential for 429s, a short constant pause for other
# transient errors (connection drops, 5xx, timeouts). Module-level so tests
# can zero them out.
RATE_LIMIT_BACKOFF_BASE_S = 1.0
RATE_LIMIT_BACKOFF_MAX_S = 30.0
TRANSIENT_BACKOFF_S = 1.0
# Per-attempt cap, forwarded to the provider client. Generous: a 60k-token
# reasoning reply can legitimately run many minutes.
DEFAULT_TIMEOUT_S = 900.0


def format_eta(start: float, done: int, total: int) -> str:
    """Crude time-left note: mean seconds per completion times what is left.

    `start` is the time.perf_counter() reading from just before the fan-out.
    Deliberately simple — it knows nothing about which calls are still in
    flight or how providers differ in speed, just enough to answer "wait or
    walk away?". Returns "" once everything is done, and carries its own
    leading separator, so callers can append it to a line unconditionally.
    """
    if done >= total:
        return ""
    seconds = (time.perf_counter() - start) / done * (total - done)
    if seconds >= 60:
        return f"  ~{seconds / 60:.0f}m left"
    return f"  ~{seconds:.0f}s left"


def format_latency(latency_ms: float | None, retries: int = 0) -> str:
    """How long one API call took, as a phrase for a progress line.

    Reads CallUsage.latency_ms, which prompt_model_async measures around the
    acompletion await — the time from the request going out to the reply
    landing, for the attempt that succeeded, not the retries before it. Those
    are reported separately, as "2 RETRIES" after the time, because they are
    what explains a call that took far longer in wall clock than the time
    shown; a first-try success says nothing at all.

    Returns "" when the latency was never recorded, and carries its own
    leading separator, so callers can append it unconditionally.
    """
    if latency_ms is None:
        return ""
    took = f", {latency_ms / 1000:.1f}s"
    if retries:
        took += f" {retries} RETR{'Y' if retries == 1 else 'IES'}"
    return took


class ProviderRateLimiter:
    """Per-provider concurrency caps as lazily created asyncio.Semaphores.

    Keyed on LiteLLMModel.provider_cls, so every model of one provider shares
    one cap — five OpenAI models with a limit of 4 still make at most 4
    concurrent OpenAI calls. Semaphores are created on first use because they
    must be instantiated inside the running event loop.
    """

    def __init__(self, limits: dict[str, int] | None = None):
        self._limits = {**DEFAULT_PROVIDER_LIMITS, **(limits or {})}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def limit(self, provider_cls: str) -> int:
        """The concurrency cap for a provider class name."""
        return self._limits.get(provider_cls, DEFAULT_UNKNOWN_LIMIT)

    def semaphore(self, provider_cls: str) -> asyncio.Semaphore:
        """The (lazily created) semaphore for a provider class name."""
        if provider_cls not in self._semaphores:
            self._semaphores[provider_cls] = asyncio.Semaphore(self.limit(provider_cls))
        return self._semaphores[provider_cls]

    def __repr__(self) -> str:
        return f"ProviderRateLimiter(limits={self._limits})"


async def prompt_model_async(
    model: Any,
    messages: list[dict],
    max_tokens: int,
    *,
    num_retries: int = DEFAULT_NUM_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> LLMResponse:
    """Send `messages` to `model`, returning its text, finish reason and cost.

    The async twin of module_globals.prompt_model, and kept in step with it:
    same normalized model id, same temperature gate, same LLMResponse. The
    kwargs logic is knowingly duplicated rather than shared — factoring it out
    would couple the two lazy-import boundaries the tests monkeypatch through.
    The differences: a full message list instead of a single user prompt (so a
    multi-turn caller can pass a running conversation), and retry/timeout
    settings, which matter once calls run concurrently and a provider starts
    returning 429s.

    Retries happen here, not in litellm: num_retries=0 and max_retries=0
    disable litellm's wrapper retries and the provider SDK's, both of which
    are silent. Rate limits back off exponentially, other transient errors
    (connection drops, 5xx, timeouts) pause briefly, and each retry prints a
    one-line warning to stderr so a throttled run is visible. Anything else
    (auth failures, bad requests) raises immediately.

    `model` is a LiteLLMModel, duck-typed (.id, ._litellm_model_id,
    .supports_temperature) so importing this module never imports fbsim-core,
    which pulls in litellm at module level.
    """
    # Imported here so tests can monkeypatch litellm.acompletion; the name is
    # resolved per call. See test_usage.py's `calls` fixture for why hoisting
    # this to module level would break the stub.
    from litellm import (
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
        acompletion,
        supports_reasoning,
    )

    kwargs = {
        "model": model._litellm_model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "num_retries": 0,
        "max_retries": 0,
        "timeout": timeout,
    }
    if model.supports_temperature and not supports_reasoning(model._litellm_model_id):
        kwargs["temperature"] = 0.0

    for attempt in range(num_retries + 1):
        start = time.perf_counter()
        try:
            response = await acompletion(**kwargs)
        except (
            RateLimitError,
            APIConnectionError,
            InternalServerError,
            ServiceUnavailableError,
            Timeout,
        ) as e:
            if attempt == num_retries:
                raise
            if isinstance(e, RateLimitError):
                delay = min(
                    RATE_LIMIT_BACKOFF_BASE_S * 2**attempt, RATE_LIMIT_BACKOFF_MAX_S
                )
            else:
                delay = TRANSIENT_BACKOFF_S
            print(
                f"[retry] {model.id}: {type(e).__name__} on attempt"
                f" {attempt + 1}/{num_retries + 1}, retrying in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)
        else:
            break
    latency_ms = (time.perf_counter() - start) * 1000

    choice = response.choices[0]  # type: ignore
    return LLMResponse(
        text=choice.message.content,
        finish_reason=choice.finish_reason,
        usage=usage_from_response(response, model.id, latency_ms),
        # `attempt` is the index of the try that succeeded, so it counts the
        # failed ones before it: 0 on a first-try success.
        retries=attempt,
    )


@dataclass(frozen=True)
class PromptJob:
    """One model call to make: who to ask, what to send, how to name it back.

    key is an opaque caller identity — e.g. (batch_id, model_name) — echoed
    back on the PromptResult so the consumer can route the reply without
    keeping its own bookkeeping. model_name is the id the caller asked for
    (the config spelling), carried separately because model is duck-typed.
    """

    key: Hashable
    model: Any
    model_name: str
    messages: list[dict]
    max_tokens: int


@dataclass(frozen=True)
class PromptResult:
    """One finished PromptJob: a response, or the exception that ended it.

    Exactly one of response/error is set. Errors are carried, not raised, so
    one failed call never tears down the rest of a fan-out; the consumer
    decides what a failure costs.
    """

    job: PromptJob
    response: LLMResponse | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_prompts(
    jobs: list[PromptJob],
    *,
    limits: dict[str, int] | None = None,
    num_retries: int = DEFAULT_NUM_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> AsyncIterator[PromptResult]:
    """Run every job concurrently, yielding results in completion order.

    Calls to the same provider are capped by a shared ProviderRateLimiter
    (`limits` overrides individual defaults). The semaphore is held across
    prompt_model_async's retries, so a provider in 429 territory stays
    throttled while it backs off rather than being hit by the next waiting
    call.

    Yields one PromptResult per job as each finishes. Only Exception is
    captured into results — CancelledError and KeyboardInterrupt propagate, so
    Ctrl-C still stops the run; the finally block then cancels whatever is
    still in flight.
    """
    limiter = ProviderRateLimiter(limits)
    # We shuffle the list of jobs, so we get better time estimates sooner
    random.shuffle(jobs)

    async def worker(job: PromptJob) -> PromptResult:
        async with limiter.semaphore(job.model.provider_cls):
            try:
                response = await prompt_model_async(
                    job.model,
                    job.messages,
                    job.max_tokens,
                    num_retries=num_retries,
                    timeout=timeout,
                )
                return PromptResult(job, response, None)
            except Exception as e:  # noqa: BLE001 - carried to the consumer
                return PromptResult(job, None, e)

    tasks = [asyncio.create_task(worker(job)) for job in jobs]
    try:
        for future in asyncio.as_completed(tasks):
            yield await future
    finally:
        # Early exit (an exception in the consumer, Ctrl-C) must not leave
        # workers running against a closing event loop.
        for task in tasks:
            task.cancel()
