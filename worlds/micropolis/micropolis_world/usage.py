"""Per-call token and cost accounting for the model calls this world makes.

prompt_model() is the one place micropolis talks to an LLM, so it is the one
place tokens and dollars can be read off a reply; this module holds the record
it produces and the LiteLLM-shaped reading of it.

Kept litellm-free at import time — the one function that needs litellm imports
it lazily — so the scoring and analysis scripts can read recorded usage without
pulling in litellm.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CallUsage:
    """Tokens and dollars for one completed LLM API call.

    cost_usd is None, never 0.0, when litellm has no price-map entry for the
    model: an unknown price must never be silently reported as free. Anything
    summing these has to decide what to say about the unpriced calls, and a
    None forces that decision instead of hiding it.

    Attributes:
        model_id: The id we asked for, e.g. "google/gemini-2.5-pro".
        response_model: What the provider echoed back, which is often the bare
            name without the provider prefix, and sometimes a dated variant of
            the alias that was requested.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed, including reasoning tokens.
        total_tokens: Input plus output.
        reasoning_tokens: Of the output tokens, those spent thinking rather
            than answering. None when the provider doesn't report it.
        cached_input_tokens: Of the input tokens, those served from a provider
            prompt cache at a discount. None when not reported.
        cache_write_tokens: Input tokens billed at a premium to populate a
            provider prompt cache. None when not reported.
        cost_usd: What the call cost, or None if the model has no price entry.
        latency_ms: Wall-clock time spent in the API call.
    """

    model_id: str
    response_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """The usage as plain JSON-serializable data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CallUsage":
        """Read back what to_dict() wrote, ignoring fields we don't know.

        Unknown keys are dropped rather than raising so that a record written
        by a later version of this dataclass still loads here.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        """One-line human summary, for a progress line next to a call."""
        cost = "cost unknown" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        parts = [cost, f"{self.input_tokens} in", f"{self.output_tokens} out"]
        if self.reasoning_tokens:
            parts.append(f"{self.reasoning_tokens} reasoning")
        if self.cached_input_tokens:
            parts.append(f"{self.cached_input_tokens} cached")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One LLM reply: its text, why it stopped, and what it cost.

    text is None when the model produced no content — a reasoning model can
    spend its whole budget thinking and stop at finish_reason "length" with
    nothing written. That call is still billed, so usage is populated either
    way and a None text must not be read as a free call.

    Attributes:
        text: The reply, or None if the model wrote nothing.
        finish_reason: Why generation stopped ("stop", "length", ...).
        usage: Tokens and cost for the call.
    """

    text: str | None
    finish_reason: str | None
    usage: CallUsage


def _detail(obj: Any, *names: str) -> int | None:
    """First present attribute among `names`, or None if obj has none of them.

    litellm's usage-detail wrappers delete their unset optional fields in
    __init__, so plain attribute access raises AttributeError rather than
    returning None, and the whole wrapper is None when nothing was reported.
    Field names also drift between litellm versions — cache writes were
    cache_creation_tokens before 1.96 and cache_write_tokens after — so more
    than one name is tried and a rename degrades a field to None rather than
    raising.
    """
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def cost_from_response(response: Any, model_id: str | None = None) -> float | None:
    """USD cost of a litellm response, or None if the model has no price entry.

    Never raises: litellm.completion_cost() throws for any model missing from
    its price map, and pricing a call must not be able to fail the call that
    was already paid for. An unpriced call reports None rather than 0.0.

    Args:
        response: The object litellm's completion() returned.
        model_id: The id we asked for. Passed to litellm as an extra candidate
            name, which matters when the provider echoes back a bare
            "gemini-2.5-pro" while the price map is keyed "gemini/gemini-2.5-pro".

    Returns:
        The cost in USD, or None if litellm can't price the model.
    """
    # Only ever populated behind a LiteLLM proxy or on the batch path, not by a
    # plain completion() call — but it's free to check and it's the only correct
    # value when it is there.
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("response_cost") is not None:
        try:
            return float(hidden["response_cost"])
        except (TypeError, ValueError):
            pass

    # Imported here so the scoring-only scripts don't pull in litellm.
    import litellm

    try:
        cost = litellm.completion_cost(completion_response=response, model=model_id)
    except Exception:  # noqa: BLE001 - litellm raises bare Exception for unmapped models
        return None
    return None if cost is None else float(cost)


def usage_from_response(
    response: Any,
    model_id: str,
    latency_ms: float | None = None,
) -> CallUsage:
    """Tokens and cost for a litellm response.

    Args:
        response: The object litellm's completion() returned.
        model_id: The id we asked for, recorded as CallUsage.model_id.
        latency_ms: Wall-clock time the call took, if measured.

    Returns:
        The call's usage. Fields the provider didn't report are None.
    """
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    # litellm leaves total_tokens at 0 unless the provider sent it, so fall back
    # to the sum rather than reporting a total that contradicts its own parts.
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    return CallUsage(
        model_id=model_id,
        response_model=getattr(response, "model", None),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or (input_tokens + output_tokens),
        reasoning_tokens=_detail(completion_details, "reasoning_tokens"),
        cached_input_tokens=_detail(prompt_details, "cached_tokens"),
        cache_write_tokens=_detail(
            prompt_details, "cache_write_tokens", "cache_creation_tokens"
        ),
        cost_usd=cost_from_response(response, model_id),
        latency_ms=latency_ms,
    )
