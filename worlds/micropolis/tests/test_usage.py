"""Tests for reading tokens and cost off a model reply.

Fixtures are built from LiteLLM's own response types rather than hand-rolled
stubs, so a field that LiteLLM renames or stops populating fails a test here
instead of silently reporting None in a run. LiteLLM is stubbed by replacing
the name prompt_model binds, so nothing here calls a model or needs a key.
"""

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from fbsim_core.evaluation.models import LiteLLMModel

import micropolis_world.module_globals as g
from micropolis_world.usage import CallUsage, cost_from_response, usage_from_response


def make_response(
    model: str = "gpt-4o",
    content: str | None = "hello",
    finish_reason: str = "stop",
    usage: Usage | None = None,
) -> ModelResponse:
    """A LiteLLM response of the shape completion() returns."""
    return ModelResponse(
        id="chatcmpl-test",
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ],
        created=1700000000,
        model=model,
        object="chat.completion",
        usage=usage or Usage(prompt_tokens=100, completion_tokens=50),
    )


@pytest.fixture
def calls(monkeypatch):
    """Stub out litellm.completion, recording the kwargs prompt_model sends.

    Patching the attribute on litellm works only because prompt_model does its
    `from litellm import completion` inside the function body, so the name is
    resolved per call. Hoisting that import to module level would rebind it once
    at import time and these tests would start making real API calls — which is
    exactly what happens to fbsim-core's models.py, and why it isn't stubbed here.
    """
    recorded = []

    def fake_completion(**kwargs):
        recorded.append(kwargs)
        return make_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    return recorded


# --- reading a response -----------------------------------------------------


def test_reads_every_reported_field():
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        completion_tokens_details={"reasoning_tokens": 40},
        prompt_tokens_details={"cached_tokens": 20},
    )
    got = usage_from_response(make_response(usage=usage), "openai/gpt-4o", 123.5)

    assert got.model_id == "openai/gpt-4o"
    assert got.response_model == "gpt-4o"
    assert got.input_tokens == 100
    assert got.output_tokens == 50
    assert got.reasoning_tokens == 40
    assert got.cached_input_tokens == 20
    assert got.latency_ms == 123.5


def test_missing_details_are_none_not_an_error():
    """The regression test for LiteLLM's deleted-when-unset detail fields.

    Usage(prompt_tokens=..., completion_tokens=...) leaves both detail wrappers
    None, and the wrappers themselves `del` their unset optional attributes, so
    plain attribute access raises AttributeError. This fails loudly if the
    getattr-based reads are ever replaced with direct access.
    """
    got = usage_from_response(make_response(), "openai/gpt-4o")

    assert got.reasoning_tokens is None
    assert got.cached_input_tokens is None
    assert got.cache_write_tokens is None
    assert got.latency_ms is None


def test_total_falls_back_to_the_sum():
    """LiteLLM leaves total_tokens at 0 unless the provider sends it."""
    assert usage_from_response(make_response(), "openai/gpt-4o").total_tokens == 150


def test_total_prefers_what_the_provider_reported():
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=175)
    got = usage_from_response(make_response(usage=usage), "openai/gpt-4o")
    assert got.total_tokens == 175


# --- pricing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id, response_model",
    [
        ("openai/gpt-4o", "gpt-4o"),
        ("anthropic/claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        # The prefix LiteLLM prices this under ("gemini/") is not the one we ask
        # for ("google/"), which is why model_id is passed as an extra candidate.
        ("google/gemini-2.5-pro", "gemini-2.5-pro"),
    ],
)
def test_priced_models_cost_something(model_id, response_model):
    cost = cost_from_response(make_response(model=response_model), model_id)
    # A range, not an exact figure: LiteLLM ships its price map in-tree and
    # updates it constantly, so pinning the number schedules a failure.
    assert cost is not None
    assert 0 < cost < 1


def test_unpriced_model_is_none_and_never_raises():
    """LiteLLM raises for a model absent from its price map; we report None.

    None rather than 0.0 so an unknown price is never mistaken for a free call.
    """
    response = make_response(model="definitely-not-a-real-model")
    assert cost_from_response(response, "openai/definitely-not-a-real-model") is None


def test_unpriced_model_still_reports_its_tokens():
    response = make_response(model="definitely-not-a-real-model")
    got = usage_from_response(response, "openai/definitely-not-a-real-model")
    assert got.cost_usd is None
    assert got.input_tokens == 100
    assert got.output_tokens == 50


def test_hidden_params_cost_wins_when_present():
    """The LiteLLM-proxy path, where the cost is precomputed on the response."""
    response = make_response()
    response._hidden_params = {"response_cost": 0.0123}
    assert cost_from_response(response, "openai/gpt-4o") == 0.0123


# --- CallUsage as a record --------------------------------------------------


def test_roundtrips_through_json_shaped_data():
    usage = CallUsage(model_id="openai/gpt-4o", input_tokens=1, cost_usd=0.5)
    assert CallUsage.from_dict(usage.to_dict()) == usage


def test_from_dict_ignores_fields_it_does_not_know():
    """A record written by a later version of CallUsage still loads."""
    got = CallUsage.from_dict({"model_id": "openai/gpt-4o", "some_future_field": 7})
    assert got.model_id == "openai/gpt-4o"


def test_describe_says_unknown_rather_than_zero():
    assert "cost unknown" in CallUsage(model_id="x").describe()
    assert "$0.0500" in CallUsage(model_id="x", cost_usd=0.05).describe()


# --- prompt_model ------------------------------------------------------------


def test_prompt_model_returns_text_finish_reason_and_cost(calls):
    got = g.prompt_model(LiteLLMModel("openai/gpt-4o"), "hi", max_tokens=10)

    assert got.text == "hello"
    assert got.finish_reason == "stop"
    assert got.usage.model_id == "openai/gpt-4o"
    assert got.usage.input_tokens == 100
    assert got.usage.output_tokens == 50
    assert got.usage.cost_usd is not None
    assert got.usage.latency_ms is not None


def test_prompt_model_sends_the_normalized_model_id(calls):
    """google/ is asked for; gemini/ is what LiteLLM's API expects."""
    g.prompt_model(LiteLLMModel("google/gemini-2.5-pro"), "hi", max_tokens=10)
    assert calls[0]["model"] == "gemini/gemini-2.5-pro"


def test_prompt_model_omits_temperature_only_for_models_that_reject_it(calls):
    g.prompt_model(LiteLLMModel("openai/gpt-4o"), "hi", max_tokens=10)
    g.prompt_model(LiteLLMModel("openai/gpt-5"), "hi", max_tokens=10)

    supported, unsupported = calls
    assert supported["temperature"] == 0.0
    assert "temperature" not in unsupported
    # The cap is always sent, unlike fbsim-core's get_response.
    assert supported["max_tokens"] == unsupported["max_tokens"] == 10


def test_prompt_model_reports_usage_for_an_empty_reply(monkeypatch):
    """A reasoning model that thinks past its budget writes nothing but bills."""

    def fake_completion(**kwargs):
        return make_response(
            content=None,
            finish_reason="length",
            usage=Usage(prompt_tokens=100, completion_tokens=4000),
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    got = g.prompt_model(LiteLLMModel("openai/gpt-4o"), "hi", max_tokens=4000)

    assert got.text is None
    assert got.finish_reason == "length"
    assert got.usage.output_tokens == 4000
    assert got.usage.cost_usd is not None
    assert got.usage.cost_usd > 0
