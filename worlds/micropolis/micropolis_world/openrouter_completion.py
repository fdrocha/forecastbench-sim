#!/usr/bin/env -S uv run python3
"""Drop-in `completion` / `acompletion` / `completion_cost` for OpenRouter.

    from openrouter_completion import completion, acompletion, completion_cost

Same call shape as LiteLLM. Differences that matter:

* `model` is a bare OpenRouter slug. If it is in MODELS — loaded at import
  from `model_specs.json5` beside this file — that spec supplies the
  request's provider routing, sampling defaults and reasoning config.
  A slug with no spec is sent as-is.
* Every ModelSpec field is optional. `endpoint` pins a provider (with
  `ignore`/`quantizations` narrowing it); `sampling` sets body defaults;
  `max_tokens` caps the output; `reasoning_effort` or
  `reasoning_budget_tokens` — never both — set `reasoning`. Nothing is
  validated against the OpenRouter API.
* Caller kwargs override the spec, key by key. A caller-supplied `provider`
  or `reasoning` dict wins outright.
* The response mirrors LiteLLM's ModelResponse for attribute access, keeps
  every OpenRouter field in place (`provider`, `usage.cost`, ...), and puts
  provenance under `_hidden_params`.
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import json5

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 600.0


# ---------------------------------------------------------------------------
# Auth — resolved at call time, never at import.
# ---------------------------------------------------------------------------


def _headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# Model registry, from model_specs.json5. Every field is optional; a slug
# with no entry is sent to OpenRouter as-is.
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    endpoint: str | None = None  # FULL provider slug, e.g. "deepinfra/turbo"
    quantizations: list[str] | None = None  # e.g. ["fp8"]
    ignore: list[str] = field(default_factory=list)  # provider variants to exclude
    sampling: dict[str, Any] = field(default_factory=dict)  # temperature, top_p, ...
    max_tokens: int | None = None  # output cap; not reasoning_budget_tokens
    reasoning_effort: str | None = None  # -> reasoning.effort
    reasoning_budget_tokens: int | None = None  # -> reasoning.max_tokens

    def __post_init__(self) -> None:
        if self.reasoning_effort and self.reasoning_budget_tokens:
            raise ValueError(
                "reasoning_effort and reasoning_budget_tokens are mutually "
                "exclusive; set at most one"
            )


MODEL_SPECS_PATH = Path(__file__).with_name("model_specs.json5")


def load_model_specs(path: Path = MODEL_SPECS_PATH) -> dict[str, ModelSpec]:
    """Read the registry; empty (with a warning) if the file is absent."""
    if not path.exists():
        warnings.warn(
            f"{path.name} not found at {path} — MODELS is empty, "
            f"every model will be sent with OpenRouter's defaults",
            stacklevel=2,
        )
        return {}
    raw = json5.loads(path.read_text())
    return {slug: ModelSpec(**spec) for slug, spec in raw.items()}


MODELS: dict[str, ModelSpec] = load_model_specs()


# ---------------------------------------------------------------------------
# Response object: LiteLLM-style attribute access over OpenRouter's JSON.
# ---------------------------------------------------------------------------


def _wrap(v: Any) -> Any:
    if isinstance(v, AttrDict):
        return v
    if isinstance(v, dict):
        return AttrDict(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


class AttrDict(dict):
    """dict with attribute access; nested dicts/lists wrap on access.
    `resp.choices[0].message.content` and `resp["choices"][0]["message"]` both work."""

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self[name])
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def model_dump(self) -> dict:
        return {
            **json.loads(json.dumps(self)),
            "_hidden_params": getattr(self, "_hidden_params", {}),
        }

    def json(self, **kw: Any) -> str:  # LiteLLM parity
        return json.dumps(self.model_dump(), **kw)


def completion_cost(completion_response: Any = None) -> float:
    """LiteLLM-shaped; here it just reads OpenRouter's billed cost off the response.
    Only supports being called with completion_response."""
    if completion_response is None:
        raise ValueError("completion_cost needs a completion_response")
    cost = (completion_response.get("usage") or {}).get("cost")
    if cost is None:
        cost = getattr(completion_response, "_hidden_params", {}).get("response_cost")
    if cost is None:
        raise ValueError("response carries no cost field")
    return float(cost)


# ---------------------------------------------------------------------------
# Request body.
# ---------------------------------------------------------------------------

_CONTROL = {"fetch_generation", "api_key", "timeout", "stream"}


def _build(model: str, messages: list[dict], kwargs: dict) -> dict:
    if kwargs.get("stream"):
        raise NotImplementedError("stream=True is not supported by this module")

    spec = MODELS.get(model)
    body: dict[str, Any] = {
        "model": model,  # bare slug — :nitro/:floor change routing
        "messages": messages,
        "transforms": [],  # disable middle-out compression explicitly
        **(spec.sampling if spec else {}),
    }

    if spec:
        if spec.max_tokens:
            body["max_tokens"] = spec.max_tokens

        provider: dict[str, Any] = {
            "require_parameters": True,  # refuse endpoints that would drop our params
        }
        if spec.endpoint:
            provider["only"] = [spec.endpoint]
            provider["allow_fallbacks"] = False
        if spec.quantizations:
            provider["quantizations"] = spec.quantizations
        if spec.ignore:
            provider["ignore"] = spec.ignore
        body["provider"] = provider

        if spec.reasoning_effort:
            body["reasoning"] = {"effort": spec.reasoning_effort}
        elif spec.reasoning_budget_tokens:
            body["reasoning"] = {"max_tokens": spec.reasoning_budget_tokens}

    # Caller kwargs win, key by key; `provider`/`reasoning` dicts win outright.
    body.update({k: v for k, v in kwargs.items() if k not in _CONTROL})
    return body


# ---------------------------------------------------------------------------
# Response handling.
# ---------------------------------------------------------------------------


def _finish(model: str, body: dict, r: httpx.Response) -> AttrDict:
    if r.status_code >= 400:
        raise RuntimeError(f"{model}: HTTP {r.status_code}: {r.text}")
    data = r.json()
    if "error" in data:  # OpenRouter can return 200 with an error body
        raise RuntimeError(f"{model}: {data['error']}")

    resp = AttrDict(data)
    served = data.get("provider", "") or ""
    usage = data.get("usage") or {}
    rtok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

    # `reasoning_tokens: 0` is legitimate — a model at low effort spends none on
    # an easy prompt — so only a missing field means the endpoint ignored us.
    if "reasoning" in body and rtok is None:
        raise RuntimeError(
            f"{model}: reasoning requested but the response reports no "
            f"reasoning_tokens at all — the endpoint likely ignored it"
        )

    resp._hidden_params = {
        "custom_llm_provider": "openrouter",
        "response_cost": usage.get("cost"),  # LiteLLM convention
        "provider": served,
        "request_body": body,
    }
    return resp


# ---------------------------------------------------------------------------
# Public API — LiteLLM call shape.
# ---------------------------------------------------------------------------


def completion(model: str, messages: list[dict], **kwargs: Any) -> AttrDict:
    body = _build(model, messages, kwargs)
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(kwargs.get("api_key")),
        json=body,
        timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
    )
    return _finish(model, body, r)


async def acompletion(model: str, messages: list[dict], **kwargs: Any) -> AttrDict:
    from httpx import AsyncClient  # async deps only when the async path is used

    body = _build(model, messages, kwargs)
    async with AsyncClient(timeout=kwargs.get("timeout", DEFAULT_TIMEOUT)) as client:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            headers=_headers(kwargs.get("api_key")),
            json=body,
        )
    return _finish(model, body, r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

USAGE = """usage: openrouter_completion.py <model-slug> < prompt.txt

Send a prompt (read from stdin) to one OpenRouter model and print the full
JSON response. The slug's entry in model_specs.json5, if it has one, supplies
the provider routing, sampling and reasoning settings.

  echo 'What is 17*23?' | ./openrouter_completion.py deepseek/deepseek-chat

Needs OPENROUTER_API_KEY in the environment."""


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(USAGE)
    prompt = sys.stdin.read().strip()
    if not prompt:
        sys.exit("no prompt on stdin")

    response = completion(sys.argv[1], [{"role": "user", "content": prompt}])
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
