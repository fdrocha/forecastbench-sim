#!/usr/bin/env -S uv run python3
"""Drop-in `completion` / `acompletion` / `completion_cost` for OpenRouter.

    from orouter_litellm import completion, acompletion, completion_cost

Same call shape as LiteLLM. Differences that matter:

* `model` is an OpenRouter slug (an `openrouter/` prefix is tolerated). If the
  slug is in MODELS — loaded at import from `model_specs.json5` beside this
  file — it runs pinned to that endpoint; otherwise it runs unpinned with a
  warning.
* `reasoning_effort` is REQUIRED for models that can reason. Effort-native
  models get `reasoning.effort`; budget-only models get a fixed
  `reasoning.max_tokens` from EFFORT_BUDGET. `max_tokens` is never required.
* Caller kwargs override the spec's sampling defaults. A caller-supplied
  `provider` dict is merged over the spec's; a caller-supplied `reasoning`
  dict bypasses the effort mapping entirely.
* The response mirrors LiteLLM's ModelResponse for attribute access, keeps
  every OpenRouter field in place (`provider`, `usage.cost`, ...), and puts
  provenance under `_hidden_params`.
"""

import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import json5

import micropolis_world.module_globals as g

BASE_URL = "https://openrouter.ai/api/v1"
SNAPSHOT_DIR = g.DATA_DIR / "openrouter_model_snapshots"
DEFAULT_TIMEOUT = 600.0
EFFORT_BUDGET = {"low": 1024, "medium": 2048, "high": 4096}  # budget-only models

# ---------------------------------------------------------------------------
# Auth — resolved at call time, never at import.
# ---------------------------------------------------------------------------


def _headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# Pinned model registry, from model_specs.json5. Anything not there runs
# unpinned (with a warning).
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    endpoint: str  # FULL provider slug, e.g. "deepinfra/turbo".
    # A base slug matches every variant.
    quantizations: list[str] | None = (
        None  # e.g. ["fp8"]; None for first-party proprietary
    )
    ignore: list[str] = field(
        default_factory=list
    )  # variants to exclude when pinning a
    # provider's default endpoint by base slug
    sampling: dict[str, Any] = field(default_factory=dict)  # {} where the model rejects
    # sampling params (o-series, Claude w/ thinking)


MODEL_SPECS_PATH = Path(__file__).with_name("model_specs.json5")


def load_model_specs(path: Path = MODEL_SPECS_PATH) -> dict[str, ModelSpec]:
    """Read the pinned registry; empty (with a warning) if the file is absent."""
    if not path.exists():
        warnings.warn(
            f"{path.name} not found at {path} — MODELS is empty, "
            f"every model will run UNPINNED",
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
# Pre-flight: resolve the pin, classify reasoning, snapshot, diff. Cached.
# ---------------------------------------------------------------------------

_PREFLIGHT: dict[str, dict] = {}


def _norm_slug(model: str) -> str:
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


def _fetch_endpoints(slug: str, api_key: str | None) -> list[dict]:
    author, _, name = slug.partition("/")
    r = httpx.get(
        f"{BASE_URL}/models/{author}/{name}/endpoints",
        headers=_headers(api_key),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]["endpoints"]


def _endpoint_slug(ep: dict) -> str:
    # If this returns "" for you, print(ep.keys()) and adjust.
    return ep.get("tag") or ep.get("provider_slug") or ep.get("slug") or ""


def _stable(ep: dict) -> dict:
    """Metadata whose change means 'different artifact'."""
    return {
        "slug": _endpoint_slug(ep),
        "provider_name": ep.get("provider_name"),
        "quantization": ep.get("quantization"),
        "context_length": ep.get("context_length"),
        "max_completion_tokens": ep.get("max_completion_tokens"),
        "supported_parameters": sorted(ep.get("supported_parameters") or []),
        "reasoning": ep.get("reasoning"),
    }


def _reasoning_mode(rinfo: dict | None) -> str:
    """effort | budget | fixed | none, from an endpoint's `reasoning` object."""
    if not rinfo:
        return "none"
    if "supported_efforts" in rinfo:  # None here means "all values accepted"
        return "effort"
    if rinfo.get("supports_max_tokens"):
        return "budget"
    return "fixed"  # reasons, but exposes no control


def preflight(model: str, api_key: str | None = None) -> dict:
    slug = _norm_slug(model)
    if slug in _PREFLIGHT:
        return _PREFLIGHT[slug]

    eps = _fetch_endpoints(slug, api_key)
    spec = MODELS.get(slug)

    if spec is None:
        warnings.warn(
            f"{slug}: not in MODELS registry — running UNPINNED "
            f"(provider chosen by OpenRouter's load balancer)",
            stacklevel=3,
        )
        # Classify reasoning from whatever the endpoints collectively advertise.
        rinfos = [e.get("reasoning") for e in eps if e.get("reasoning")]
        rinfo = (
            next((r for r in rinfos if "supported_efforts" in r), None)
            or next((r for r in rinfos if r.get("supports_max_tokens")), None)
            or (rinfos[0] if rinfos else None)
        )
        pf = {
            "slug": slug,
            "spec": None,
            "endpoint": None,
            "rinfo": rinfo,
            "mode": _reasoning_mode(rinfo),
        }
        _PREFLIGHT[slug] = pf
        return pf

    exact = [e for e in eps if _endpoint_slug(e) == spec.endpoint]
    matches = exact or [
        e
        for e in eps
        if _endpoint_slug(e).split("/")[0] == spec.endpoint
        and _endpoint_slug(e) not in spec.ignore
    ]
    if spec.quantizations:
        matches = [e for e in matches if e.get("quantization") in spec.quantizations]
    if len(matches) != 1:
        found = [(_endpoint_slug(e), e.get("quantization")) for e in eps]
        raise RuntimeError(
            f"{slug}: pin {spec.endpoint!r} q={spec.quantizations} "
            f"ignore={spec.ignore} resolved to {len(matches)} endpoints, "
            f"need exactly 1. Available: {found}"
        )
    meta = _stable(matches[0])

    params = set(meta["supported_parameters"])
    for p in spec.sampling:
        if params and p not in params:
            raise RuntimeError(
                f"{slug}: pinned endpoint doesn't list sampling param {p!r}"
            )

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    safe = slug.replace("/", "__")
    prior = sorted(SNAPSHOT_DIR.glob(f"{safe}.*.json"))
    if prior:
        old = json.loads(prior[-1].read_text())
        if old != meta:
            raise RuntimeError(
                f"{slug}: endpoint metadata drifted since {prior[-1].name}:\n"
                f"  was {old}\n  now {meta}"
            )
    (SNAPSHOT_DIR / f"{safe}.{int(time.time())}.json").write_text(
        json.dumps(meta, indent=1)
    )

    pf = {
        "slug": slug,
        "spec": spec,
        "endpoint": meta,
        "rinfo": meta["reasoning"],
        "mode": _reasoning_mode(meta["reasoning"]),
    }
    _PREFLIGHT[slug] = pf
    return pf


# ---------------------------------------------------------------------------
# Request body.
# ---------------------------------------------------------------------------

_CONTROL = {"reasoning_effort", "fetch_generation", "api_key", "timeout", "stream"}
_LITELLM_ONLY = {  # accepted for call-site compatibility, dropped with a warning
    "drop_params",
    "num_retries",
    "max_retries",
    "metadata",
    "custom_llm_provider",
    "api_base",
    "api_version",
    "mock_response",
    "caching",
    "fallbacks",
    "allowed_openai_params",
    "additional_drop_params",
    "litellm_call_id",
    "litellm_logging_obj",
}


def _build(pf: dict, messages: list[dict], kwargs: dict) -> dict:
    if kwargs.get("stream"):
        raise NotImplementedError("stream=True is not supported by this module")
    dropped = _LITELLM_ONLY & kwargs.keys()
    if dropped:
        warnings.warn(f"ignoring LiteLLM-only kwargs: {sorted(dropped)}", stacklevel=3)

    spec: ModelSpec | None = pf["spec"]
    provider: dict[str, Any] = {
        "require_parameters": True,  # refuse endpoints that would drop our params
        "data_collection": "deny",  # hold the candidate pool constant
    }
    if spec:
        provider.update({"only": [spec.endpoint], "allow_fallbacks": False})
        if spec.quantizations:
            provider["quantizations"] = spec.quantizations
        if spec.ignore:
            provider["ignore"] = spec.ignore
    provider.update(kwargs.get("provider") or {})  # caller wins, key by key

    body: dict[str, Any] = {
        "model": pf["slug"],  # bare slug — :nitro/:floor change routing
        "messages": messages,
        "transforms": [],  # disable middle-out compression explicitly
        **(spec.sampling if spec else {}),
    }
    body.update(
        {
            k: v
            for k, v in kwargs.items()
            if k not in _CONTROL | _LITELLM_ONLY | {"provider"}
        }
    )
    body["provider"] = provider

    # Reasoning: caller-supplied dict bypasses everything below.
    if "reasoning" not in body:
        effort = kwargs.get("reasoning_effort")
        mode = pf["mode"]
        if mode == "effort":
            if effort is None:
                raise ValueError(f"{pf['slug']}: reasoning_effort is required")
            allowed = pf["rinfo"].get("supported_efforts")
            if allowed is not None and effort not in allowed:
                raise ValueError(f"{pf['slug']}: effort {effort!r} not in {allowed}")
            body["reasoning"] = {"effort": effort}
        elif mode == "budget":
            if effort is None:
                raise ValueError(f"{pf['slug']}: reasoning_effort is required")
            if effort not in EFFORT_BUDGET:
                raise ValueError(
                    f"{pf['slug']}: budget-only model; effort must be one of "
                    f"{sorted(EFFORT_BUDGET)}, got {effort!r}"
                )
            body["reasoning"] = {"max_tokens": EFFORT_BUDGET[effort]}
        elif effort is not None:
            warnings.warn(
                f"{pf['slug']}: reasoning_effort ignored "
                f"(model exposes no reasoning control: mode={mode})",
                stacklevel=3,
            )
    return body


# ---------------------------------------------------------------------------
# Response handling.
# ---------------------------------------------------------------------------


def _fetch_generation(gen_id: str, api_key: str | None) -> dict:
    for _ in range(5):  # the record lags the response slightly
        r = httpx.get(
            f"{BASE_URL}/generation",
            params={"id": gen_id},
            headers=_headers(api_key),
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("data", {})
        time.sleep(1)
    return {}


def _finish(
    pf: dict, body: dict, r: httpx.Response, fetch_generation: bool, api_key: str | None
) -> AttrDict:
    if r.status_code >= 400:
        raise RuntimeError(f"{pf['slug']}: HTTP {r.status_code}: {r.text}")
    data = r.json()
    if "error" in data:  # OpenRouter can return 200 with an error body
        raise RuntimeError(f"{pf['slug']}: {data['error']}")

    resp = AttrDict(data)
    served = data.get("provider", "") or ""
    usage = data.get("usage") or {}
    rtok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

    meta = pf["endpoint"]
    if (
        meta
        and meta["provider_name"]
        and meta["provider_name"].lower() not in served.lower()
    ):
        raise RuntimeError(
            f"{pf['slug']}: pinned {meta['provider_name']!r}, served by {served!r}"
        )
    if "reasoning" in body and not rtok:
        raise RuntimeError(
            f"{pf['slug']}: reasoning requested but reasoning_tokens={rtok!r}"
        )

    hidden: dict[str, Any] = {
        "custom_llm_provider": "openrouter",
        "response_cost": usage.get("cost"),  # LiteLLM convention
        "upstream_inference_cost": (usage.get("cost_details") or {}).get(
            "upstream_inference_cost"
        ),
        "provider": served,
        "pinned": meta is not None,
        "endpoint_meta": meta,
        "reasoning_mode": pf["mode"],
        "request_body": body,
        "request_sha": hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()[:16],
    }
    if fetch_generation:
        gen = _fetch_generation(data.get("id", ""), api_key)
        hidden["generation"] = {
            k: gen.get(k)
            for k in (
                "provider_name",
                "native_tokens_reasoning",
                "native_tokens_prompt",
                "native_tokens_completion",
                "native_finish_reason",
                "is_byok",
                "total_cost",
                "upstream_inference_cost",
                "cache_discount",
                "latency",
            )
        }
    resp._hidden_params = hidden
    return resp


# ---------------------------------------------------------------------------
# Public API — LiteLLM call shape.
# ---------------------------------------------------------------------------


def completion(model: str, messages: list[dict], **kwargs: Any) -> AttrDict:
    api_key = kwargs.get("api_key")
    pf = preflight(model, api_key=api_key)
    body = _build(pf, messages, kwargs)
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(api_key),
        json=body,
        timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
    )
    return _finish(pf, body, r, bool(kwargs.get("fetch_generation")), api_key)


async def acompletion(model: str, messages: list[dict], **kwargs: Any) -> AttrDict:
    import asyncio  # async deps only when the async path is used
    from httpx import AsyncClient

    api_key = kwargs.get("api_key")
    pf = await asyncio.to_thread(preflight, model, api_key)
    body = _build(pf, messages, kwargs)
    async with AsyncClient(timeout=kwargs.get("timeout", DEFAULT_TIMEOUT)) as client:
        r = await client.post(
            f"{BASE_URL}/chat/completions", headers=_headers(api_key), json=body
        )
    return await asyncio.to_thread(
        _finish, pf, body, r, bool(kwargs.get("fetch_generation")), api_key
    )


# ---------------------------------------------------------------------------
# CLI:  python orouter_litellm.py <model-slug> [effort] < prompt.txt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_arg = sys.argv[1]
    effort_arg = sys.argv[2] if len(sys.argv) > 2 else "medium"
    prompt = sys.stdin.read().strip()
    if not prompt:
        sys.exit("no prompt on stdin")

    response = completion(
        model_arg,
        [{"role": "user", "content": prompt}],
        reasoning_effort=effort_arg,
        fetch_generation=True,
    )
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
