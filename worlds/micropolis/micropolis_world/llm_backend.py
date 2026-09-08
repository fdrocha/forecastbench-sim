"""The one import boundary between micropolis and its LLM client.

Everything that talks to a model imports `acompletion` / `completion` /
`completion_cost` and the transient-error classes from here, so switching
backends is a matter of flipping which block below is live.

The two backends agree on: the call shape (`model`, `messages`, `timeout`),
`completion_cost(completion_response=...)`, the five exception classes the
retry loop in prompting.py catches, and `to_model_id()`, slug -> model id.

Callers hand `model` the slug (see model_ids.py: an OpenRouter id with an
optional `:suffix` picking a ModelSpec) and the backend translates internally;
`to_model_id` is exported for code that needs the underlying id, such as the
external-scores join.
"""

# Env vars the live backend needs beyond what GCP Secret Manager fills in.
# Checked once by module_globals.ensure_api_keys().
REQUIRED_ENV_KEYS = ("OPENROUTER_API_KEY",)

# --- OpenRouter (live) -----------------------------------------------------
from .openrouter_completion import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    acompletion,
    completion,
    completion_cost,
    to_model_id,
)

# --- LiteLLM (to switch back, comment out the block above and uncomment this,
# and set REQUIRED_ENV_KEYS = () since load_api_keys_from_gcp covers it)
# from litellm import (
#     APIConnectionError,
#     AuthenticationError,
#     InternalServerError,
#     RateLimitError,
#     ServiceUnavailableError,
#     Timeout,
#     acompletion,
#     completion,
#     completion_cost,
# )
#
# # Slugs LiteLLM cannot reach directly: no first-party provider in our key
# # set, so they go through its OpenRouter passthrough. Anything not listed
# # here LiteLLM prices and routes under the bare slug.
# _VIA_OPENROUTER = {
#     "meta-llama/llama-4-scout",
#     "deepseek/deepseek-chat",
#     "deepseek/deepseek-v4-flash-0731",
#     "qwen/qwen3-235b-a22b",
#     "qwen/qwen3.5-flash-02-23",
#     "moonshotai/kimi-k2",
# }
# # Bare slug -> the dated alias LiteLLM's price map is keyed on.
# _LITELLM_ALIASES = {"anthropic/claude-haiku-4.5": "anthropic/claude-haiku-4-5-20251001"}
#
# # Callers pass the slug and no longer translate, so this block would also
# # have to wrap litellm's completion/acompletion to apply to_model_id (and
# # the suffixed spec's reasoning settings) before the call.
# def to_model_id(slug: str) -> str:
#     """A slug turned into what LiteLLM expects.
#
#     Strips any :suffix, then LiteLLM needs the openrouter/ passthrough
#     prefix, its dated aliases, and gemini/ in place of google/.
#     """
#     model_id = slug.split(":", 1)[0]
#     if model_id in _VIA_OPENROUTER:
#         return f"openrouter/{model_id}"
#     model_id = _LITELLM_ALIASES.get(model_id, model_id)
#     if model_id.startswith("google/"):
#         return f"gemini/{model_id[len('google/'):]}"
#     return model_id


# The re-exports themselves: this module is a pass-through, so the names it
# imports are its public surface rather than unused imports. Both backend
# blocks above provide exactly these, which is what makes the swap a swap.
__all__ = (
    "APIConnectionError",
    "AuthenticationError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "acompletion",
    "completion",
    "completion_cost",
    "to_model_id",
)
