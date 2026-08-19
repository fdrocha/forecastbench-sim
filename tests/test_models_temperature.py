"""Sampling protocol tests: no temperature is sent by default anywhere;
explicit overrides still work (and are withheld from models that reject the
parameter)."""
import asyncio
import json
from types import SimpleNamespace

from conftest import load_script
from fbsim_core.evaluation import models

elicit = load_script("scripts/uplift_v2/natcond_elicit_v2.py")


def _fake_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_default_sends_no_temperature(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(models, "completion", fake_completion)
    m = models.LiteLLMModel("openai/gpt-4o")
    assert m.get_response("hi") == "ok"
    assert "temperature" not in captured
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    # denylist models behave identically on the default path (no gating)
    models.LiteLLMModel("openai/o3").get_response("hi")
    assert "temperature" not in captured
    # max_tokens handling is unchanged
    m.get_response("hi", max_tokens=64)
    assert captured["max_tokens"] == 64 and "temperature" not in captured


def test_explicit_override_still_sends_temperature(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(models, "completion", fake_completion)
    models.LiteLLMModel("openai/gpt-4o").get_response("hi", temperature=0.0)
    assert captured["temperature"] == 0.0  # legacy opt-in behavior
    # the override is withheld from models that reject the parameter
    models.LiteLLMModel("openai/o3").get_response("hi", temperature=0.7)
    assert "temperature" not in captured


def test_async_path_matches(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(models, "acompletion", fake_acompletion)
    m = models.LiteLLMModel("anthropic/claude-sonnet-4-5")
    assert asyncio.run(m.get_response_async("hi")) == "ok"
    assert "temperature" not in captured
    asyncio.run(m.get_response_async("hi", temperature=1.0))
    assert captured["temperature"] == 1.0


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self, *a):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture_chat_body(monkeypatch, provider):
    import urllib.request
    monkeypatch.setenv("CHAT_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_PROVIDER", provider)
    captured = {}
    payload = ({"content": [{"text": "<probability>0.5</probability>"}]}
               if provider == "anthropic" else
               {"choices": [{"message": {"content": "<probability>0.5</probability>"}}]})

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        captured["url"] = req.full_url
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = elicit.chat("some-model", [{"role": "user", "content": "hi"}])
    assert out == "<probability>0.5</probability>"
    return captured


def test_elicit_chat_sends_no_temperature_openai_compat(monkeypatch):
    captured = _capture_chat_body(monkeypatch, "openai")
    body = captured["body"]
    assert "temperature" not in body
    assert body["model"] == "some-model"
    assert body["max_tokens"] == 8192  # fireworks default budget unchanged


def test_elicit_chat_sends_no_temperature_anthropic(monkeypatch):
    captured = _capture_chat_body(monkeypatch, "anthropic")
    body = captured["body"]
    assert "temperature" not in body
    assert body["max_tokens"] == 16000
    assert "anthropic.com" in captured["url"]
