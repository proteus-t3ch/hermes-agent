"""Strict model runs on POST /v1/runs (``strict_model: true``).

Contract under test:
- ``strict_model`` must be a boolean; when true, ``provider`` and ``model``
  are both required (400 otherwise).
- Server-side selection state that would change the provider/model (a
  ``model_routes`` alias, a session ``/model`` override) is rejected (409),
  never applied.  An unresolvable provider is rejected before 202.
- The strict agent is built with no fallback chain, without consulting the
  global primary provider, and ``try_activate_fallback`` refuses to swap.
- Run status / ``run.completed`` report the runtime read from the agent
  after the run (``effective_provider`` / ``effective_model`` /
  ``fallback_activated``) plus ``requested_*``, ``strict_model`` and the
  client ``metadata`` echo.
- ``GET /v1/capabilities`` advertises ``features.strict_model_runs``.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _StrictModelViolation,
    _parse_strict_model_request,
    _sanitize_run_metadata,
    cors_middleware,
    security_headers_middleware,
)

STRICT_BODY = {
    "input": "hello",
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "strict_model": True,
    "metadata": {
        "execution_id": "exec-1",
        "correlation_id": "corr-1",
        "kind": "triage",
    },
}

ANTHROPIC_RUNTIME = {
    "provider": "anthropic",
    "api_key": "sk-test-anthropic",
    # The official endpoint: strict anthropic runs refuse proxies (v2).
    "base_url": "https://api.anthropic.com",
    "api_mode": "anthropic_messages",
}


def _make_adapter(extra=None) -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra=dict(extra or {})))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


class _RanAgent:
    """Plain (non-MagicMock) agent double so attribute reads are real."""

    def __init__(self, provider, model, *, fallback=False, cost=None, primary=None):
        self.provider = provider
        self.model = model
        self._fallback_activated = fallback
        self._primary_runtime = primary
        self.session_prompt_tokens = 3
        self.session_completion_tokens = 2
        self.session_total_tokens = 5
        if cost is not None:
            self.session_estimated_cost_usd = cost
        self.run_conversation = MagicMock(return_value={"final_response": "done"})

    def interrupt(self, message=None):
        pass


async def _wait_terminal(cli, run_id):
    status = {}
    for _ in range(60):
        resp = await cli.get(f"/v1/runs/{run_id}")
        status = await resp.json()
        if status.get("status") in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.05)
    return status


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "strict", "has_error"),
    [
        ({"input": "x"}, False, False),
        ({"strict_model": None}, False, False),
        ({"strict_model": False}, False, False),
        ({"strict_model": "true", "provider": "a", "model": "m"}, False, True),
        ({"strict_model": 1, "provider": "a", "model": "m"}, False, True),
        ({"strict_model": True, "model": "m"}, True, True),
        ({"strict_model": True, "provider": "a"}, True, True),
        ({"strict_model": True, "provider": "  ", "model": "m"}, True, True),
        ({"strict_model": True, "provider": "auto", "model": "m"}, True, True),
        ({"strict_model": True, "provider": "anthropic", "model": "m"}, True, False),
    ],
)
def test_parse_strict_model_request(body, strict, has_error):
    got_strict, error = _parse_strict_model_request(body)
    assert got_strict is strict
    assert bool(error) is has_error


def test_sanitize_run_metadata_keeps_bounded_string_pairs_only():
    raw = {
        "execution_id": "e" * 1000,
        "kind": "triage",
        "n": 3,
        "nested": {"a": "b"},
        "bad\nkey": "x",
        "ok": "line\nbreak",
    }
    raw.update({f"k{i}": "v" for i in range(40)})
    cleaned = _sanitize_run_metadata(raw)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in cleaned.items())
    assert "n" not in cleaned and "nested" not in cleaned
    assert "bad\nkey" not in cleaned and "ok" not in cleaned
    assert len(cleaned["execution_id"]) <= 256
    assert cleaned["kind"] == "triage"
    assert len(cleaned) <= 16
    assert _sanitize_run_metadata(["not", "a", "dict"]) == {}


# ---------------------------------------------------------------------------
# Synchronous rejection (nothing is queued)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"strict_model": "yes"},
        {"provider": None},
        {"model": None},
        {"provider": "auto"},
        # The advertised virtual model means "gateway default" — not exact.
        {"model": "hermes-agent"},
    ],
)
async def test_strict_validation_returns_400(overrides):
    adapter = _make_adapter()
    body = {k: v for k, v in {**STRICT_BODY, **overrides}.items() if v is not None}
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            resp = await cli.post("/v1/runs", json=body)
            data = await resp.json()
    assert resp.status == 400
    assert data["error"]["code"] == "invalid_strict_model"
    assert adapter._run_statuses == {}
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_strict_rejects_model_routes_alias_with_409():
    adapter = _make_adapter(
        {"model_routes": {"claude-haiku-4-5-20251001": {"model": "other/model", "provider": "openrouter"}}}
    )
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            data = await resp.json()
    assert resp.status == 409
    assert data["error"]["code"] == "strict_model_conflict"
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_strict_rejects_session_model_override_with_409():
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(
            adapter,
            "_session_model_override_for",
            return_value={"model": "gpt-5", "provider": "openai"},
        ), patch.object(adapter, "_create_agent") as mock_create:
            resp = await cli.post("/v1/runs", json={**STRICT_BODY, "session_id": "s1"})
            data = await resp.json()
    assert resp.status == 409
    assert data["error"]["code"] == "strict_model_conflict"
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_strict_unresolvable_provider_returns_409_before_202(monkeypatch):
    def _boom(*_a, **_kw):
        raise RuntimeError("no credentials for anthropic")

    monkeypatch.setattr("gateway.platforms.api_server._resolve_request_runtime_agent_kwargs", _boom)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", _boom)
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            data = await resp.json()
    assert resp.status == 409
    assert data["error"]["code"] == "strict_model_unavailable"
    assert adapter._run_statuses == {}
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_strict_provider_resolving_elsewhere_returns_409(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: {"provider": "openrouter"},
    )
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
    assert resp.status == 409
    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Accepted strict run: status + run.completed carry the effective runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_run_status_reports_effective_requested_and_metadata(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(ANTHROPIC_RUNTIME),
    )
    adapter = _make_adapter()
    agent = _RanAgent("anthropic", "claude-haiku-4-5-20251001", cost=0.0123)
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent) as mock_create:
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            status = await _wait_terminal(cli, run_id)

    assert mock_create.call_args.kwargs["strict_model"] is True
    assert mock_create.call_args.kwargs["requested_provider"] == "anthropic"
    assert mock_create.call_args.kwargs["requested_model"] == "claude-haiku-4-5-20251001"
    assert status["status"] == "completed"
    assert status["effective_provider"] == "anthropic"
    assert status["effective_model"] == "claude-haiku-4-5-20251001"
    assert status["fallback_activated"] is False
    assert status["requested_provider"] == "anthropic"
    assert status["requested_model"] == "claude-haiku-4-5-20251001"
    assert status["strict_model"] is True
    assert status["metadata"] == STRICT_BODY["metadata"]
    assert status["usage"]["estimated_cost_usd"] == pytest.approx(0.0123)
    # Legacy echo field is still present for backward compatibility.
    assert status["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_run_completed_event_carries_effective_runtime(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(ANTHROPIC_RUNTIME),
    )
    adapter = _make_adapter()
    agent = _RanAgent("anthropic", "claude-haiku-4-5-20251001")
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            run_id = (await resp.json())["run_id"]
            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            body = await events_resp.text()

    completed = None
    for line in body.splitlines():
        if line.startswith("data:") and "run.completed" in line:
            completed = json.loads(line[len("data:"):].strip())
    assert completed is not None
    assert completed["effective_provider"] == "anthropic"
    assert completed["effective_model"] == "claude-haiku-4-5-20251001"
    assert completed["fallback_activated"] is False


@pytest.mark.asyncio
async def test_status_reports_runtime_read_from_agent_not_request():
    """A non-strict run that fell back mid-run must show the fallback runtime."""
    adapter = _make_adapter()
    agent = _RanAgent(
        "openrouter",
        "fallback/model",
        fallback=True,
        primary={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    )
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hi", "provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            )
            run_id = (await resp.json())["run_id"]
            status = await _wait_terminal(cli, run_id)

    assert status["model"] == "claude-haiku-4-5-20251001"  # request echo
    assert status["effective_provider"] == "openrouter"
    assert status["effective_model"] == "fallback/model"
    assert status["fallback_activated"] is True
    assert status["strict_model"] is False


@pytest.mark.asyncio
async def test_failed_strict_run_reports_failure_without_effective_runtime(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(ANTHROPIC_RUNTIME),
    )
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(
            adapter,
            "_create_agent",
            side_effect=_StrictModelViolation("strict model run conflicts with a session /model override"),
        ):
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            status = await _wait_terminal(cli, run_id)

    assert status["status"] == "failed"
    assert "strict model run" in status["error"]
    assert status["strict_model"] is True
    assert status["metadata"] == STRICT_BODY["metadata"]
    # No agent was built, so nothing executed: no effective runtime claimed.
    assert "effective_provider" not in status


@pytest.mark.asyncio
async def test_non_strict_run_does_not_pass_strict_flag():
    adapter = _make_adapter()
    agent = _RanAgent("minimax", "MiniMax-M3")
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent) as mock_create:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hi", "provider": "minimax", "model": "MiniMax-M3"},
            )
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            status = await _wait_terminal(cli, run_id)

    assert "strict_model" not in mock_create.call_args.kwargs
    assert status["status"] == "completed"
    assert status["strict_model"] is False
    assert status["metadata"] == {}
    assert "estimated_cost_usd" not in status["usage"]


@pytest.mark.asyncio
async def test_capabilities_advertise_strict_model_runs():
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.get("/v1/capabilities")
        data = await resp.json()
    assert data["features"]["strict_model_runs"] is True


# ---------------------------------------------------------------------------
# _create_agent strict construction
# ---------------------------------------------------------------------------


def _patch_runtime(monkeypatch, *, request_runtime=None):
    def _global_primary_must_not_be_called():
        raise AssertionError("strict runs must not resolve the global primary")

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", _global_primary_must_not_be_called)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: [{"provider": "openrouter", "model": "fallback/model"}]),
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(request_runtime or ANTHROPIC_RUNTIME),
    )


def _fake_agent_class(captured, *, provider=None, model=None):
    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.provider = provider or kwargs.get("provider")
            self.model = model or kwargs.get("model")
            self._fallback_chain = list(kwargs.get("fallback_model") or [])

    return FakeAgent


def test_create_agent_strict_has_no_fallback_and_skips_global_primary(monkeypatch):
    _patch_runtime(monkeypatch)
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    agent = adapter._create_agent(
        session_id="s1",
        requested_provider="anthropic",
        requested_model="claude-haiku-4-5-20251001",
        strict_model=True,
    )

    assert captured["fallback_model"] is None
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["base_url"] == ANTHROPIC_RUNTIME["base_url"]
    assert agent._strict_runtime is True
    assert agent._fallback_chain == []
    assert agent._hermes_api_runtime["route_source"] == "strict_request"


def test_create_agent_non_strict_still_gets_fallback_chain(monkeypatch):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "openrouter", "api_key": "sk-g", "base_url": "https://or.example/v1"},
    )
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    agent = adapter._create_agent(
        session_id="s1",
        requested_provider="anthropic",
        requested_model="claude-haiku-4-5-20251001",
    )

    assert captured["fallback_model"] == [{"provider": "openrouter", "model": "fallback/model"}]
    assert getattr(agent, "_strict_runtime", False) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"route": {"model": "other/model", "provider": "openrouter"}},
        {"session_model": "other/model"},
        {"confirmed_runtime_lock": True},
    ],
)
def test_create_agent_strict_rejects_server_side_selection(monkeypatch, kwargs):
    _patch_runtime(monkeypatch)
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    with pytest.raises(_StrictModelViolation):
        adapter._create_agent(
            session_id="s1",
            requested_provider="anthropic",
            requested_model="claude-haiku-4-5-20251001",
            strict_model=True,
            **kwargs,
        )
    assert captured == {}


def test_create_agent_strict_rejects_session_override(monkeypatch):
    _patch_runtime(monkeypatch)
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(
        adapter, "_session_model_override_for", lambda key: {"model": "gpt-5", "provider": "openai"}
    )

    with pytest.raises(_StrictModelViolation):
        adapter._create_agent(
            session_id="s1",
            requested_provider="anthropic",
            requested_model="claude-haiku-4-5-20251001",
            strict_model=True,
        )
    assert captured == {}


def test_create_agent_strict_fails_when_provider_resolves_elsewhere(monkeypatch):
    _patch_runtime(monkeypatch, request_runtime={"provider": "openrouter", "api_key": "k"})
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    with pytest.raises(_StrictModelViolation):
        adapter._create_agent(
            session_id="s1",
            requested_provider="anthropic",
            requested_model="claude-haiku-4-5-20251001",
            strict_model=True,
        )
    assert captured == {}


def test_create_agent_strict_fails_when_agent_initialises_on_other_model(monkeypatch):
    _patch_runtime(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        "run_agent.AIAgent", _fake_agent_class(captured, model="claude-sonnet-normalised")
    )
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    with pytest.raises(_StrictModelViolation):
        adapter._create_agent(
            session_id="s1",
            requested_provider="anthropic",
            requested_model="claude-haiku-4-5-20251001",
            strict_model=True,
        )


# ---------------------------------------------------------------------------
# try_activate_fallback refuses for strict agents
# ---------------------------------------------------------------------------


def test_try_activate_fallback_refuses_for_strict_agent():
    from agent.chat_completion_helpers import try_activate_fallback
    from agent.error_classifier import FailoverReason

    class Agent:
        pass

    agent = Agent()
    agent._strict_runtime = True
    agent.provider = "anthropic"
    agent.model = "claude-haiku-4-5-20251001"
    agent._fallback_activated = False
    agent._primary_runtime = {"provider": "anthropic"}
    # A non-empty chain would normally be walked; strict must ignore it.
    agent._fallback_chain = [{"provider": "openrouter", "model": "fallback/model"}]
    agent._fallback_index = 0

    for reason in (None, FailoverReason.rate_limit, FailoverReason.model_not_found):
        assert try_activate_fallback(agent, reason) is False

    assert agent.provider == "anthropic"
    assert agent.model == "claude-haiku-4-5-20251001"
    assert agent._fallback_index == 0
    assert agent._fallback_activated is False


# ---------------------------------------------------------------------------
# v2 — run-binding 202 echo, no auxiliary calls, official endpoint only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_202_echoes_the_accepted_request(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(ANTHROPIC_RUNTIME),
    )
    adapter = _make_adapter()
    agent = _RanAgent("anthropic", "claude-haiku-4-5-20251001")
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            data = await resp.json()
            await _wait_terminal(cli, data["run_id"])
    assert resp.status == 202
    assert data["strict_model"] is True
    assert data["requested_provider"] == "anthropic"
    assert data["requested_model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_non_strict_202_carries_no_echo():
    adapter = _make_adapter()
    agent = _RanAgent("nous", "z-ai/glm-5.2")
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            resp = await cli.post("/v1/runs", json={"input": "hi"})
            data = await resp.json()
            await _wait_terminal(cli, data["run_id"])
    assert resp.status == 202
    assert "strict_model" not in data
    assert "requested_model" not in data


@pytest.mark.asyncio
async def test_strict_refuses_a_proxy_endpoint_before_202(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: {**ANTHROPIC_RUNTIME, "base_url": "https://litellm.internal/v1"},
    )
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            data = await resp.json()
    assert resp.status == 409
    assert data["error"]["code"] == "strict_model_unavailable"
    mock_create.assert_not_called()


@pytest.mark.parametrize("url,ok", [
    ("", True),
    ("https://api.anthropic.com", True),
    ("https://api.anthropic.com/v1/", True),
    ("http://api.anthropic.com", False),
    ("https://api.anthropic.com.attacker.test", False),
    ("https://litellm.internal/v1", False),
])
def test_strict_endpoint_allowed(url, ok):
    from gateway.platforms.api_server import _strict_endpoint_allowed

    assert _strict_endpoint_allowed("anthropic", url) is ok
    # Other providers are not constrained (documented limitation).
    assert _strict_endpoint_allowed("openrouter", "https://anything.example") is True


def test_create_agent_strict_disables_background_review_and_compression(monkeypatch):
    _patch_runtime(monkeypatch)
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    agent = adapter._create_agent(
        session_id="s1", requested_provider="anthropic",
        requested_model="claude-haiku-4-5-20251001", strict_model=True,
    )
    assert captured["skip_background_review"] is True
    assert agent.compression_enabled is False
    assert agent._aux_models_used == []


def test_create_agent_non_strict_keeps_background_review_and_compression(monkeypatch):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: dict(ANTHROPIC_RUNTIME))
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", _fake_agent_class(captured))
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    agent = adapter._create_agent(session_id="s1")
    assert "skip_background_review" not in captured
    assert not hasattr(agent, "compression_enabled") or agent.compression_enabled is not False


def test_create_agent_strict_rejects_an_agent_on_a_proxy_endpoint(monkeypatch):
    from gateway.platforms.api_server import _StrictModelViolation

    _patch_runtime(monkeypatch)
    captured = {}

    class ProxyAgent(_fake_agent_class(captured)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.base_url = "https://litellm.internal/v1"

    monkeypatch.setattr("run_agent.AIAgent", ProxyAgent)
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    with pytest.raises(_StrictModelViolation):
        adapter._create_agent(
            session_id="s1", requested_provider="anthropic",
            requested_model="claude-haiku-4-5-20251001", strict_model=True,
        )


def test_auxiliary_call_is_refused_and_recorded_inside_a_strict_run():
    from agent import auxiliary_client as aux

    class A:
        pass

    agent = A()
    token = aux.enter_strict_aux_scope(agent)
    try:
        with pytest.raises(aux.StrictAuxiliaryViolation):
            aux.call_llm("compression", provider="openrouter", model="google/gemini-flash", messages=[])
    finally:
        aux.exit_strict_aux_scope(token)
    assert agent._aux_models_used == [{"provider": "openrouter", "model": "google/gemini-flash"}]
    assert "strict model run" in agent._strict_aux_violation
    # Outside the scope the guard is inert.
    aux._refuse_auxiliary_in_strict_run("compression", "openrouter", "x")


def test_async_auxiliary_call_is_refused_inside_a_strict_run():
    from agent import auxiliary_client as aux

    class A:
        pass

    agent = A()

    async def go():
        token = aux.enter_strict_aux_scope(agent)
        try:
            await aux.async_call_llm("title", messages=[])
        finally:
            aux.exit_strict_aux_scope(token)

    with pytest.raises(aux.StrictAuxiliaryViolation):
        asyncio.run(go())
    assert agent._aux_models_used == [{"provider": "auto", "model": "auto"}]


@pytest.mark.asyncio
async def test_strict_status_reports_aux_models_used(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(ANTHROPIC_RUNTIME),
    )
    adapter = _make_adapter()
    agent = _RanAgent("anthropic", "claude-haiku-4-5-20251001")
    agent._strict_runtime = True
    agent._aux_models_used = []
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            status = await _wait_terminal(cli, (await resp.json())["run_id"])
    assert status["aux_models_used"] == []


@pytest.mark.asyncio
async def test_strict_run_thread_enters_the_auxiliary_guard(monkeypatch):
    """The scope is active WHILE the agent runs, so any auxiliary call it makes is refused."""
    from agent import auxiliary_client as aux

    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: dict(ANTHROPIC_RUNTIME),
    )
    seen = {}
    adapter = _make_adapter()
    agent = _RanAgent("anthropic", "claude-haiku-4-5-20251001")
    agent._strict_runtime = True
    agent._aux_models_used = []

    def run_conversation(**_kw):
        seen["scope_agent"] = aux._STRICT_AUX_AGENT.get()
        return {"final_response": "done"}

    agent.run_conversation = run_conversation
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            resp = await cli.post("/v1/runs", json=STRICT_BODY)
            await _wait_terminal(cli, (await resp.json())["run_id"])
    assert seen["scope_agent"] is agent
    assert aux._STRICT_AUX_AGENT.get() is None
