"""Strict-model runs: the auxiliary-call guard survives worker-thread boundaries.

The guard (``agent.auxiliary_client._STRICT_AUX_AGENT``) is a ContextVar entered in the
run thread.  A plain ``threading.Thread`` or ``executor.submit(fn)`` does not inherit
ContextVars, so an auxiliary call reached across such a boundary ran UNGUARDED.

Concrete instance found: the turn prologue's auto-title (agent.turn_context ->
agent.title_generator.maybe_auto_title) started a plain thread that called the
auxiliary title model; ``api_server`` is not an untitled platform, so every strict run
sent its input to the auxiliary title model, outside the pinned contract and unrecorded.

Fix, three run-local legs (no process-global state):
  1. strict agents never start the auto-title (agent.turn_context);
  2. the auto-title thread runs inside a context copied AT SUBMISSION (ctx.run);
  3. a strict agent's runtime dict carries ``strict_model_run`` and call_llm /
     async_call_llm refuse on it even when the ContextVar was not inherited.

Every test stubs the step right after the guard, so even an unfixed guard can never
reach a provider or the network.
"""
import asyncio
import concurrent.futures
import contextvars
import threading

import pytest

from agent import auxiliary_client as aux

REACHED = "auxiliary call reached the provider path"


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch):
    """Anything that gets PAST the guard records itself and stops — no network, ever."""
    reached = []

    def _sync(task):
        reached.append(task)
        raise RuntimeError(REACHED)

    async_reached = _sync  # same recorder for the async entry point
    monkeypatch.setattr(aux, "_acquire_sync_aux_semaphore", _sync)
    monkeypatch.setattr(aux, "_acquire_async_aux_semaphore", async_reached)
    return reached


class _Agent:
    def __init__(self, strict=True, name="a", platform="api_server"):
        self.name = name
        self._strict_runtime = strict
        self.platform = platform
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"
        self.base_url = "https://api.anthropic.com"
        self.api_key = "k"
        self.api_mode = "anthropic_messages"
        self._session_db_created = True


def _call(task="compression", **kw):
    """Run call_llm and classify: 'refused' by the guard, or 'reached' the provider path."""
    try:
        aux.call_llm(task, messages=[], **kw)
    except aux.StrictAuxiliaryViolation:
        return "refused"
    except RuntimeError as e:
        if REACHED in str(e):
            return "reached"
        raise
    return "returned"


def _wait_title_threads():
    for t in threading.enumerate():
        if t.name == "auto-title":
            t.join(timeout=2)


# ---------------------------------------------------------------------------
# 1. direct, same context
# ---------------------------------------------------------------------------

def test_direct_same_context_call_is_refused_and_recorded(_no_provider):
    agent = _Agent()
    token = aux.enter_strict_aux_scope(agent)
    try:
        assert _call() == "refused"
    finally:
        aux.exit_strict_aux_scope(token)
    assert agent._aux_models_used == [{"provider": "auto", "model": "auto"}]
    assert _no_provider == []


# ---------------------------------------------------------------------------
# 2. worker-thread boundaries
# ---------------------------------------------------------------------------

def test_propagated_worker_thread_keeps_the_guard(_no_provider):
    from tools.thread_context import propagate_context_to_thread

    agent = _Agent()
    out = []
    token = aux.enter_strict_aux_scope(agent)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            out.append(ex.submit(propagate_context_to_thread(_call)).result())
    finally:
        aux.exit_strict_aux_scope(token)
    assert out == ["refused"]
    assert _no_provider == []


def test_plain_worker_thread_is_refused_through_the_strict_runtime(_no_provider):
    """No ContextVar crosses a plain thread — the run-local runtime marker still refuses."""
    rt = {"model": "claude-haiku-4-5-20251001", "provider": "anthropic", "strict_model_run": True}
    out = []
    t = threading.Thread(target=lambda: out.append(_call(main_runtime=rt)))
    t.start(); t.join()
    assert out == ["refused"]
    assert _no_provider == []


def test_async_entry_point_refuses_through_the_strict_runtime(_no_provider):
    rt = {"strict_model_run": True}

    async def go():
        await aux.async_call_llm("title", messages=[], main_runtime=rt)

    with pytest.raises(aux.StrictAuxiliaryViolation):
        asyncio.run(go())
    assert _no_provider == []


def test_strict_run_starts_no_auto_title(monkeypatch, _no_provider):
    """The concrete leak: a strict run's turn prologue must not dispatch the title call."""
    from agent import title_generator, turn_context

    dispatched = []
    monkeypatch.setattr(title_generator, "maybe_auto_title", lambda *a, **k: dispatched.append(True))
    msgs = [{"role": "user", "content": "Analyse this project for schedule risk."}]

    def agent(strict):
        a = _Agent(strict=strict)
        a._session_db = object()      # pass the session gates, so the strict check is what decides
        a.session_id = "run_1"
        return a

    # Control: an identical NON-strict agent reaches the dispatch — the test is not vacuous.
    turn_context._maybe_title_session_at_turn_start(agent(False), msgs)
    assert dispatched == [True], "control failed: the test never reaches the title dispatch"
    dispatched.clear()
    turn_context._maybe_title_session_at_turn_start(agent(True), msgs)
    assert dispatched == [], "a strict run sent its input to the auxiliary title model"


def test_title_thread_runs_inside_the_strict_guard(monkeypatch):
    """Defence in depth: if a title thread is started inside a strict scope, it sees the guard."""
    from agent import title_generator

    seen = {}
    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(title_generator, "apply_instant_title", lambda *a, **k: None)
    monkeypatch.setattr(title_generator, "_session_is_untitled", lambda *a, **k: True)
    monkeypatch.setattr(title_generator, "auto_title_session",
                        lambda *a, **k: seen.setdefault("agent", aux._STRICT_AUX_AGENT.get()))
    agent = _Agent()
    token = aux.enter_strict_aux_scope(agent)
    try:
        title_generator.maybe_auto_title(object(), "run_1", "Analyse this project.", conversation_history=[])
    finally:
        aux.exit_strict_aux_scope(token)
    _wait_title_threads()
    assert seen.get("agent") is agent, "the title thread ran outside the strict guard"


# ---------------------------------------------------------------------------
# 3-5. concurrency: no bleed between runs, no stale context on reused workers
# ---------------------------------------------------------------------------

def test_concurrent_strict_runs_record_only_their_own_attempts(_no_provider):
    a, b = _Agent(name="a"), _Agent(name="b")
    barrier = threading.Barrier(2)
    results = {}

    def run(agent, task):
        token = aux.enter_strict_aux_scope(agent)
        try:
            barrier.wait()          # both scopes active at once
            results[agent.name] = _call(task)
        finally:
            aux.exit_strict_aux_scope(token)

    ta = threading.Thread(target=run, args=(a, "compression"))
    tb = threading.Thread(target=run, args=(b, "title"))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert results == {"a": "refused", "b": "refused"}
    assert a._aux_models_used == [{"provider": "auto", "model": "auto"}]
    assert b._aux_models_used == [{"provider": "auto", "model": "auto"}]
    assert a._strict_aux_violation.startswith("auxiliary call (compression)")
    assert b._strict_aux_violation.startswith("auxiliary call (title)")


def test_ordinary_run_beside_a_strict_run_is_not_governed(_no_provider):
    strict = _Agent()
    in_scope = threading.Event(); done = threading.Event()
    out = {}

    def strict_run():
        token = aux.enter_strict_aux_scope(strict)
        try:
            in_scope.set(); done.wait(2)
        finally:
            aux.exit_strict_aux_scope(token)

    t = threading.Thread(target=strict_run); t.start(); in_scope.wait(2)
    out["ordinary"] = _call(main_runtime={"model": "z-ai/glm-5.2", "provider": "nous"})
    done.set(); t.join()
    assert out == {"ordinary": "reached"}, "a non-strict run inherited another run's strict policy"
    assert strict._aux_models_used == []


def test_reused_executor_worker_keeps_no_stale_context(_no_provider):
    agent = _Agent()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:   # one reused worker thread
        token = aux.enter_strict_aux_scope(agent)
        try:
            first = ex.submit(contextvars.copy_context().run, _call).result()
        finally:
            aux.exit_strict_aux_scope(token)
        second = ex.submit(contextvars.copy_context().run, _call).result()
    assert first == "refused"
    assert second == "reached", "the reused worker kept the previous task's strict context"


# ---------------------------------------------------------------------------
# the run-local marker belongs to strict agents only
# ---------------------------------------------------------------------------

def test_only_strict_agents_mark_their_runtime():
    from run_agent import AIAgent

    strict = AIAgent.__new__(AIAgent); strict.__dict__.update(_Agent(strict=True).__dict__)
    plain = AIAgent.__new__(AIAgent); plain.__dict__.update(_Agent(strict=False).__dict__)
    assert strict._current_main_runtime().get("strict_model_run") is True
    assert "strict_model_run" not in plain._current_main_runtime()
