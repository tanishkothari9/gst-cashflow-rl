"""
FastAPI application for the GST Cash Flow Optimization Environment.

Endpoints:
    GET  /health — Liveness probe
    POST /reset  — Reset environment, returns initial observation
    POST /step   — Execute action, returns next observation
    GET  /state  — Current episode state
    WS   /ws     — WebSocket for persistent concurrent sessions

The OpenEnv HTTP server is stateless (fresh env per request), so /step
would always get ledger=None. We remove OpenEnv's /reset and /step routes
and replace them with stateful session-aware handlers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import Body
from openenv.core.env_server import create_app
from pydantic import BaseModel

from gst_cashflow_env.models import GSTAction, GSTObservation
from server.gst_environment import GSTEnvironment


# Build the base app (provides /ws, /state, /schema, /metadata)
app = create_app(
    GSTEnvironment,
    GSTAction,
    GSTObservation,
    env_name="gst-cashflow-env",
    max_concurrent_envs=64,
)

# Remove OpenEnv's stateless /reset and /step so ours take priority
_OVERRIDE = {"/reset", "/step", "/health"}
app.routes[:] = [
    r for r in app.routes
    if not (hasattr(r, "path") and r.path in _OVERRIDE)
]

# ---------------------------------------------------------------------------
# Session store — episode_id → live GSTEnvironment instance
# ---------------------------------------------------------------------------

_sessions: Dict[str, GSTEnvironment] = {}
_sessions_lock = asyncio.Lock()
_MAX_SESSIONS = 64


def _evict_oldest() -> None:
    if _sessions:
        del _sessions[next(iter(_sessions))]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[str] = "L1"
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]
    episode_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "env": "gst-cashflow-env", "active_sessions": len(_sessions)}


@app.post("/reset")
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict:
    async with _sessions_lock:
        if len(_sessions) >= _MAX_SESSIONS:
            _evict_oldest()
        env = GSTEnvironment(difficulty=request.difficulty or "L1")
        obs = env.reset(seed=request.seed, episode_id=request.episode_id)
        _sessions[obs.episode_id] = env

    return {
        "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
async def step(request: StepRequest = Body(...)) -> dict:
    episode_id = request.episode_id

    async with _sessions_lock:
        if episode_id and episode_id in _sessions:
            env = _sessions[episode_id]
        elif _sessions:
            last_key = next(reversed(_sessions))
            env = _sessions[last_key]
            episode_id = last_key
        else:
            return {"error": "No active session — call /reset first.", "done": True}

    try:
        action = GSTAction.model_validate(request.action)
    except Exception as exc:
        return {"error": f"Invalid action: {exc}", "done": False}

    obs = env.step(action)

    if obs.done:
        async with _sessions_lock:
            _sessions.pop(episode_id, None)

    return {
        "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
