"""
FastAPI application for the GST Cash Flow Optimization Environment.

Endpoints:
    GET  /health — Liveness probe (used by Docker HEALTHCHECK and HF Spaces)
    POST /reset  — Reset the environment and return initial observation
    POST /step   — Execute an action and return next observation
    GET  /state  — Get current episode state
    WS   /ws     — WebSocket endpoint for persistent concurrent sessions

The /reset and /step routes override the OpenEnv defaults because the
OpenEnv HTTP server is stateless (creates a fresh env per request). We
maintain a single persistent GSTEnvironment instance per session keyed
by episode_id so that state is preserved between reset and step calls.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import Body
from openenv.core.env_server import create_app
from pydantic import BaseModel

from gst_cashflow_env.models import GSTAction, GSTObservation
from server.gst_environment import GSTEnvironment


app = create_app(
    GSTEnvironment,
    GSTAction,
    GSTObservation,
    env_name="gst-cashflow-env",
    max_concurrent_envs=64,
)


# ---------------------------------------------------------------------------
# Session store — maps episode_id → GSTEnvironment instance
# The OpenEnv HTTP server is stateless (new env per request); we fix this
# by keeping live env instances here and routing by episode_id.
# ---------------------------------------------------------------------------

_sessions: Dict[str, GSTEnvironment] = {}
_sessions_lock = asyncio.Lock()
_MAX_SESSIONS = 64


def _evict_oldest() -> None:
    """Remove the oldest session when the store is full."""
    if _sessions:
        oldest = next(iter(_sessions))
        del _sessions[oldest]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[str] = "L1"
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]
    episode_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Overridden /reset and /step — stateful, session-aware
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict:
    """Reset the environment and return the initial observation."""
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
    """Execute an action and return the next observation."""
    action_data = request.action

    # Resolve which env session to use
    episode_id = request.episode_id
    env: Optional[GSTEnvironment] = None

    async with _sessions_lock:
        if episode_id and episode_id in _sessions:
            env = _sessions[episode_id]
        elif _sessions:
            # Fall back to most-recently created session
            last_key = next(reversed(_sessions))
            env = _sessions[last_key]
        else:
            return {"error": "No active session. Call /reset first.", "done": True}

    # Parse action
    try:
        action = GSTAction.model_validate(action_data)
    except Exception as exc:
        return {"error": f"Invalid action: {exc}", "done": False}

    obs = env.step(action)

    # Clean up terminated sessions
    if obs.done and episode_id and episode_id in _sessions:
        async with _sessions_lock:
            _sessions.pop(episode_id, None)

    return {
        "observation": obs.model_dump(exclude={"reward", "done", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
    }


# ---------------------------------------------------------------------------
# Health endpoint — required by Docker HEALTHCHECK and HF Spaces
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "env": "gst-cashflow-env", "active_sessions": len(_sessions)}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
