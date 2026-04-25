"""
FastAPI application for the GST Cash Flow Optimization Environment.

Endpoints:
    GET  /health — Liveness probe (used by Docker HEALTHCHECK and HF Spaces)
    POST /reset  — Reset the environment and return initial observation
    POST /step   — Execute an action and return next observation
    GET  /state  — Get current episode state
    WS   /ws     — WebSocket endpoint for persistent concurrent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server import create_app

from gst_cashflow_env.models import GSTAction, GSTObservation
from server.gst_environment import GSTEnvironment


app = create_app(
    GSTEnvironment,
    GSTAction,
    GSTObservation,
    env_name="gst-cashflow-env",
    max_concurrent_envs=64,
)


# Health endpoint — required by both Docker HEALTHCHECK and HF Spaces routing.
# create_app does not add one, so we add it here on the returned FastAPI instance.
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "env": "gst-cashflow-env"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
