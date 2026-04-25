"""
FastAPI application for the GST Cash Flow Optimization Environment.

Endpoints:
    POST /reset  — Reset the environment and return initial observation
    POST /step   — Execute an action and return next observation
    GET  /state  — Get current episode state
    GET  /schema — Get action/observation schemas
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


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
