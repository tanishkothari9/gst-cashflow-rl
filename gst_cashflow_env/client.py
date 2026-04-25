"""GST Cash Flow Optimization Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from gst_cashflow_env.models import GSTAction, GSTObservation, Transaction


class GSTEnvClient(EnvClient[GSTAction, GSTObservation, State]):
    """
    WebSocket client for the GST Cash Flow Optimization Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions for LLM agent training.

    Example (async):
        >>> async with GSTEnvClient(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(seed=42)
        ...     while not result.done:
        ...         action = GSTAction(action_type="DO_NOTHING")
        ...         result = await client.step(action)

    Example (sync wrapper):
        >>> env = GSTEnvClient(base_url="http://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset(seed=42)
        ...     result = env.step(GSTAction(action_type="FILE_GSTR3B"))
    """

    def _step_payload(self, action: GSTAction) -> Dict[str, Any]:
        """Convert GSTAction to JSON payload for the step WebSocket message."""
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.transaction_id is not None:
            payload["transaction_id"] = action.transaction_id
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GSTObservation]:
        """Parse server WebSocket response into StepResult[GSTObservation]."""
        obs_data: Dict[str, Any] = payload.get("observation", {})

        def _parse_transactions(raw: List[Dict]) -> List[Transaction]:
            return [Transaction(**t) for t in raw]

        observation = GSTObservation(
            day=obs_data.get("day", 1),
            days_to_filing=obs_data.get("days_to_filing", 20),
            cash_balance=obs_data.get("cash_balance", 0.0),
            gst_collected_so_far=obs_data.get("gst_collected_so_far", 0.0),
            itc_secured_so_far=obs_data.get("itc_secured_so_far", 0.0),
            net_gst_if_filed_now=obs_data.get("net_gst_if_filed_now", 0.0),
            baseline_gst=obs_data.get("baseline_gst", 0.0),
            pending_sales=_parse_transactions(obs_data.get("pending_sales", [])),
            pending_purchases=_parse_transactions(obs_data.get("pending_purchases", [])),
            vendor_scores=obs_data.get("vendor_scores", {}),
            retailer_scores=obs_data.get("retailer_scores", {}),
            episode_id=obs_data.get("episode_id", ""),
            difficulty_level=obs_data.get("difficulty_level", "L1"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
