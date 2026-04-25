"""
GST Cash Flow Optimization Environment.

Implements the OpenEnv Environment interface:
  - reset(seed, episode_id, **kwargs) → GSTObservation
  - step(action, **kwargs) → GSTObservation  (done/reward embedded in observation)
  - state (property) → State

One episode = one 30-day GSTR-3B filing cycle for an Indian apparel SME.

STATE TRANSITIONS:
  FULFILL_SALE  → cash += sale_total, gst_collected += sale_gst
  PAY_VENDOR    → cash -= vendor_total, itc_secured += vendor_gst (if cash sufficient)
  DEFER_SALE    → transaction stays pending, retailer score decays, urgency increases
  DEFER_VENDOR  → transaction stays pending, vendor score decays, urgency increases
  FILE_GSTR3B   → compute_gstr3b() rolled, terminal reward, done=True
  DO_NOTHING    → time passes, daily burn applied

ANTI-HACKING RULES:
  1. FILE_GSTR3B not allowed before Day 8 (prevents instant-file exploit)
  2. PAY_VENDOR fails gracefully if cash < invoice total (enforces cash constraint)
  3. Urgency decays upward each day (forces eventual action on deferred items)
  4. Relationship scores degrade on each deferral (relational cost of procrastination)
  5. baseline_gst is fixed at episode start (cannot be gamed by state manipulation)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from gst_cashflow_env.models import GSTAction, GSTObservation, Transaction
from server.ledger import GSTLedger
from server.scenario_generator import ERPNextScenarioGenerator
import server.reward as reward_module


class GSTEnvironment(Environment):
    """
    GST Cash Flow Optimization RL Environment.

    Supports concurrent WebSocket sessions — each client gets its own
    isolated environment instance via the factory pattern in create_app.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Urgency increases by this amount per day for all pending transactions
    URGENCY_DECAY_RATE: float = 0.05
    # Relationship score drop per deferral (scales with urgency)
    SCORE_DECAY_FACTOR: float = 0.10

    def __init__(self, difficulty: str = "L1") -> None:
        super().__init__()
        self.difficulty = difficulty
        self.scenario_generator = ERPNextScenarioGenerator()

        # Episode state — initialized in reset()
        self.ledger: Optional[GSTLedger] = None
        self._baseline_gst: float = 0.0
        self._episode_id: str = ""
        self._day: int = 1
        self._episode_days: int = 30
        self._vendor_scores: Dict[str, float] = {}
        self._retailer_scores: Dict[str, float] = {}
        self._filing_day: Optional[int] = None
        self._gstr3b_result: Optional[Dict] = None

    # ------------------------------------------------------------------
    # OpenEnv API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GSTObservation:
        """
        Start a new episode.

        Generates a fresh scenario, initializes the ledger, and returns
        the first observation. All state from the previous episode is discarded.
        """
        difficulty = kwargs.get("difficulty", self.difficulty)
        scenario = self.scenario_generator.generate_episode(difficulty, seed=seed)

        self._episode_id = episode_id or scenario["episode_id"]
        self._baseline_gst = scenario["baseline_gst"]
        self._episode_days = scenario["episode_days"]
        self._day = 1
        self._filing_day = None
        self._gstr3b_result = None

        self.ledger = GSTLedger(
            opening_cash=scenario["opening_cash"],
            transactions=scenario["transactions"],
        )

        # All relationships start at full health (1.0)
        self._vendor_scores = {
            t.party_name: 1.0
            for t in scenario["transactions"]
            if t.transaction_type == "purchase"
        }
        self._retailer_scores = {
            t.party_name: 1.0
            for t in scenario["transactions"]
            if t.transaction_type == "sale"
        }

        return self._build_observation()

    # ------------------------------------------------------------------
    # OpenEnv API: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: GSTAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GSTObservation:
        """
        Execute one action and advance the simulation by one day.

        The agent submits one action per step. After executing the action,
        daily burn is deducted and the day counter advances. The environment
        terminates when FILE_GSTR3B is called (or forced at episode_days).

        Returns GSTObservation with done=True and reward set at terminal steps.
        """
        assert self.ledger is not None, "Call reset() before step()"

        prev_obs = self._build_observation()
        ledger_update: Dict = {}

        # --- ANTI-HACKING: minimum episode length before filing allowed ---
        if action.action_type == "FILE_GSTR3B" and self._day < 8:
            # Override to DO_NOTHING — episode is too new to file
            action = GSTAction(action_type="DO_NOTHING")

        # --- EXECUTE ACTION ---
        if action.action_type == "FULFILL_SALE":
            ledger_update = self.ledger.fulfill_sale(action.transaction_id, self._day)
            if "error" not in ledger_update:
                # Fulfilling a sale on time improves the retailer relationship
                party = self._get_party_name(action.transaction_id)
                if party and party in self._retailer_scores:
                    self._retailer_scores[party] = min(
                        1.0, self._retailer_scores[party] + 0.05
                    )

        elif action.action_type == "PAY_VENDOR":
            ledger_update = self.ledger.pay_vendor(action.transaction_id, self._day)
            if "error" in ledger_update:
                # Insufficient cash — silently convert to DO_NOTHING
                ledger_update = {}
                action = GSTAction(action_type="DO_NOTHING")

        elif action.action_type == "DEFER_SALE":
            self._apply_deferral(action, is_sale=True)

        elif action.action_type == "DEFER_VENDOR":
            self._apply_deferral(action, is_sale=False)

        elif action.action_type == "FILE_GSTR3B":
            return self._handle_filing(action, prev_obs, ledger_update)

        # --- ADVANCE TIME ---
        self.ledger.apply_daily_burn()
        self._apply_urgency_decay()
        self._day += 1

        new_obs = self._build_observation()
        step_reward = reward_module.compute_step_reward(action, prev_obs, new_obs, ledger_update)

        # --- FORCED TERMINAL: reached episode_days without filing ---
        done = self._day > self._episode_days
        if done:
            gstr3b_result = self.ledger.compute_gstr3b(self._episode_days)
            self._gstr3b_result = gstr3b_result
            self._filing_day = self._episode_days
            terminal_reward = reward_module.compute_terminal_reward(
                new_obs, gstr3b_result, filing_day=self._episode_days
            )
            step_reward += terminal_reward
            new_obs = self._build_observation()
            new_obs.done = True
            new_obs.reward = step_reward
            new_obs.metadata = {
                "gstr3b": gstr3b_result,
                "filing_day": self._episode_days,
                "episode_id": self._episode_id,
                "forced_terminal": True,
            }
            return new_obs

        new_obs.reward = step_reward
        return new_obs

    # ------------------------------------------------------------------
    # OpenEnv API: state (property)
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return current episode metadata."""
        return State(
            episode_id=self._episode_id,
            step_count=self._day,
            # State allows extra fields via extra="allow"
            difficulty=self.difficulty,
            cash=self.ledger.cash if self.ledger else 0.0,
            day=self._day,
            filing_day=self._filing_day,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_filing(
        self,
        action: GSTAction,
        prev_obs: GSTObservation,
        ledger_update: Dict,
    ) -> GSTObservation:
        """Process FILE_GSTR3B: roll GSTR-1 dice, compute terminal reward, done=True."""
        self._filing_day = self._day
        gstr3b_result = self.ledger.compute_gstr3b(self._day)
        self._gstr3b_result = gstr3b_result

        new_obs = self._build_observation()

        step_reward = reward_module.compute_step_reward(action, prev_obs, new_obs, ledger_update)
        terminal_reward = reward_module.compute_terminal_reward(
            new_obs, gstr3b_result, filing_day=self._day
        )
        total_reward = step_reward + terminal_reward

        new_obs.done = True
        new_obs.reward = total_reward
        new_obs.metadata = {
            "gstr3b": gstr3b_result,
            "filing_day": self._day,
            "episode_id": self._episode_id,
            "forced_terminal": False,
        }
        return new_obs

    def _apply_deferral(self, action: GSTAction, is_sale: bool) -> None:
        """
        Degrade the relationship score when a transaction is deferred.

        The degradation scales with urgency: deferring a high-urgency retailer
        damages the relationship significantly; deferring a low-urgency vendor is fine.
        """
        if not action.transaction_id:
            return

        txn = self.ledger.get_transaction(action.transaction_id)
        if txn is None:
            return

        # Urgency-weighted score degradation
        degradation = txn.urgency_score * self.SCORE_DECAY_FACTOR

        if is_sale and txn.party_name in self._retailer_scores:
            self._retailer_scores[txn.party_name] = max(
                0.0, self._retailer_scores[txn.party_name] - degradation
            )
        elif not is_sale and txn.party_name in self._vendor_scores:
            self._vendor_scores[txn.party_name] = max(
                0.0, self._vendor_scores[txn.party_name] - degradation
            )

        # Mark transaction as deferred in the ledger
        self.ledger.defer_transaction(action.transaction_id)

    def _apply_urgency_decay(self) -> None:
        """
        Each passing day increases the urgency of all pending transactions.

        This prevents the agent from indefinitely deferring transactions —
        urgency will eventually reach 1.0 and the deferral penalty becomes maximal.
        Urgency caps at 1.0.
        """
        for txn in self.ledger.transactions.values():
            if txn.status == "pending":
                txn.urgency_score = min(1.0, txn.urgency_score + self.URGENCY_DECAY_RATE)
            elif txn.status == "deferred":
                # Deferred transactions: restore to pending with increased urgency
                txn.urgency_score = min(1.0, txn.urgency_score + self.URGENCY_DECAY_RATE)
                txn.status = "pending"  # Restore so agent can retry

    def _get_party_name(self, transaction_id: Optional[str]) -> Optional[str]:
        """Get the party_name for a transaction ID from the ledger."""
        if not transaction_id:
            return None
        txn = self.ledger.get_transaction(transaction_id)
        return txn.party_name if txn else None

    def _build_observation(self) -> GSTObservation:
        """Construct the current GSTObservation from ledger state."""
        assert self.ledger is not None

        pending_sales = self.ledger.get_pending_sales()
        pending_purchases = self.ledger.get_pending_purchases()

        days_to_filing = max(0, 20 - self._day)  # Days until the 20th

        return GSTObservation(
            # Time
            day=self._day,
            days_to_filing=days_to_filing,
            # Financials
            cash_balance=self.ledger.cash,
            gst_collected_so_far=self.ledger.gst_collected,
            itc_secured_so_far=self.ledger.itc_secured,
            net_gst_if_filed_now=max(0.0, self.ledger.gst_collected - self.ledger.itc_secured),
            baseline_gst=self._baseline_gst,
            # Pending transactions
            pending_sales=pending_sales,
            pending_purchases=pending_purchases,
            # Relationship scores
            vendor_scores=dict(self._vendor_scores),
            retailer_scores=dict(self._retailer_scores),
            # Episode metadata
            episode_id=self._episode_id,
            difficulty_level=self.difficulty,
            # Openenv base fields (defaults)
            done=False,
            reward=None,
        )
