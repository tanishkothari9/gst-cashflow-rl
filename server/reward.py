"""
Reward functions for the GST Cash Flow Optimization environment.

All functions are PURE: no state mutation, no randomness, no side effects.
Every component has an inline comment explaining the economic rationale.
These can be tested exhaustively without needing to run the full environment.

Design philosophy:
  - Dense per-step rewards give signal during early training when episodes are short.
  - Terminal reward at filing time reflects the true economic outcome.
  - Every component is scaled so that a well-behaved agent achieves +100–+300 per episode.
  - Anti-hacking: baseline_gst comparison prevents satisficing on suboptimal ITC.
"""

from __future__ import annotations

from typing import Optional

from gst_cashflow_env.models import GSTAction, GSTObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_transaction_urgency(transaction_id: str, obs: GSTObservation) -> float:
    """Look up the urgency_score for a given transaction from the observation."""
    all_txns = obs.pending_sales + obs.pending_purchases
    for t in all_txns:
        if t.id == transaction_id:
            return t.urgency_score
    return 0.5  # Fallback: moderate urgency if transaction not found


def _get_vendor_reliability(transaction_id: str, obs: GSTObservation) -> float:
    """Look up vendor_gstr1_reliability for a given purchase transaction."""
    for t in obs.pending_purchases:
        if t.id == transaction_id:
            return t.vendor_gstr1_reliability
    return 0.5  # Fallback


def _get_purchase_gst(transaction_id: str, obs: GSTObservation) -> float:
    """Return the gst_amount for a purchase transaction in the observation."""
    for t in obs.pending_purchases:
        if t.id == transaction_id:
            return t.gst_amount
    return 0.0


# ---------------------------------------------------------------------------
# Per-step reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    action: GSTAction,
    prev_obs: GSTObservation,
    new_obs: GSTObservation,
    ledger_update: dict,
) -> float:
    """
    Dense per-step reward. Called after every action, before terminal reward.

    Shapes the learning signal so the agent receives gradient on every step,
    not just at filing time. All components are economically grounded.
    """
    reward = 0.0

    # --- TIME PRESSURE ---
    # Every step costs a small amount — incentivizes the agent to be decisive
    # and not idle. With 30 steps max, this caps at -15 which is recoverable.
    reward -= 0.5

    # --- ITC SECURED ---
    # Reward paying vendors: each ₹1,000 of expected ITC secured = +1 reward.
    # Expected ITC = gst_amount × vendor_reliability (not guaranteed until filing).
    # We reward expected ITC, not actual, because actual is stochastic —
    # the agent should learn to prefer high-reliability vendors.
    if action.action_type == "PAY_VENDOR" and action.transaction_id:
        itc_secured = ledger_update.get("itc_secured", 0.0)
        reliability = _get_vendor_reliability(action.transaction_id, prev_obs)
        expected_itc = itc_secured * reliability
        reward += expected_itc / 1_000.0

    # --- SALES FULFILLMENT ---
    # Flat reward for each sale fulfilled. Cash generation is necessary for
    # subsequent vendor payments. High-urgency sales get a bonus because
    # serving urgent retailers preserves the relationship.
    if action.action_type == "FULFILL_SALE":
        reward += 2.0
        if action.transaction_id:
            urgency = _get_transaction_urgency(action.transaction_id, prev_obs)
            reward += urgency * 3.0  # Urgency bonus: up to +3 for urgency=1.0

    # --- DEFERRAL PENALTIES ---
    # Penalize deferring transactions — damages relationships and delays ITC.
    # High-urgency deferrals are heavily penalized; low-urgency are tolerable
    # when cash constraints demand it. Maximum penalty: -8.0 × 1.0 = -8.0
    if action.action_type in ("DEFER_SALE", "DEFER_VENDOR") and action.transaction_id:
        urgency = _get_transaction_urgency(action.transaction_id, prev_obs)
        reward -= urgency * 8.0

    # --- CASH FLOOR VIOLATION ---
    # Hard penalty for dropping below operating floor (₹50,000).
    # Below this level the business cannot pay salaries — existential risk.
    # This should be the biggest per-step penalty to make the constraint binding.
    if new_obs.cash_balance < 50_000:
        reward -= 20.0

    # --- DEADLINE PRESSURE ---
    # Escalating penalty for unpaid vendor ITC close to the filing deadline.
    # With 3 days or fewer left, every ₹1,000 of unpaid ITC is increasingly
    # at risk of being lost for this month. Forces urgency near Day 17–18.
    if new_obs.days_to_filing <= 3:
        unpaid_itc = sum(
            t.gst_amount
            for t in new_obs.pending_purchases
            if t.status == "pending"
        )
        reward -= (unpaid_itc / 1_000.0) * 5.0

    # --- DO NOTHING PENALTY ---
    # Waiting is sometimes the right call (insufficient cash), but should not
    # be the agent's default strategy. This penalty prevents a lazy policy
    # that does nothing until the deadline.
    if action.action_type == "DO_NOTHING":
        reward -= 2.0

    return reward


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------

def compute_terminal_reward(
    final_obs: GSTObservation,
    gstr3b_result: dict,
    filing_day: int,
) -> float:
    """
    Sparse terminal reward. Called once when FILE_GSTR3B is executed (or forced at Day 30).

    This is the most important reward signal — it reflects the actual economic
    outcome of all decisions made during the episode. The per-step rewards
    guide exploration; the terminal reward sets the true objective.
    """
    reward = 0.0

    # --- CORE OBJECTIVE: GST SAVED vs NAIVE BASELINE ---
    # Compare agent's net GST vs what a naive greedy agent would have paid.
    # ₹500 saved = +1 reward point.
    # For typical episodes (₹30,000–₹60,000 baseline), a perfect agent
    # scores +60 to +120 just from this component.
    gst_saved = final_obs.baseline_gst - gstr3b_result["net_payable"]
    reward += gst_saved / 500.0

    # --- ITC UTILIZATION RATE ---
    # Reward the percentage of secured ITC that was actually claimed.
    # Perfect utilization (1.0) gives +100. This directly incentivizes
    # paying high-reliability vendors before the filing deadline.
    itc_utilization = gstr3b_result.get("itc_utilization_pct", 0.0)
    reward += itc_utilization * 100.0

    # --- FILING PUNCTUALITY ---
    # Bonus for filing early (comfortable margin) vs on-time vs late.
    # Filing 2+ days early demonstrates the agent learned not to cut it close.
    if filing_day <= 18:
        reward += 70.0   # Early filing — good margin, maximum bonus
    elif filing_day <= 20:
        reward += 50.0   # On time — acceptable
    else:
        reward -= 100.0  # Late — hard penalty (late fees + compliance risk)

    # --- STAKEHOLDER HEALTH ---
    # Reward maintaining good relationships with vendors and retailers.
    # Relationship scores degrade each time a transaction is deferred.
    # Perfect scores (1.0 each) contribute +30 (15 per pool).
    avg_vendor = (
        sum(final_obs.vendor_scores.values()) / max(1, len(final_obs.vendor_scores))
    )
    avg_retailer = (
        sum(final_obs.retailer_scores.values()) / max(1, len(final_obs.retailer_scores))
    )
    reward += (avg_vendor + avg_retailer) * 15.0

    # --- CASH SURVIVAL ---
    # Cash survival is a hard business constraint.
    # Healthy cash → +30 (business is stable, can operate next month).
    # Tight but alive → +10 (survived but fragile).
    # Dead (negative cash) → -200 (catastrophic; dominates all other rewards).
    if final_obs.cash_balance >= 50_000:
        reward += 30.0
    elif final_obs.cash_balance > 0:
        reward += 10.0
    else:
        reward -= 200.0  # Business cannot operate

    return reward


# ---------------------------------------------------------------------------
# L1-specific simplified reward (for early curriculum training)
# ---------------------------------------------------------------------------

def compute_l1_reward(
    gst_saved: float,
    filed_on_time: bool,
) -> float:
    """
    Simplified 2-component reward for L1 (early curriculum).

    Uses only GST savings and on-time filing to avoid conflicting signals
    when the model has no prior knowledge of the domain.
    This guarantees a clear, learnable signal for the first training stage.
    """
    return gst_saved * 0.01 + (1.0 if filed_on_time else -1.0)
