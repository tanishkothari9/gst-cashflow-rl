"""
Tests for reward functions — verifying signals are economically sensible.

Run with: PYTHONPATH=. pytest tests/test_reward.py -v
"""

import pytest
from gst_cashflow_env.models import GSTAction, GSTObservation, Transaction
from server.reward import (
    compute_step_reward,
    compute_terminal_reward,
    compute_l1_reward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sale_txn(id: str, base: float, urgency: float = 0.5, status: str = "pending") -> Transaction:
    gst = round(base * 0.12, 2)
    t = Transaction(
        id=id, party_name=f"Retailer-{id}", transaction_type="sale",
        base_amount=base, gst_rate=0.12, gst_amount=gst,
        total_amount=round(base + gst, 2), hsn_code="6104",
        due_day=10, urgency_score=urgency, vendor_gstr1_reliability=0.0,
    )
    t.status = status
    return t


def _make_purchase_txn(
    id: str, base: float, reliability: float = 0.9, status: str = "pending"
) -> Transaction:
    gst = round(base * 0.05, 2)
    t = Transaction(
        id=id, party_name=f"Vendor-{id}", transaction_type="purchase",
        base_amount=base, gst_rate=0.05, gst_amount=gst,
        total_amount=round(base + gst, 2), hsn_code="6006",
        due_day=15, urgency_score=0.5, vendor_gstr1_reliability=reliability,
    )
    t.status = status
    return t


def _base_obs(**kwargs) -> GSTObservation:
    defaults = dict(
        day=5, days_to_filing=15, cash_balance=200_000.0,
        gst_collected_so_far=0.0, itc_secured_so_far=0.0,
        net_gst_if_filed_now=0.0, baseline_gst=60_000.0,
        pending_sales=[], pending_purchases=[],
        vendor_scores={"Vendor": 1.0}, retailer_scores={"Retailer": 1.0},
        episode_id="test-ep", difficulty_level="L1",
    )
    defaults.update(kwargs)
    return GSTObservation(**defaults)


# ---------------------------------------------------------------------------
# Test: step reward — ITC securing
# ---------------------------------------------------------------------------

def test_step_reward_pay_vendor_gives_positive_reward():
    """Paying a reliable vendor should give a positive step reward."""
    purchase = _make_purchase_txn("P-001", 300_000, reliability=0.95)
    obs = _base_obs(pending_purchases=[purchase])
    action = GSTAction(action_type="PAY_VENDOR", transaction_id="P-001")
    # itc_secured = gst_amount = 300000 * 0.05 = 15000
    ledger_update = {"itc_secured": 15_000.0}

    reward = compute_step_reward(action, obs, obs, ledger_update)
    # Expected ITC reward = (15000 * 0.95) / 1000 = 14.25
    # Plus time penalty = -0.5
    # Total ~ 13.75
    assert reward > 0.0, f"Expected positive reward but got {reward}"


def test_step_reward_pay_unreliable_vendor_lower_reward():
    """Paying an unreliable vendor gives less reward than a reliable one."""
    purchase_reliable = _make_purchase_txn("P-RELY", 300_000, reliability=0.95)
    purchase_unreliable = _make_purchase_txn("P-UNRELY", 300_000, reliability=0.20)

    obs_r = _base_obs(pending_purchases=[purchase_reliable])
    obs_u = _base_obs(pending_purchases=[purchase_unreliable])

    ledger_update = {"itc_secured": 15_000.0}

    action_r = GSTAction(action_type="PAY_VENDOR", transaction_id="P-RELY")
    action_u = GSTAction(action_type="PAY_VENDOR", transaction_id="P-UNRELY")

    reward_reliable = compute_step_reward(action_r, obs_r, obs_r, ledger_update)
    reward_unreliable = compute_step_reward(action_u, obs_u, obs_u, ledger_update)

    assert reward_reliable > reward_unreliable


# ---------------------------------------------------------------------------
# Test: step reward — sales fulfillment
# ---------------------------------------------------------------------------

def test_step_reward_fulfill_sale_positive():
    """Fulfilling a sale always gives positive step reward."""
    sale = _make_sale_txn("S-001", 100_000, urgency=0.5)
    obs = _base_obs(pending_sales=[sale])
    action = GSTAction(action_type="FULFILL_SALE", transaction_id="S-001")

    reward = compute_step_reward(action, obs, obs, {})
    # Base +2.0, urgency +1.5, time -0.5 → +3.0
    assert reward > 0.0


def test_step_reward_high_urgency_sale_bigger_reward():
    """High-urgency sales should give bigger reward than low-urgency."""
    sale_high = _make_sale_txn("S-HIGH", 100_000, urgency=1.0)
    sale_low = _make_sale_txn("S-LOW", 100_000, urgency=0.1)

    obs_h = _base_obs(pending_sales=[sale_high])
    obs_l = _base_obs(pending_sales=[sale_low])

    r_high = compute_step_reward(
        GSTAction(action_type="FULFILL_SALE", transaction_id="S-HIGH"), obs_h, obs_h, {}
    )
    r_low = compute_step_reward(
        GSTAction(action_type="FULFILL_SALE", transaction_id="S-LOW"), obs_l, obs_l, {}
    )

    assert r_high > r_low


# ---------------------------------------------------------------------------
# Test: step reward — cash floor violation
# ---------------------------------------------------------------------------

def test_step_reward_cash_floor_violation():
    """Dropping below ₹50,000 cash floor triggers -20 penalty."""
    obs_before = _base_obs(cash_balance=100_000.0)
    obs_after = _base_obs(cash_balance=30_000.0)  # Below floor

    action = GSTAction(action_type="DO_NOTHING")
    reward = compute_step_reward(action, obs_before, obs_after, {})

    # DO_NOTHING penalty (-2) + time (-0.5) + cash floor (-20) = -22.5
    assert reward <= -20.0


def test_step_reward_no_cash_violation_above_floor():
    """Cash above ₹50,000 should NOT trigger the floor penalty."""
    obs = _base_obs(cash_balance=75_000.0)
    action = GSTAction(action_type="DO_NOTHING")
    reward = compute_step_reward(action, obs, obs, {})

    # Only time penalty + do_nothing penalty = -2.5
    assert reward == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# Test: step reward — deadline pressure
# ---------------------------------------------------------------------------

def test_step_reward_deadline_pressure_near_filing():
    """With 3 or fewer days to filing, unpaid ITC triggers escalating penalty."""
    purchase = _make_purchase_txn("P-001", 100_000, reliability=0.9)  # GST=5000
    obs = _base_obs(days_to_filing=2, pending_purchases=[purchase])
    action = GSTAction(action_type="DO_NOTHING")

    reward = compute_step_reward(action, obs, obs, {})
    # Time -0.5, do_nothing -2.0, deadline (5000/1000)*5 = -25 → total ≈ -27.5
    assert reward <= -25.0


def test_step_reward_no_deadline_pressure_far_from_filing():
    """Far from the deadline, no deadline penalty applies."""
    purchase = _make_purchase_txn("P-001", 100_000, reliability=0.9)
    obs = _base_obs(days_to_filing=15, pending_purchases=[purchase])
    action = GSTAction(action_type="DO_NOTHING")

    reward = compute_step_reward(action, obs, obs, {})
    assert reward == pytest.approx(-2.5)  # Only time + do_nothing


# ---------------------------------------------------------------------------
# Test: terminal reward — GST saved
# ---------------------------------------------------------------------------

def test_terminal_reward_gst_saved():
    """More GST saved vs baseline → higher terminal reward."""
    obs = _base_obs(cash_balance=100_000.0, baseline_gst=60_000.0)
    result_good = {"net_payable": 10_000.0, "itc_utilization_pct": 0.9}
    result_bad = {"net_payable": 50_000.0, "itc_utilization_pct": 0.1}

    reward_good = compute_terminal_reward(obs, result_good, filing_day=18)
    reward_bad = compute_terminal_reward(obs, result_bad, filing_day=18)

    assert reward_good > reward_bad


def test_terminal_reward_early_filing_bonus():
    """Filing on Day 18 gives more reward than filing on Day 20."""
    obs = _base_obs(cash_balance=100_000.0, baseline_gst=60_000.0)
    result = {"net_payable": 20_000.0, "itc_utilization_pct": 0.7}

    reward_early = compute_terminal_reward(obs, result, filing_day=17)
    reward_late = compute_terminal_reward(obs, result, filing_day=20)

    assert reward_early > reward_late


def test_terminal_reward_late_filing_penalty():
    """Filing after Day 20 incurs a -100 penalty."""
    obs = _base_obs(cash_balance=100_000.0, baseline_gst=60_000.0)
    result = {"net_payable": 20_000.0, "itc_utilization_pct": 0.7}

    reward_on_time = compute_terminal_reward(obs, result, filing_day=20)
    reward_late = compute_terminal_reward(obs, result, filing_day=21)

    assert reward_on_time > reward_late
    # Late filing penalty dominates
    assert reward_late < reward_on_time - 100.0


def test_terminal_reward_cash_survival():
    """Healthy cash (≥₹50K) → +30, dead cash (≤0) → -200."""
    result = {"net_payable": 0.0, "itc_utilization_pct": 1.0}

    obs_healthy = _base_obs(cash_balance=100_000.0, baseline_gst=60_000.0)
    obs_dead = _base_obs(cash_balance=-1.0, baseline_gst=60_000.0)

    reward_healthy = compute_terminal_reward(obs_healthy, result, filing_day=18)
    reward_dead = compute_terminal_reward(obs_dead, result, filing_day=18)

    # Healthy cash adds +30, dead cash subtracts -200 → 230-point swing
    assert reward_healthy > reward_dead
    assert reward_healthy - reward_dead == pytest.approx(230.0)


def test_terminal_reward_itc_utilization_scales():
    """Higher ITC utilization gives proportionally higher reward."""
    obs = _base_obs(cash_balance=100_000.0, baseline_gst=60_000.0)
    result_full = {"net_payable": 0.0, "itc_utilization_pct": 1.0}
    result_half = {"net_payable": 30_000.0, "itc_utilization_pct": 0.5}
    result_zero = {"net_payable": 60_000.0, "itc_utilization_pct": 0.0}

    r_full = compute_terminal_reward(obs, result_full, filing_day=19)
    r_half = compute_terminal_reward(obs, result_half, filing_day=19)
    r_zero = compute_terminal_reward(obs, result_zero, filing_day=19)

    assert r_full > r_half > r_zero


def test_terminal_reward_stakeholder_health():
    """Better relationship scores give higher terminal reward."""
    result = {"net_payable": 20_000.0, "itc_utilization_pct": 0.7}

    obs_good = _base_obs(
        cash_balance=100_000.0, baseline_gst=60_000.0,
        vendor_scores={"V": 1.0}, retailer_scores={"R": 1.0},
    )
    obs_bad = _base_obs(
        cash_balance=100_000.0, baseline_gst=60_000.0,
        vendor_scores={"V": 0.2}, retailer_scores={"R": 0.2},
    )

    reward_good = compute_terminal_reward(obs_good, result, filing_day=19)
    reward_bad = compute_terminal_reward(obs_bad, result, filing_day=19)

    assert reward_good > reward_bad


# ---------------------------------------------------------------------------
# Test: L1 simplified reward
# ---------------------------------------------------------------------------

def test_l1_reward_on_time_positive():
    reward = compute_l1_reward(gst_saved=10_000.0, filed_on_time=True)
    assert reward > 0.0


def test_l1_reward_late_filing_negative():
    reward = compute_l1_reward(gst_saved=0.0, filed_on_time=False)
    assert reward < 0.0


def test_l1_reward_more_gst_saved_is_better():
    r_high = compute_l1_reward(gst_saved=50_000.0, filed_on_time=True)
    r_low = compute_l1_reward(gst_saved=5_000.0, filed_on_time=True)
    assert r_high > r_low
