"""
Tests for GSTEnvironment — verifying the OpenEnv API contract.

Run with: PYTHONPATH=. pytest tests/test_environment.py -v
"""

import pytest
from gst_cashflow_env.models import GSTAction, GSTObservation
from openenv.core.env_server.types import State
from server.gst_environment import GSTEnvironment


@pytest.fixture
def env_l1():
    """L1 environment: cash-rich, few transactions, reliable vendors."""
    return GSTEnvironment(difficulty="L1")


@pytest.fixture
def env_l4():
    """L4 environment: cash-crisis, full complexity."""
    return GSTEnvironment(difficulty="L4")


# ---------------------------------------------------------------------------
# Test: reset() returns correct observation
# ---------------------------------------------------------------------------

def test_reset_returns_gst_observation(env_l1):
    obs = env_l1.reset(seed=42)
    assert isinstance(obs, GSTObservation)


def test_reset_starts_on_day_1(env_l1):
    obs = env_l1.reset(seed=42)
    assert obs.day == 1


def test_reset_observation_has_correct_difficulty(env_l1):
    obs = env_l1.reset(seed=42)
    assert obs.difficulty_level == "L1"


def test_reset_has_pending_transactions(env_l1):
    obs = env_l1.reset(seed=42)
    total = len(obs.pending_sales) + len(obs.pending_purchases)
    assert total > 0


def test_reset_cash_balance_matches_config(env_l1):
    """Opening cash must match L1 config (₹8L)."""
    obs = env_l1.reset(seed=42)
    assert obs.cash_balance == pytest.approx(800_000.0)


def test_reset_clears_previous_episode(env_l1):
    """After two resets, state should be fresh (not carry over)."""
    obs1 = env_l1.reset(seed=1)
    # Take a step
    action = GSTAction(action_type="DO_NOTHING")
    env_l1.step(action)
    # Reset again
    obs2 = env_l1.reset(seed=1)
    assert obs2.day == 1
    assert obs2.cash_balance == obs1.cash_balance


def test_reset_gst_collected_starts_at_zero(env_l1):
    obs = env_l1.reset(seed=42)
    assert obs.gst_collected_so_far == 0.0


def test_reset_done_is_false(env_l1):
    obs = env_l1.reset(seed=42)
    assert obs.done is False


# ---------------------------------------------------------------------------
# Test: step() returns correct observation
# ---------------------------------------------------------------------------

def test_step_do_nothing_advances_day(env_l1):
    env_l1.reset(seed=42)
    obs = env_l1.step(GSTAction(action_type="DO_NOTHING"))
    assert obs.day == 2


def test_step_returns_gst_observation(env_l1):
    env_l1.reset(seed=42)
    obs = env_l1.step(GSTAction(action_type="DO_NOTHING"))
    assert isinstance(obs, GSTObservation)


def test_step_reward_is_set(env_l1):
    env_l1.reset(seed=42)
    obs = env_l1.step(GSTAction(action_type="DO_NOTHING"))
    assert obs.reward is not None
    assert isinstance(obs.reward, float)


def test_step_fulfill_sale_increases_cash(env_l1):
    obs_before = env_l1.reset(seed=42)
    sale_id = obs_before.pending_sales[0].id
    expected_income = obs_before.pending_sales[0].total_amount

    obs_after = env_l1.step(GSTAction(action_type="FULFILL_SALE", transaction_id=sale_id))

    assert obs_after.cash_balance > obs_before.cash_balance
    # Cash increased by total_amount minus daily burn
    daily_burn = 2_000.0
    expected_cash = obs_before.cash_balance + expected_income - daily_burn
    assert obs_after.cash_balance == pytest.approx(expected_cash, abs=1.0)


def test_step_fulfill_sale_collects_gst(env_l1):
    obs_before = env_l1.reset(seed=42)
    sale_id = obs_before.pending_sales[0].id
    expected_gst = obs_before.pending_sales[0].gst_amount

    obs_after = env_l1.step(GSTAction(action_type="FULFILL_SALE", transaction_id=sale_id))
    assert obs_after.gst_collected_so_far == pytest.approx(expected_gst)


def test_step_pay_vendor_secures_itc(env_l1):
    obs_before = env_l1.reset(seed=42)
    purchase_id = obs_before.pending_purchases[0].id
    expected_itc = obs_before.pending_purchases[0].gst_amount

    obs_after = env_l1.step(GSTAction(action_type="PAY_VENDOR", transaction_id=purchase_id))
    assert obs_after.itc_secured_so_far == pytest.approx(expected_itc)


def test_step_pay_vendor_insufficient_cash_treated_as_do_nothing(env_l4):
    """L4 starts with ₹50K — paying a big vendor should fail gracefully."""
    obs_before = env_l4.reset(seed=42)
    # Find a purchase that costs more than opening cash
    big_purchase = None
    for p in obs_before.pending_purchases:
        if p.total_amount > obs_before.cash_balance:
            big_purchase = p
            break
    if big_purchase is None:
        pytest.skip("No purchase exceeds opening cash in this scenario")

    obs_after = env_l4.step(GSTAction(action_type="PAY_VENDOR", transaction_id=big_purchase.id))
    # Cash should only decrease by daily burn (do-nothing behavior)
    assert obs_after.itc_secured_so_far == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: anti-hacking — early filing prevented
# ---------------------------------------------------------------------------

def test_early_filing_overridden_before_day_8(env_l1):
    """FILE_GSTR3B before Day 8 must be overridden to DO_NOTHING."""
    env_l1.reset(seed=42)
    # Day 1 — try to file immediately
    obs = env_l1.step(GSTAction(action_type="FILE_GSTR3B"))
    # Should not be done (filing was blocked)
    assert obs.done is False
    assert obs.day == 2  # Day advanced


def test_filing_allowed_on_day_8(env_l1):
    """FILE_GSTR3B must succeed on or after Day 8."""
    env_l1.reset(seed=42)
    # Advance to Day 8
    for _ in range(7):
        env_l1.step(GSTAction(action_type="DO_NOTHING"))

    obs = env_l1.step(GSTAction(action_type="FILE_GSTR3B"))
    assert obs.done is True


# ---------------------------------------------------------------------------
# Test: episode termination
# ---------------------------------------------------------------------------

def test_episode_terminates_on_file_gstr3b(env_l1):
    """Filing GSTR-3B must set done=True and include reward."""
    env_l1.reset(seed=42)
    for _ in range(7):
        env_l1.step(GSTAction(action_type="DO_NOTHING"))
    obs = env_l1.step(GSTAction(action_type="FILE_GSTR3B"))

    assert obs.done is True
    assert obs.reward is not None
    assert "gstr3b" in obs.metadata


def test_episode_terminates_at_episode_days(env_l1):
    """Episode must terminate when episode_days is reached without filing."""
    env_l1.reset(seed=42)
    # L1 has 10 days — run all steps
    obs = None
    for _ in range(20):  # More than episode_days
        if obs is not None and obs.done:
            break
        obs = env_l1.step(GSTAction(action_type="DO_NOTHING"))

    assert obs.done is True


def test_terminal_observation_has_gstr3b_result(env_l1):
    """Terminal step must include GSTR-3B result in metadata."""
    env_l1.reset(seed=42)
    for _ in range(7):
        env_l1.step(GSTAction(action_type="DO_NOTHING"))
    obs = env_l1.step(GSTAction(action_type="FILE_GSTR3B"))

    assert "gstr3b" in obs.metadata
    gstr3b = obs.metadata["gstr3b"]
    assert "net_payable" in gstr3b
    assert "itc_claimed" in gstr3b
    assert "gst_collected" in gstr3b


# ---------------------------------------------------------------------------
# Test: state property
# ---------------------------------------------------------------------------

def test_state_returns_state_object(env_l1):
    env_l1.reset(seed=42)
    s = env_l1.state
    assert isinstance(s, State)


def test_state_has_episode_id(env_l1):
    env_l1.reset(seed=42)
    s = env_l1.state
    assert s.episode_id is not None
    assert isinstance(s.episode_id, str)


def test_state_step_count_matches_day(env_l1):
    env_l1.reset(seed=42)
    env_l1.step(GSTAction(action_type="DO_NOTHING"))
    env_l1.step(GSTAction(action_type="DO_NOTHING"))
    s = env_l1.state
    assert s.step_count == 3  # Day 3 after 2 steps from Day 1


# ---------------------------------------------------------------------------
# Test: full episode walkthrough
# ---------------------------------------------------------------------------

def test_full_episode_l1_completes_without_error(env_l1):
    """Run a full L1 episode — no exceptions allowed at any step."""
    obs = env_l1.reset(seed=42)
    done = False
    step_count = 0

    while not done and step_count < 50:
        # Simple greedy strategy: fulfill sales first, then pay vendors
        action = GSTAction(action_type="DO_NOTHING")
        if obs.pending_sales:
            action = GSTAction(action_type="FULFILL_SALE", transaction_id=obs.pending_sales[0].id)
        elif obs.pending_purchases:
            action = GSTAction(action_type="PAY_VENDOR", transaction_id=obs.pending_purchases[0].id)
        elif env_l1._day >= 8:
            action = GSTAction(action_type="FILE_GSTR3B")

        obs = env_l1.step(action)
        done = obs.done
        step_count += 1

    assert done, "Episode never terminated"
    assert obs.reward is not None


def test_reward_is_higher_with_vendor_payment_before_filing(env_l1):
    """
    Agent that pays vendors before filing should get more reward than
    one that files immediately without paying any vendors.
    """
    # Strategy A: pay vendor then file
    obs_a = env_l1.reset(seed=42)
    sale_id = obs_a.pending_sales[0].id
    # First fulfill sale to get cash
    env_l1.step(GSTAction(action_type="FULFILL_SALE", transaction_id=sale_id))
    purch_id = env_l1._build_observation().pending_purchases[0].id if env_l1._build_observation().pending_purchases else None
    if purch_id:
        env_l1.step(GSTAction(action_type="PAY_VENDOR", transaction_id=purch_id))
    # Advance to Day 8 minimum
    while env_l1._day < 8:
        env_l1.step(GSTAction(action_type="DO_NOTHING"))
    obs_a_final = env_l1.step(GSTAction(action_type="FILE_GSTR3B"))

    # Strategy B: file immediately at Day 8 without paying vendors
    obs_b = env_l1.reset(seed=42)
    while env_l1._day < 8:
        env_l1.step(GSTAction(action_type="DO_NOTHING"))
    obs_b_final = env_l1.step(GSTAction(action_type="FILE_GSTR3B"))

    # Strategy A should yield higher reward (ITC utilized)
    assert obs_a_final.reward >= obs_b_final.reward
