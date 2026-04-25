"""
Tests for ERPNextScenarioGenerator — verifying scenarios are valid and realistic.

Run with: PYTHONPATH=. pytest tests/test_scenarios.py -v
"""

import pytest
from server.scenario_generator import ERPNextScenarioGenerator


@pytest.fixture
def gen():
    return ERPNextScenarioGenerator()


# ---------------------------------------------------------------------------
# Basic structure validation
# ---------------------------------------------------------------------------

def test_scenario_has_required_keys(gen):
    scenario = gen.generate_episode("L1", seed=42)
    for key in ("opening_cash", "transactions", "baseline_gst", "difficulty", "episode_id", "episode_days"):
        assert key in scenario


def test_scenario_has_correct_transaction_count_l1(gen):
    scenario = gen.generate_episode("L1", seed=42)
    sales = [t for t in scenario["transactions"] if t.transaction_type == "sale"]
    purchases = [t for t in scenario["transactions"] if t.transaction_type == "purchase"]
    config = ERPNextScenarioGenerator.DIFFICULTY_CONFIG["L1"]
    assert len(sales) == config["num_sales"]
    assert len(purchases) == config["num_purchases"]


def test_scenario_has_correct_transaction_count_l4(gen):
    scenario = gen.generate_episode("L4", seed=10)
    sales = [t for t in scenario["transactions"] if t.transaction_type == "sale"]
    purchases = [t for t in scenario["transactions"] if t.transaction_type == "purchase"]
    config = ERPNextScenarioGenerator.DIFFICULTY_CONFIG["L4"]
    assert len(sales) == config["num_sales"]
    assert len(purchases) == config["num_purchases"]


# ---------------------------------------------------------------------------
# Financial invariants
# ---------------------------------------------------------------------------

def test_transaction_gst_amounts_are_correct(gen):
    """gst_amount must equal base_amount * gst_rate (within ₹1 rounding)."""
    for difficulty in ("L1", "L2", "L3", "L4"):
        scenario = gen.generate_episode(difficulty, seed=7)
        for t in scenario["transactions"]:
            expected_gst = t.base_amount * t.gst_rate
            assert abs(t.gst_amount - expected_gst) <= 1.0, (
                f"{t.id}: gst_amount={t.gst_amount} vs expected={expected_gst}"
            )


def test_transaction_total_amounts_are_correct(gen):
    """total_amount must equal base_amount + gst_amount."""
    for difficulty in ("L1", "L2", "L3", "L4"):
        scenario = gen.generate_episode(difficulty, seed=8)
        for t in scenario["transactions"]:
            expected_total = t.base_amount + t.gst_amount
            assert abs(t.total_amount - expected_total) <= 1.0


def test_baseline_gst_equals_sum_of_sale_gst(gen):
    """baseline_gst must be the sum of all sale gst_amounts."""
    for difficulty in ("L1", "L2", "L3", "L4"):
        scenario = gen.generate_episode(difficulty, seed=9)
        expected = sum(t.gst_amount for t in scenario["transactions"] if t.transaction_type == "sale")
        assert abs(scenario["baseline_gst"] - expected) < 0.01


def test_opening_cash_matches_difficulty_config(gen):
    """Opening cash must match the difficulty configuration."""
    for difficulty, config in ERPNextScenarioGenerator.DIFFICULTY_CONFIG.items():
        scenario = gen.generate_episode(difficulty, seed=1)
        assert scenario["opening_cash"] == config["opening_cash"]


# ---------------------------------------------------------------------------
# Cash constraint
# ---------------------------------------------------------------------------

def test_l3_l4_vendor_bills_exceed_opening_cash(gen):
    """L3 and L4 scenarios must have total vendor bills > opening cash."""
    for difficulty in ("L3", "L4"):
        scenario = gen.generate_episode(difficulty, seed=42)
        total_vendor = sum(
            t.total_amount for t in scenario["transactions"] if t.transaction_type == "purchase"
        )
        assert total_vendor > scenario["opening_cash"], (
            f"{difficulty}: vendor bills ({total_vendor:.0f}) must exceed cash ({scenario['opening_cash']:.0f})"
        )


# ---------------------------------------------------------------------------
# Field validity
# ---------------------------------------------------------------------------

def test_all_transaction_ids_are_unique(gen):
    """Transaction IDs must be unique within an episode."""
    scenario = gen.generate_episode("L4", seed=42)
    ids = [t.id for t in scenario["transactions"]]
    assert len(ids) == len(set(ids))


def test_all_transactions_start_pending(gen):
    """All transactions must start in 'pending' status."""
    scenario = gen.generate_episode("L4", seed=42)
    for t in scenario["transactions"]:
        assert t.status == "pending"


def test_urgency_scores_in_range(gen):
    """All urgency scores must be in [0.0, 1.0]."""
    scenario = gen.generate_episode("L4", seed=42)
    for t in scenario["transactions"]:
        assert 0.0 <= t.urgency_score <= 1.0


def test_vendor_reliability_in_range(gen):
    """Vendor GSTR-1 reliability must be in [0.0, 1.0]."""
    scenario = gen.generate_episode("L4", seed=42)
    for t in scenario["transactions"]:
        assert 0.0 <= t.vendor_gstr1_reliability <= 1.0


def test_due_days_in_episode_range(gen):
    """All due days must be valid (1–30)."""
    scenario = gen.generate_episode("L4", seed=42)
    for t in scenario["transactions"]:
        assert 1 <= t.due_day <= 30


def test_hsn_codes_are_known(gen):
    """HSN codes must be from the product catalog."""
    known_hsns = {v["hsn"] for v in ERPNextScenarioGenerator.PRODUCT_CATALOG.values()}
    scenario = gen.generate_episode("L4", seed=42)
    for t in scenario["transactions"]:
        assert t.hsn_code in known_hsns


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_scenario(gen):
    """The same seed should produce identical scenarios."""
    s1 = gen.generate_episode("L3", seed=123)
    s2 = gen.generate_episode("L3", seed=123)
    assert s1["baseline_gst"] == s2["baseline_gst"]
    assert s1["opening_cash"] == s2["opening_cash"]
    assert len(s1["transactions"]) == len(s2["transactions"])
    for t1, t2 in zip(s1["transactions"], s2["transactions"]):
        assert t1.id == t2.id
        assert t1.base_amount == t2.base_amount
        assert t1.gst_rate == t2.gst_rate


def test_different_seeds_produce_different_scenarios(gen):
    """Different seeds should (almost certainly) produce different scenarios."""
    s1 = gen.generate_episode("L3", seed=1)
    s2 = gen.generate_episode("L3", seed=9999)
    # At least one transaction should differ in base_amount
    amounts_1 = {t.base_amount for t in s1["transactions"]}
    amounts_2 = {t.base_amount for t in s2["transactions"]}
    assert amounts_1 != amounts_2


def test_unknown_difficulty_raises(gen):
    """Requesting an invalid difficulty level should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown difficulty"):
        gen.generate_episode("L99")
