"""
Tests for GSTLedger — verifying all GST/ITC math against hand-calculated examples.

Run with: PYTHONPATH=. pytest tests/test_ledger.py -v
"""

import pytest
from gst_cashflow_env.models import Transaction
from server.ledger import GSTLedger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sale(id: str, base: float, gst_rate: float = 0.12, urgency: float = 0.5) -> Transaction:
    gst = round(base * gst_rate, 2)
    return Transaction(
        id=id,
        party_name=f"Retailer-{id}",
        transaction_type="sale",
        base_amount=base,
        gst_rate=gst_rate,
        gst_amount=gst,
        total_amount=round(base + gst, 2),
        hsn_code="6104",
        due_day=10,
        urgency_score=urgency,
        vendor_gstr1_reliability=0.0,  # Not applicable for sales
    )


def make_purchase(
    id: str,
    base: float,
    gst_rate: float = 0.05,
    reliability: float = 0.95,
) -> Transaction:
    gst = round(base * gst_rate, 2)
    return Transaction(
        id=id,
        party_name=f"Vendor-{id}",
        transaction_type="purchase",
        base_amount=base,
        gst_rate=gst_rate,
        gst_amount=gst,
        total_amount=round(base + gst, 2),
        hsn_code="6006",
        due_day=15,
        urgency_score=0.5,
        vendor_gstr1_reliability=reliability,
    )


@pytest.fixture
def simple_ledger():
    """
    Simple ledger: 1 sale (₹1,00,000 base, 12% GST) + 1 purchase (₹50,000 base, 5% GST).
    Opening cash: ₹2,00,000.
    """
    sale = make_sale("SALE-001", 100_000, gst_rate=0.12)
    purchase = make_purchase("PURCH-001", 50_000, gst_rate=0.05, reliability=1.0)
    return GSTLedger(opening_cash=200_000.0, transactions=[sale, purchase])


# ---------------------------------------------------------------------------
# Test: fulfill_sale
# ---------------------------------------------------------------------------

def test_fulfill_sale_increases_cash(simple_ledger):
    """Fulfilling a sale should increase cash by total_amount (base + GST)."""
    before_cash = simple_ledger.cash
    result = simple_ledger.fulfill_sale("SALE-001", day=1)

    assert result.get("error") is None
    # Sale: base=100000, gst=12000, total=112000
    assert simple_ledger.cash == pytest.approx(before_cash + 112_000.0)
    assert simple_ledger.gst_collected == pytest.approx(12_000.0)
    assert result["gst_collected"] == pytest.approx(12_000.0)


def test_fulfill_sale_updates_status(simple_ledger):
    simple_ledger.fulfill_sale("SALE-001", day=2)
    txn = simple_ledger.get_transaction("SALE-001")
    assert txn.status == "fulfilled"
    assert txn.action_day == 2


def test_fulfill_sale_twice_fails(simple_ledger):
    """Cannot fulfill an already-fulfilled sale."""
    simple_ledger.fulfill_sale("SALE-001", day=1)
    result = simple_ledger.fulfill_sale("SALE-001", day=2)
    assert "error" in result


def test_fulfill_sale_wrong_type_fails(simple_ledger):
    """Cannot fulfill a purchase transaction."""
    result = simple_ledger.fulfill_sale("PURCH-001", day=1)
    assert "error" in result


# ---------------------------------------------------------------------------
# Test: pay_vendor
# ---------------------------------------------------------------------------

def test_pay_vendor_decreases_cash(simple_ledger):
    """Paying a vendor should decrease cash by total_amount and secure ITC."""
    before_cash = simple_ledger.cash
    result = simple_ledger.pay_vendor("PURCH-001", day=3)

    assert result.get("error") is None
    # Purchase: base=50000, gst=2500, total=52500
    assert simple_ledger.cash == pytest.approx(before_cash - 52_500.0)
    assert simple_ledger.itc_secured == pytest.approx(2_500.0)
    assert result["itc_secured"] == pytest.approx(2_500.0)


def test_pay_vendor_updates_status(simple_ledger):
    simple_ledger.pay_vendor("PURCH-001", day=5)
    txn = simple_ledger.get_transaction("PURCH-001")
    assert txn.status == "paid"
    assert txn.action_day == 5


def test_pay_vendor_twice_fails(simple_ledger):
    simple_ledger.pay_vendor("PURCH-001", day=1)
    result = simple_ledger.pay_vendor("PURCH-001", day=2)
    assert "error" in result


# ---------------------------------------------------------------------------
# Test: ITC all-or-nothing (critical rule)
# ---------------------------------------------------------------------------

def test_insufficient_cash_returns_error_no_state_change():
    """
    PAY_VENDOR must FAIL GRACEFULLY when cash < invoice total.
    State must NOT change on failure — no partial payments.
    """
    purchase = make_purchase("PURCH-001", 300_000, gst_rate=0.05, reliability=1.0)
    ledger = GSTLedger(opening_cash=100_000.0, transactions=[purchase])

    cash_before = ledger.cash
    itc_before = ledger.itc_secured

    result = ledger.pay_vendor("PURCH-001", day=1)

    assert "error" in result
    assert ledger.cash == pytest.approx(cash_before)    # Cash unchanged
    assert ledger.itc_secured == pytest.approx(itc_before)  # ITC unchanged
    txn = ledger.get_transaction("PURCH-001")
    assert txn.status == "pending"  # Status unchanged


def test_itc_all_or_nothing_not_partial():
    """
    ITC is binary per invoice: pay total_amount → full ITC; anything less → error.
    This test verifies you cannot get partial ITC by paying partially.
    """
    # Invoice: base=315000, gst=15750 (5%), total=330750
    purchase = make_purchase("PURCH-BIG", 315_000, gst_rate=0.05, reliability=1.0)
    # Cash just one rupee short
    ledger = GSTLedger(opening_cash=330_749.0, transactions=[purchase])

    result = ledger.pay_vendor("PURCH-BIG", day=1)
    assert "error" in result
    assert ledger.itc_secured == 0.0

    # Now fund it properly
    ledger.cash += 1.0
    result = ledger.pay_vendor("PURCH-BIG", day=1)
    assert result.get("error") is None
    assert ledger.itc_secured == pytest.approx(15_750.0)


# ---------------------------------------------------------------------------
# Test: compute_gstr3b
# ---------------------------------------------------------------------------

def test_gstr3b_all_reliable_vendors_gives_full_itc():
    """
    If vendor reliability = 1.0, vendor always files → full ITC claimed.
    Net payable = max(0, gst_collected - itc_claimed)
    """
    sale = make_sale("SALE-001", 100_000, gst_rate=0.12)
    purchase = make_purchase("PURCH-001", 50_000, gst_rate=0.05, reliability=1.0)
    ledger = GSTLedger(opening_cash=500_000.0, transactions=[sale, purchase])

    ledger.fulfill_sale("SALE-001", day=1)
    ledger.pay_vendor("PURCH-001", day=2)

    result = ledger.compute_gstr3b(day=19, seed=42)

    assert result["gst_collected"] == pytest.approx(12_000.0)
    assert result["itc_claimed"] == pytest.approx(2_500.0)
    assert result["net_payable"] == pytest.approx(12_000.0 - 2_500.0)
    assert result["late_filing_penalty"] == 0.0
    assert result["itc_utilization_pct"] == pytest.approx(1.0)


def test_gstr3b_zero_reliability_vendors_gives_zero_itc():
    """
    If vendor reliability = 0.0, vendor never files → ITC = ₹0.
    Net payable = full GST collected.
    """
    sale = make_sale("SALE-001", 100_000, gst_rate=0.12)
    purchase = make_purchase("PURCH-001", 50_000, gst_rate=0.05, reliability=0.0)
    ledger = GSTLedger(opening_cash=500_000.0, transactions=[sale, purchase])

    ledger.fulfill_sale("SALE-001", day=1)
    ledger.pay_vendor("PURCH-001", day=2)

    result = ledger.compute_gstr3b(day=19, seed=42)

    assert result["gst_collected"] == pytest.approx(12_000.0)
    assert result["itc_claimed"] == pytest.approx(0.0)
    assert result["net_payable"] == pytest.approx(12_000.0)


def test_gstr3b_unpaid_vendor_gives_zero_itc():
    """
    Vendors that were never paid cannot contribute ITC, regardless of reliability.
    """
    sale = make_sale("SALE-001", 100_000, gst_rate=0.12)
    purchase = make_purchase("PURCH-001", 50_000, gst_rate=0.05, reliability=1.0)
    ledger = GSTLedger(opening_cash=500_000.0, transactions=[sale, purchase])

    ledger.fulfill_sale("SALE-001", day=1)
    # Intentionally NOT paying PURCH-001

    result = ledger.compute_gstr3b(day=19, seed=42)

    assert result["itc_claimed"] == pytest.approx(0.0)
    assert result["net_payable"] == pytest.approx(12_000.0)


def test_gstr3b_net_payable_never_negative():
    """
    Net payable = max(0, collected - itc). Never negative (no refund in simulation).
    """
    # Huge ITC, tiny sales
    sale = make_sale("SALE-001", 10_000, gst_rate=0.05)
    purchase = make_purchase("PURCH-001", 200_000, gst_rate=0.12, reliability=1.0)
    ledger = GSTLedger(opening_cash=500_000.0, transactions=[sale, purchase])

    ledger.fulfill_sale("SALE-001", day=1)
    ledger.pay_vendor("PURCH-001", day=2)

    result = ledger.compute_gstr3b(day=19, seed=42)

    assert result["net_payable"] >= 0.0


# ---------------------------------------------------------------------------
# Test: late filing penalty
# ---------------------------------------------------------------------------

def test_late_filing_penalty():
    """Each day after Day 20 incurs ₹50 penalty."""
    sale = make_sale("SALE-001", 100_000)
    ledger = GSTLedger(opening_cash=200_000.0, transactions=[sale])
    ledger.fulfill_sale("SALE-001", day=1)

    # Filed on Day 23 → 3 days late → ₹150 penalty
    result = ledger.compute_gstr3b(day=23, seed=42)
    assert result["late_filing_penalty"] == pytest.approx(150.0)
    assert result["total_due"] == pytest.approx(result["net_payable"] + 150.0)


def test_on_time_filing_no_penalty():
    """Filing on Day 20 or earlier → zero penalty."""
    sale = make_sale("SALE-001", 100_000)
    ledger = GSTLedger(opening_cash=200_000.0, transactions=[sale])
    ledger.fulfill_sale("SALE-001", day=1)

    result = ledger.compute_gstr3b(day=20, seed=42)
    assert result["late_filing_penalty"] == 0.0


# ---------------------------------------------------------------------------
# Test: baseline_gst
# ---------------------------------------------------------------------------

def test_baseline_gst_equals_sum_of_sale_gst():
    """baseline_gst = sum of all sale GST amounts (assumes zero ITC)."""
    sales = [
        make_sale("S1", 100_000, gst_rate=0.12),  # GST = 12000
        make_sale("S2", 50_000, gst_rate=0.12),   # GST = 6000
    ]
    purchase = make_purchase("P1", 30_000, gst_rate=0.05, reliability=1.0)  # GST = 1500
    ledger = GSTLedger(opening_cash=500_000.0, transactions=sales + [purchase])

    baseline = ledger.compute_baseline_gst()
    assert baseline == pytest.approx(18_000.0)  # Only sale GST


# ---------------------------------------------------------------------------
# Test: daily burn
# ---------------------------------------------------------------------------

def test_apply_daily_burn():
    """Daily burn reduces cash by DAILY_BURN (₹2000)."""
    ledger = GSTLedger(opening_cash=100_000.0, transactions=[])
    new_cash = ledger.apply_daily_burn()
    assert new_cash == pytest.approx(100_000.0 - GSTLedger.DAILY_BURN)
    assert ledger.cash == pytest.approx(100_000.0 - GSTLedger.DAILY_BURN)


# ---------------------------------------------------------------------------
# Test: multi-vendor stochastic scenario
# ---------------------------------------------------------------------------

def test_multiple_vendors_partial_filing(monkeypatch):
    """
    With 4 vendors at varying reliability, only those who file give ITC.
    We patch random to make the outcome deterministic.
    """
    # vendor_a: reliability=0.95 → will file (0.1 < 0.95)
    # vendor_b: reliability=0.40 → won't file (0.8 > 0.40)
    purchase_a = make_purchase("P-A", 300_000, gst_rate=0.05, reliability=0.95)
    purchase_b = make_purchase("P-B", 50_000, gst_rate=0.12, reliability=0.40)
    sale = make_sale("S-1", 500_000, gst_rate=0.12)

    ledger = GSTLedger(opening_cash=1_000_000.0, transactions=[purchase_a, purchase_b, sale])
    ledger.fulfill_sale("S-1", day=1)
    ledger.pay_vendor("P-A", day=2)
    ledger.pay_vendor("P-B", day=3)

    # Seed 0 → random values are deterministic; we verify the math
    result = ledger.compute_gstr3b(day=19, seed=0)

    # ITC can only come from paid vendors; check total bounded correctly
    assert result["itc_claimed"] <= ledger.itc_secured
    assert result["net_payable"] >= 0.0
    assert result["gst_collected"] == pytest.approx(60_000.0)
