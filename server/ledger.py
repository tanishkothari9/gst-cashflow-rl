"""
GST/ITC Calculation Engine.

Pure math. No RL logic, no reward logic, no randomness (except in compute_gstr3b
where the Bernoulli roll for vendor GSTR-1 filing is intentional and documented).

This is the single source of truth for all financial calculations.

RULES ENCODED HERE (all legally accurate as of GSTR-3B rules 2025):
  1. ITC is all-or-nothing per invoice — ₹1 short of full payment = ₹0 ITC
  2. ITC only claimable if vendor paid AND vendor files GSTR-1 (stochastic)
  3. Net payable = max(0, gst_collected - itc_claimed) — no refunds in simulation
  4. Cash cannot go negative — PAY_VENDOR enforces this as a hard constraint
  5. Late filing (day > 20): ₹50 per day penalty
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional

from gst_cashflow_env.models import Transaction


class GSTLedger:
    """
    Tracks all GST obligations exactly as ERPNext would for an apparel SME.

    One GSTLedger instance per episode. All mutations go through methods —
    never modify self.transactions directly from outside.
    """

    CASH_FLOOR: float = 50_000.0      # Minimum operating cash (salaries + rent)
    DAILY_BURN: float = 2_000.0       # Fixed daily overhead

    def __init__(self, opening_cash: float, transactions: List[Transaction]) -> None:
        self.cash: float = opening_cash
        # Deep-copy so mutations don't bleed between episodes
        self.transactions: Dict[str, Transaction] = {
            t.id: copy.deepcopy(t) for t in transactions
        }
        self.gst_collected: float = 0.0
        self.itc_secured: float = 0.0    # Secured = fully paid; not yet claimed
        self.itc_claimed: float = 0.0    # Claimed = secured + vendor filed GSTR-1

    # ------------------------------------------------------------------
    # Core actions
    # ------------------------------------------------------------------

    def fulfill_sale(self, transaction_id: str, day: int) -> Dict:
        """
        Execute a sale order.

        Cash increases by total_amount (base + GST).
        GST collected increases by gst_amount.

        Returns:
            {"cash_delta": float, "gst_collected": float}
            or {"error": str} on invalid input
        """
        txn = self.transactions.get(transaction_id)
        if txn is None:
            return {"error": f"Transaction {transaction_id} not found"}
        if txn.transaction_type != "sale":
            return {"error": f"{transaction_id} is not a sale"}
        if txn.status != "pending":
            return {"error": f"{transaction_id} is already {txn.status}"}

        # Customer pays on delivery — cash arrives immediately
        self.cash += txn.total_amount
        self.gst_collected += txn.gst_amount

        txn.status = "fulfilled"
        txn.action_day = day

        return {"cash_delta": txn.total_amount, "gst_collected": txn.gst_amount}

    def pay_vendor(self, transaction_id: str, day: int) -> Dict:
        """
        Pay a vendor invoice in full.

        ITC is SECURED here (tracked) but NOT CLAIMED.
        Actual ITC claim is determined at FILE_GSTR3B time via Bernoulli roll.

        All-or-nothing: we only pay in full — no partial payments.

        Returns:
            {"cash_delta": float, "itc_secured": float}
            or {"error": str} if insufficient cash or invalid transaction
        """
        txn = self.transactions.get(transaction_id)
        if txn is None:
            return {"error": f"Transaction {transaction_id} not found"}
        if txn.transaction_type != "purchase":
            return {"error": f"{transaction_id} is not a purchase"}
        if txn.status != "pending":
            return {"error": f"{transaction_id} is already {txn.status}"}
        # Hard cash constraint — cannot go into negative cash
        if self.cash < txn.total_amount:
            return {
                "error": (
                    f"Insufficient cash: have ₹{self.cash:.0f}, "
                    f"need ₹{txn.total_amount:.0f} for {transaction_id}"
                )
            }

        # Deduct full invoice amount (base + GST)
        self.cash -= txn.total_amount
        # ITC secured — pending vendor's GSTR-1 filing
        self.itc_secured += txn.gst_amount

        txn.status = "paid"
        txn.action_day = day

        return {"cash_delta": -txn.total_amount, "itc_secured": txn.gst_amount}

    def defer_transaction(self, transaction_id: str) -> Dict:
        """Mark a transaction as deferred (agent chose to postpone it)."""
        txn = self.transactions.get(transaction_id)
        if txn is None:
            return {"error": f"Transaction {transaction_id} not found"}
        if txn.status != "pending":
            return {"error": f"{transaction_id} is already {txn.status}"}
        txn.status = "deferred"
        return {"deferred": transaction_id}

    def restore_deferred(self, transaction_id: str) -> None:
        """Restore a deferred transaction to pending (allows re-attempting)."""
        txn = self.transactions.get(transaction_id)
        if txn and txn.status == "deferred":
            txn.status = "pending"

    def apply_daily_burn(self) -> float:
        """
        Deduct daily fixed operating costs (salaries, rent, utilities).
        Called every step regardless of action taken.

        Returns: new cash balance
        """
        self.cash -= self.DAILY_BURN
        return self.cash

    # ------------------------------------------------------------------
    # Filing
    # ------------------------------------------------------------------

    def compute_gstr3b(self, day: int, seed: Optional[int] = None) -> Dict:
        """
        Compute the final GSTR-3B at filing time.

        For each fully paid vendor invoice:
          - Roll Bernoulli(vendor_gstr1_reliability) to determine if vendor filed GSTR-1
          - If filed: ITC appears in GSTR-2B → claimable
          - If not: ITC does NOT appear → cannot be claimed this month

        CRITICAL: The Bernoulli roll happens HERE, not at pay_vendor time.
        This is the only source of randomness in the ledger.

        Args:
            day: The filing day (used for late-filing penalty calculation)
            seed: Optional RNG seed for reproducible tests

        Returns:
            Full GSTR-3B breakdown dict
        """
        rng = random.Random(seed) if seed is not None else random

        itc_claimed = 0.0
        vendor_filing_results: Dict[str, bool] = {}

        for txn in self.transactions.values():
            if txn.transaction_type == "purchase" and txn.status == "paid":
                # Roll the dice: did this vendor file their GSTR-1 this month?
                filed = rng.random() < txn.vendor_gstr1_reliability
                txn.itc_appeared_in_gstr2b = filed
                vendor_filing_results[f"{txn.party_name}|{txn.id}"] = filed
                if filed:
                    itc_claimed += txn.gst_amount

        self.itc_claimed = itc_claimed

        net_payable = max(0.0, self.gst_collected - itc_claimed)

        # ₹50/day penalty for each day past the 20th
        late_filing_penalty = max(0.0, (day - 20) * 50.0) if day > 20 else 0.0

        return {
            "gst_collected": self.gst_collected,
            "itc_secured": self.itc_secured,          # What was paid (attempted)
            "itc_claimed": self.itc_claimed,           # What appeared in GSTR-2B
            "net_payable": net_payable,
            "late_filing_penalty": late_filing_penalty,
            "total_due": net_payable + late_filing_penalty,
            "itc_utilization_pct": (
                itc_claimed / max(1.0, self.itc_secured)
            ),
            "vendor_filing_results": vendor_filing_results,
            "filing_day": day,
        }

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def compute_baseline_gst(self) -> float:
        """
        What a naive greedy agent would pay.

        Naive strategy: fulfill ALL sales on Day 1 (max cash, no ITC urgency),
        pay ALL vendors on Day 21 (after filing deadline → zero ITC claimed).

        This is computed once at episode start and stored in the observation.
        Storing it once prevents the agent from gaming it by manipulating state.

        Returns: total GST that would be paid with zero ITC utilization
        """
        return sum(
            t.gst_amount
            for t in self.transactions.values()
            if t.transaction_type == "sale"
        )

    # ------------------------------------------------------------------
    # Helpers / introspection
    # ------------------------------------------------------------------

    def get_pending_sales(self) -> List[Transaction]:
        """All sale transactions currently in 'pending' status."""
        return [
            t for t in self.transactions.values()
            if t.transaction_type == "sale" and t.status == "pending"
        ]

    def get_pending_purchases(self) -> List[Transaction]:
        """All purchase transactions currently in 'pending' status."""
        return [
            t for t in self.transactions.values()
            if t.transaction_type == "purchase" and t.status == "pending"
        ]

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        return self.transactions.get(transaction_id)
