"""
Data models for the GST Cash Flow Optimization Environment.

These are the contracts between client and server.
All models use Pydantic v2. The separation between Transaction (pure data),
GSTObservation (what the agent sees), and GSTAction (what the agent does)
mirrors real ERPNext data structures.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Transaction(BaseModel):
    """
    Represents one vendor invoice (purchase) or one customer sales order (sale).

    Each Transaction is one independently-binary ITC unit:
    - For purchases: ITC is claimable only on FULL payment AND vendor GSTR-1 filing.
    - For sales: paying customers → cash inflow + GST collection.
    One vendor can have multiple Transaction objects (multiple invoices).
    """

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(..., description="Unique ID, e.g. 'TXN-001'")
    party_name: str = Field(..., description="'Retailer A' or 'Supplier X'")
    transaction_type: Literal["sale", "purchase"]

    base_amount: float = Field(..., gt=0, description="Amount before GST (INR)")
    gst_rate: float = Field(..., description="Tax rate: 0.05, 0.12, or 0.18")
    gst_amount: float = Field(..., description="base_amount * gst_rate (computed)")
    total_amount: float = Field(..., description="base_amount + gst_amount (computed)")

    hsn_code: str = Field(..., description="e.g. '6104' for women's apparel")
    due_day: int = Field(..., ge=1, le=30, description="Day the payment/fulfillment is due")
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="0.0 low → 1.0 critical")

    # Meaningful only for purchases; set to 0.0 for sales as a safe default
    vendor_gstr1_reliability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability vendor files GSTR-1 (0.0–1.0). Sales: 0.0 (unused).",
    )

    # Mutable state — updated during episode execution
    status: Literal["pending", "fulfilled", "paid", "deferred"] = Field(
        default="pending",
        description="Current lifecycle status of this transaction",
    )
    action_day: Optional[int] = Field(
        default=None, description="Which day the action was taken"
    )
    itc_appeared_in_gstr2b: Optional[bool] = Field(
        default=None, description="Revealed at filing: did vendor's GSTR-1 appear?"
    )

    @model_validator(mode="after")
    def validate_amounts(self) -> "Transaction":
        """Enforce that computed amount fields are consistent."""
        expected_gst = round(self.base_amount * self.gst_rate, 2)
        if abs(self.gst_amount - expected_gst) > 1.0:
            raise ValueError(
                f"gst_amount {self.gst_amount} ≠ base_amount × gst_rate "
                f"({self.base_amount} × {self.gst_rate} = {expected_gst})"
            )
        expected_total = round(self.base_amount + self.gst_amount, 2)
        if abs(self.total_amount - expected_total) > 1.0:
            raise ValueError(
                f"total_amount {self.total_amount} ≠ base_amount + gst_amount "
                f"({self.base_amount} + {self.gst_amount} = {expected_total})"
            )
        if self.transaction_type == "purchase" and self.vendor_gstr1_reliability == 0.0:
            # 0.0 is a valid reliability value (unreliable vendor) — not an error.
            pass
        return self


class GSTObservation(Observation):
    """
    What the agent sees at the start of each day.

    Inherits done, reward, metadata from openenv Observation base class.
    All GST-specific fields are added here.
    """

    # --- Time ---
    day: int = Field(..., ge=1, le=30, description="Current day (1–30)")
    days_to_filing: int = Field(
        ..., description="Days remaining until GSTR-3B deadline (20 - day)"
    )

    # --- Financials ---
    cash_balance: float = Field(..., description="Current cash in bank (INR)")
    gst_collected_so_far: float = Field(
        default=0.0, description="GST accumulated from all fulfilled sales this month"
    )
    itc_secured_so_far: float = Field(
        default=0.0,
        description="ITC from fully paid vendor invoices (pending GSTR-1 filing roll)",
    )
    net_gst_if_filed_now: float = Field(
        default=0.0,
        description="gst_collected - itc_secured: current net GST position",
    )
    baseline_gst: float = Field(
        ...,
        description=(
            "What a naive greedy agent would pay (all sales first, all vendors late). "
            "Fixed at episode start — used in terminal reward."
        ),
    )

    # --- Pending transactions (what the agent can act on today) ---
    pending_sales: List[Transaction] = Field(
        default_factory=list, description="Unfulfilled retailer orders"
    )
    pending_purchases: List[Transaction] = Field(
        default_factory=list, description="Unpaid vendor invoices"
    )

    # --- Relationship scores (degrade when transactions are deferred) ---
    vendor_scores: Dict[str, float] = Field(
        default_factory=dict, description="{'Supplier X': 0.85} — degrades on defer"
    )
    retailer_scores: Dict[str, float] = Field(
        default_factory=dict, description="{'Retailer A': 0.92} — degrades on defer"
    )

    # --- Episode metadata ---
    episode_id: str = Field(default="", description="Unique identifier for this episode")
    difficulty_level: Literal["L1", "L2", "L3", "L4"] = Field(
        default="L1", description="Curriculum difficulty level"
    )


class GSTAction(Action):
    """
    One action the agent takes per day.

    Inherits metadata from openenv Action base class.
    transaction_id is required for all actions except FILE_GSTR3B and DO_NOTHING.
    """

    action_type: Literal[
        "FULFILL_SALE",   # Execute a sale: cash in, GST collected
        "PAY_VENDOR",     # Pay vendor fully: ITC secured (if vendor files)
        "DEFER_SALE",     # Postpone fulfillment: retailer urgency decays, score drops
        "DEFER_VENDOR",   # Defer vendor payment: vendor score drops
        "FILE_GSTR3B",    # End episode: triggers GSTR-3B calculation + terminal reward
        "DO_NOTHING",     # Wait (costs -0.5 per step to discourage laziness)
    ] = Field(..., description="Which action to take this day")

    transaction_id: Optional[str] = Field(
        default=None,
        description=(
            "Required for FULFILL_SALE, PAY_VENDOR, DEFER_SALE, DEFER_VENDOR. "
            "Not used for FILE_GSTR3B or DO_NOTHING."
        ),
    )

    @model_validator(mode="after")
    def validate_transaction_id(self) -> "GSTAction":
        """Ensure transaction_id is provided for actions that require it."""
        needs_txn = {"FULFILL_SALE", "PAY_VENDOR", "DEFER_SALE", "DEFER_VENDOR"}
        if self.action_type in needs_txn and not self.transaction_id:
            raise ValueError(
                f"action_type='{self.action_type}' requires transaction_id to be set"
            )
        return self
