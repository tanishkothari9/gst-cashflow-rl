"""
ERPNext-realistic scenario generator for the GST Cash Flow Optimization environment.

Based on Smart Choice India — a mid-sized apparel manufacturer in Bengaluru
running ERPNext. Generates transaction data that matches real Indian SME patterns:
  - HSN codes and GST rates from real Indian GST law (effective 2025)
  - Vendor reliability scores calibrated to real MSME GSTR-1 filing behavior
  - Opening cash constraints set to force the agent to earn before paying
  - Urgency scores that reflect seasonal retail patterns (festival, clearance, etc.)

Curriculum levels L1–L4 progressively increase complexity to enable stable training.
"""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, Optional

from gst_cashflow_env.models import Transaction


class ERPNextScenarioGenerator:
    """
    Generates complete episode scenarios for GST Cash Flow Optimization training.

    Each scenario contains:
      - opening_cash: starting bank balance
      - transactions: mix of retailer sales orders and vendor purchase invoices
      - baseline_gst: pre-computed naive-agent GST (used in terminal reward)
      - difficulty: L1/L2/L3/L4
      - episode_id: unique string for tracking

    One vendor can have multiple Transaction objects (multiple invoices).
    """

    # -----------------------------------------------------------------------
    # Real HSN codes and GST rates (Indian GST law, 2025)
    # -----------------------------------------------------------------------
    PRODUCT_CATALOG: Dict[str, Dict] = {
        "kurtas":        {"hsn": "6104", "gst_rate": 0.12, "type": "sale"},
        "shirts":        {"hsn": "6105", "gst_rate": 0.12, "type": "sale"},
        "fabric":        {"hsn": "6006", "gst_rate": 0.05, "type": "purchase"},
        "packaging":     {"hsn": "4819", "gst_rate": 0.18, "type": "purchase"},
        "trims_buttons": {"hsn": "9606", "gst_rate": 0.12, "type": "purchase"},
    }

    # Vendor reliability profiles (calibrated to real Indian MSME filing behavior)
    # Higher payment_strictness → vendor penalizes late payment more severely
    VENDOR_PROFILES: Dict[str, Dict] = {
        "Supplier X (Fabric - Large)":   {"reliability": 0.95, "payment_strictness": 0.9},
        "Supplier Y (Fabric - Medium)":  {"reliability": 0.60, "payment_strictness": 0.6},
        "Supplier Z (Trims)":            {"reliability": 0.90, "payment_strictness": 0.7},
        "Packaging Co":                  {"reliability": 0.40, "payment_strictness": 0.5},
    }

    RETAILER_PROFILES: Dict[str, Dict] = {
        "Retailer A (Large Chain)":  {"urgency_base": 0.8, "payment_speed": "fast"},
        "Retailer B (Mid Retailer)": {"urgency_base": 0.5, "payment_speed": "medium"},
        "Retailer C (Small Shop)":   {"urgency_base": 0.3, "payment_speed": "slow"},
    }

    # Curriculum configuration
    DIFFICULTY_CONFIG: Dict[str, Dict] = {
        "L1": {
            "num_sales": 3,
            "num_purchases": 2,
            "episode_days": 10,
            "opening_cash": 800_000.0,    # ₹8L — cash rich, easy
            "vendor_reliability_fixed": 0.95,  # All reliable for basic learning
            "description": "Cash rich, few transactions, reliable vendors",
        },
        "L2": {
            "num_sales": 6,
            "num_purchases": 4,
            "episode_days": 15,
            "opening_cash": 400_000.0,    # ₹4L — moderate constraint
            "vendor_reliability_fixed": 0.80,
            "description": "Moderate cash, more transactions",
        },
        "L3": {
            "num_sales": 10,
            "num_purchases": 6,
            "episode_days": 20,
            "opening_cash": 200_000.0,    # ₹2L — tight
            "vendor_reliability_fixed": None,  # Variable — use real profiles
            "description": "Tight cash, variable vendor reliability",
        },
        "L4": {
            "num_sales": 15,
            "num_purchases": 8,
            "episode_days": 30,
            "opening_cash": 50_000.0,     # ₹50K — cash crisis, full complexity
            "vendor_reliability_fixed": None,
            "description": "Cash crisis, full 30-day cycle, all complexity",
        },
    }

    # Base amounts for sales and purchases (INR) — randomized within realistic ranges
    SALE_BASE_RANGE = (50_000, 300_000)       # Retailer order values
    PURCHASE_BASE_RANGE = (30_000, 350_000)   # Vendor invoice values

    def generate_episode(
        self,
        difficulty: str = "L1",
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a complete episode scenario.

        Args:
            difficulty: L1/L2/L3/L4 curriculum level
            seed: Optional random seed for reproducibility in tests

        Returns:
            {
                "opening_cash": float,
                "transactions": List[Transaction],
                "baseline_gst": float,   # Pre-computed at episode start
                "difficulty": str,
                "episode_id": str,
                "episode_days": int,
            }
        """
        if difficulty not in self.DIFFICULTY_CONFIG:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. Choose from {list(self.DIFFICULTY_CONFIG)}"
            )

        config = self.DIFFICULTY_CONFIG[difficulty]
        rng = random.Random(seed)

        transactions: List[Transaction] = []

        # -----------------------------------------------------------------------
        # Generate sales transactions (retailer orders)
        # -----------------------------------------------------------------------
        sale_products = ["kurtas", "shirts"]
        retailer_names = list(self.RETAILER_PROFILES.keys())

        for i in range(config["num_sales"]):
            retailer = retailer_names[i % len(retailer_names)]
            profile = self.RETAILER_PROFILES[retailer]
            product = rng.choice(sale_products)
            catalog = self.PRODUCT_CATALOG[product]

            base_amount = round(rng.uniform(*self.SALE_BASE_RANGE), -3)  # Round to ₹1000
            gst_rate = catalog["gst_rate"]
            gst_amount = round(base_amount * gst_rate, 2)
            total_amount = round(base_amount + gst_amount, 2)

            # Urgency: base urgency + seasonal jitter
            urgency = min(1.0, profile["urgency_base"] + rng.uniform(-0.1, 0.2))

            # Due day spread across the episode window
            due_day = rng.randint(3, max(4, config["episode_days"] - 2))

            transactions.append(Transaction(
                id=f"SALE-{i+1:03d}",
                party_name=retailer,
                transaction_type="sale",
                base_amount=base_amount,
                gst_rate=gst_rate,
                gst_amount=gst_amount,
                total_amount=total_amount,
                hsn_code=catalog["hsn"],
                due_day=due_day,
                urgency_score=round(urgency, 2),
                vendor_gstr1_reliability=0.0,  # Not applicable for sales
            ))

        # -----------------------------------------------------------------------
        # Generate purchase transactions (vendor invoices)
        # -----------------------------------------------------------------------
        purchase_products = ["fabric", "packaging", "trims_buttons"]
        vendor_names = list(self.VENDOR_PROFILES.keys())
        reliability_fixed = config.get("vendor_reliability_fixed")

        for i in range(config["num_purchases"]):
            vendor = vendor_names[i % len(vendor_names)]
            vprofile = self.VENDOR_PROFILES[vendor]
            product = purchase_products[i % len(purchase_products)]
            catalog = self.PRODUCT_CATALOG[product]

            base_amount = round(rng.uniform(*self.PURCHASE_BASE_RANGE), -3)
            gst_rate = catalog["gst_rate"]
            gst_amount = round(base_amount * gst_rate, 2)
            total_amount = round(base_amount + gst_amount, 2)

            # Reliability: fixed for L1/L2, profile-based for L3/L4
            if reliability_fixed is not None:
                reliability = reliability_fixed
            else:
                # Add per-episode jitter to base reliability (±0.10)
                reliability = min(1.0, max(0.1, vprofile["reliability"] + rng.uniform(-0.1, 0.1)))

            due_day = rng.randint(3, max(4, config["episode_days"] - 2))

            transactions.append(Transaction(
                id=f"PURCH-{i+1:03d}",
                party_name=vendor,
                transaction_type="purchase",
                base_amount=base_amount,
                gst_rate=gst_rate,
                gst_amount=gst_amount,
                total_amount=total_amount,
                hsn_code=catalog["hsn"],
                due_day=due_day,
                urgency_score=round(vprofile["payment_strictness"], 2),
                vendor_gstr1_reliability=round(reliability, 2),
            ))

        # -----------------------------------------------------------------------
        # Cash constraint enforcement for L3/L4
        # -----------------------------------------------------------------------
        # For cash-constrained levels, ensure total vendor bills > opening cash.
        # This forces the agent to earn cash from sales before paying vendors.
        total_vendor_bills = sum(
            t.total_amount for t in transactions if t.transaction_type == "purchase"
        )
        opening_cash = config["opening_cash"]
        if difficulty in ("L3", "L4") and total_vendor_bills <= opening_cash:
            # Scale up vendor bills to be 1.5× opening cash
            scale = (opening_cash * 1.5) / max(1.0, total_vendor_bills)
            scaled = []
            for t in transactions:
                if t.transaction_type == "purchase":
                    new_base = round(t.base_amount * scale, -3)
                    new_gst = round(new_base * t.gst_rate, 2)
                    new_total = round(new_base + new_gst, 2)
                    scaled.append(Transaction(
                        id=t.id,
                        party_name=t.party_name,
                        transaction_type=t.transaction_type,
                        base_amount=new_base,
                        gst_rate=t.gst_rate,
                        gst_amount=new_gst,
                        total_amount=new_total,
                        hsn_code=t.hsn_code,
                        due_day=t.due_day,
                        urgency_score=t.urgency_score,
                        vendor_gstr1_reliability=t.vendor_gstr1_reliability,
                    ))
                else:
                    scaled.append(t)
            transactions = scaled

        # -----------------------------------------------------------------------
        # Baseline GST (pre-computed; agent cannot change this)
        # -----------------------------------------------------------------------
        baseline_gst = sum(
            t.gst_amount for t in transactions if t.transaction_type == "sale"
        )

        return {
            "opening_cash": opening_cash,
            "transactions": transactions,
            "baseline_gst": baseline_gst,
            "difficulty": difficulty,
            "episode_id": str(uuid.uuid4()),
            "episode_days": config["episode_days"],
        }
