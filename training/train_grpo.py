"""
GRPO Training Script — GST Cash Flow Optimization Environment.

Trains Qwen2.5-3B-Instruct using TRL's GRPOTrainer to act as a GST advisor
for Indian SMEs. The agent learns to sequence vendor payments and sales
fulfillments to legally minimize net GST payable within a 30-day filing cycle.

Usage:
    python training/train_grpo.py
    python training/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --curriculum L1,L2,L3,L4
    python training/train_grpo.py --episodes-per-level 500 --output-dir ./checkpoints

Curriculum:
    L1: ₹8L cash, 3 sales, 2 vendors, 10 days — learning basics
    L2: ₹4L cash, 6 sales, 4 vendors, 15 days — moderate constraint
    L3: ₹2L cash, 10 sales, 6 vendors, 20 days — tight cash
    L4: ₹50K cash, 15 sales, 8 vendors, 30 days — full complexity (cash crisis)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy imports — so the script can be imported in CPU-only environments
# ---------------------------------------------------------------------------
try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False


# ---------------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gst_cashflow_env.models import GSTAction, GSTObservation
from server.gst_environment import GSTEnvironment


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a GST optimization advisor for an Indian apparel SME (Smart Choice India).
Your goal: minimize net GST payable = GST_collected - ITC_claimed.

ITC rules:
- ITC requires FULL vendor payment (all-or-nothing per invoice)
- ITC only claimed if vendor filed their own GSTR-1 (stochastic — shown as reliability)
- Must pay vendors BEFORE filing GSTR-3B (the 20th of the month)
- Paying a vendor after filing means ITC carries to NEXT month

Strategy principles:
1. Fulfill sales first to generate cash (cash only arrives on sale fulfillment)
2. Pay reliable vendors early to maximize expected ITC
3. Avoid the ₹50,000 cash floor (salary/operating expenses)
4. File GSTR-3B on or before Day 20 to avoid late penalties

Respond with EXACTLY one JSON object on a single line — no explanation, no markdown:
{"action_type": "<ACTION>", "transaction_id": "<ID_or_null>"}

Valid action_types: FULFILL_SALE, PAY_VENDOR, DEFER_SALE, DEFER_VENDOR, FILE_GSTR3B, DO_NOTHING
transaction_id is required for FULFILL_SALE, PAY_VENDOR, DEFER_SALE, DEFER_VENDOR.
"""


def obs_to_prompt(obs: GSTObservation) -> str:
    """Convert GSTObservation to a structured LLM prompt."""
    lines = [
        f"=== Day {obs.day} / {obs.days_to_filing} days to filing deadline ===",
        f"Cash balance: ₹{obs.cash_balance:,.0f}",
        f"GST collected so far: ₹{obs.gst_collected_so_far:,.0f}",
        f"ITC secured so far: ₹{obs.itc_secured_so_far:,.0f}",
        f"Net GST if filed now: ₹{obs.net_gst_if_filed_now:,.0f}",
        f"Baseline GST (no ITC): ₹{obs.baseline_gst:,.0f}",
        "",
    ]

    if obs.pending_sales:
        lines.append("PENDING SALES (fulfill to receive cash + collect GST):")
        for s in obs.pending_sales:
            lines.append(
                f"  [{s.id}] {s.party_name} — ₹{s.total_amount:,.0f} "
                f"(base ₹{s.base_amount:,.0f} + GST ₹{s.gst_amount:,.0f} @ {s.gst_rate*100:.0f}%) "
                f"urgency={s.urgency_score:.2f} hsn={s.hsn_code}"
            )
    else:
        lines.append("PENDING SALES: none")

    lines.append("")

    if obs.pending_purchases:
        lines.append("PENDING VENDOR INVOICES (pay to secure ITC — all-or-nothing):")
        for p in obs.pending_purchases:
            expected_itc = p.gst_amount * p.vendor_gstr1_reliability
            lines.append(
                f"  [{p.id}] {p.party_name} — ₹{p.total_amount:,.0f} "
                f"(base ₹{p.base_amount:,.0f} + GST ₹{p.gst_amount:,.0f} @ {p.gst_rate*100:.0f}%) "
                f"reliability={p.vendor_gstr1_reliability:.2f} "
                f"expected_ITC=₹{expected_itc:,.0f}"
            )
    else:
        lines.append("PENDING VENDOR INVOICES: none")

    lines.append("")

    if obs.vendor_scores:
        scores = ", ".join(f"{k}: {v:.2f}" for k, v in obs.vendor_scores.items())
        lines.append(f"Vendor relationship scores: {scores}")
    if obs.retailer_scores:
        scores = ", ".join(f"{k}: {v:.2f}" for k, v in obs.retailer_scores.items())
        lines.append(f"Retailer relationship scores: {scores}")

    return "\n".join(lines)


def parse_action(response: str, obs: GSTObservation) -> GSTAction:
    """
    Parse LLM response string into GSTAction.

    Falls back to DO_NOTHING if parsing fails — never crashes the training loop.
    """
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return GSTAction(action_type="DO_NOTHING")

        data = json.loads(text[start:end])
        action_type = data.get("action_type", "DO_NOTHING")
        transaction_id = data.get("transaction_id")

        if transaction_id and transaction_id not in (
            {s.id for s in obs.pending_sales} | {p.id for p in obs.pending_purchases}
        ):
            transaction_id = None

        return GSTAction(action_type=action_type, transaction_id=transaction_id)
    except Exception:
        return GSTAction(action_type="DO_NOTHING")


# ---------------------------------------------------------------------------
# Analytical reward scorer — the core of GRPO signal
# ---------------------------------------------------------------------------

def _parse_prompt_state(prompt: str) -> Dict:
    """
    Extract key state variables from obs_to_prompt() text output.

    Parses cash, day, pending sales/purchases so the reward function
    can score action quality WITHOUT re-running the environment.
    This is how we get real reward variance across GRPO group completions.
    """
    state: Dict = {
        "day": 1,
        "days_to_filing": 20,
        "cash": 800_000.0,
        "gst_collected": 0.0,
        "itc_secured": 0.0,
        "net_gst_if_filed_now": 0.0,
        "baseline_gst": 0.0,
        "pending_sales": [],
        "pending_purchases": [],
    }

    m = re.search(r"=== Day (\d+)\s*/\s*(\d+) days to filing", prompt)
    if m:
        state["day"] = int(m.group(1))
        state["days_to_filing"] = int(m.group(2))

    m = re.search(r"Cash balance: ₹([\d,]+)", prompt)
    if m:
        state["cash"] = float(m.group(1).replace(",", ""))

    m = re.search(r"GST collected so far: ₹([\d,]+)", prompt)
    if m:
        state["gst_collected"] = float(m.group(1).replace(",", ""))

    m = re.search(r"ITC secured so far: ₹([\d,]+)", prompt)
    if m:
        state["itc_secured"] = float(m.group(1).replace(",", ""))

    m = re.search(r"Net GST if filed now: ₹([\d,]+)", prompt)
    if m:
        state["net_gst_if_filed_now"] = float(m.group(1).replace(",", ""))

    m = re.search(r"Baseline GST \(no ITC\): ₹([\d,]+)", prompt)
    if m:
        state["baseline_gst"] = float(m.group(1).replace(",", ""))

    # Pending purchases: "[ID] Party — ₹total (base ₹X + GST ₹Y @ Z%) reliability=R"
    for m in re.finditer(
        r"\[(\S+)\]\s+\S.+?—\s+₹([\d,]+)\s+\(base.+?GST ₹([\d,]+).+?reliability=([\d.]+)",
        prompt,
    ):
        state["pending_purchases"].append({
            "id": m.group(1),
            "total": float(m.group(2).replace(",", "")),
            "gst": float(m.group(3).replace(",", "")),
            "reliability": float(m.group(4)),
        })

    # Pending sales: "[ID] Party — ₹total (base ₹X + GST ₹Y @ Z%) urgency=U"
    for m in re.finditer(
        r"\[(\S+)\]\s+\S.+?—\s+₹([\d,]+)\s+\(base.+?GST ₹([\d,]+).+?urgency=([\d.]+)",
        prompt,
    ):
        state["pending_sales"].append({
            "id": m.group(1),
            "total": float(m.group(2).replace(",", "")),
            "gst": float(m.group(3).replace(",", "")),
            "urgency": float(m.group(4)),
        })

    return state


def _score_completion(completion: str, prompt: str) -> float:
    """
    Score one LLM completion against the current GST state in the prompt.

    Economic scoring rules — each action type is scored based on domain
    principles from the GST environment design:

    PAY_VENDOR:
      - Bad if unaffordable (would fail in env anyway)
      - Scored by expected_ITC = gst × reliability
      - Scored by ITC efficiency = expected_ITC / cash_spent
      - Bonus when deadline is close (ITC opportunity shrinking)

    FULFILL_SALE:
      - Good: generates cash needed for vendor payments
      - Bonus when cash is low (unlocks more vendor payments)
      - Bonus for high urgency (relationship preservation)

    FILE_GSTR3B:
      - Heavily penalized before Day 8 (environment blocks it anyway)
      - Penalized when significant ITC is still unclaimed and time remains
      - Rewarded when all high-reliability vendors are paid and deadline near

    DO_NOTHING:
      - Penalized whenever affordable actions exist
      - Acceptable only when cash insufficient for any vendor AND no sales pending

    Returns a float reward. Real variance across the 8 GRPO completions is
    what drives non-zero advantage → non-zero loss → actual learning.
    """
    # --- PARSE ACTION JSON ---
    action_type = "DO_NOTHING"
    transaction_id = None
    try:
        text = completion.strip()
        # Strip markdown fences
        if "```" in text:
            for chunk in text.split("```"):
                if "{" in chunk:
                    text = chunk.lstrip("json").strip()
                    break
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= 0:
            return -10.0  # No JSON found — model didn't follow the format
        data = json.loads(text[start:end])
        action_type = data.get("action_type", "DO_NOTHING")
        transaction_id = data.get("transaction_id")
    except (json.JSONDecodeError, Exception):
        return -10.0  # Broken JSON

    valid_actions = {
        "FULFILL_SALE", "PAY_VENDOR", "DEFER_SALE",
        "DEFER_VENDOR", "FILE_GSTR3B", "DO_NOTHING",
    }
    if action_type not in valid_actions:
        return -10.0  # Hallucinated action name

    # --- PARSE STATE FROM PROMPT ---
    state = _parse_prompt_state(prompt)
    day = state["day"]
    days_to_filing = state["days_to_filing"]
    cash = state["cash"]
    pending_sales = state["pending_sales"]
    pending_purchases = state["pending_purchases"]

    # --- SCORE BY ACTION TYPE ---

    if action_type == "DO_NOTHING":
        can_pay = any(p["total"] <= cash for p in pending_purchases)
        can_fulfill = len(pending_sales) > 0
        if can_pay or can_fulfill:
            # Something is available but agent chose to idle — bad
            penalty = -5.0
            # Escalate near deadline: idle with payable vendors is very bad
            if days_to_filing <= 5 and can_pay:
                penalty -= 5.0
            return penalty
        # Genuinely nothing available (insufficient cash, no sales) — acceptable
        return -0.5

    elif action_type == "FILE_GSTR3B":
        if day < 8:
            return -20.0  # Blocked by env anti-hack rule; agent should know this

        # Calculate ITC being abandoned (paid vendors that still haven't been paid)
        total_expected_itc_remaining = sum(
            p["gst"] * p["reliability"] for p in pending_purchases
        )
        # Penalize filing with significant ITC still on the table and time remaining
        if total_expected_itc_remaining > 3_000 and days_to_filing > 2:
            itc_loss_penalty = min(total_expected_itc_remaining / 2_000, 15.0)
            return -itc_loss_penalty

        # Also penalize filing when sales are still unfulfilled and time remains
        if pending_sales and days_to_filing > 3:
            return -6.0

        # Appropriate filing: no pending ITC worth chasing, deadline near
        if days_to_filing <= 2:
            return 20.0   # Perfect timing
        if days_to_filing <= 5:
            return 12.0   # Good timing
        return 3.0         # Early filing, nothing pending — acceptable

    elif action_type == "FULFILL_SALE":
        if not transaction_id:
            return -8.0  # transaction_id is required for FULFILL_SALE
        sale = next((s for s in pending_sales if s["id"] == transaction_id), None)
        if not sale:
            return -8.0  # Hallucinated transaction ID

        reward = 3.0                          # Base: any sale fulfillment is good
        reward += sale["urgency"] * 4.0       # High urgency = relationship at risk

        # Bonus when cash-poor: this sale unlocks vendor payments
        if cash < 300_000:
            reward += 4.0
        # Extra bonus if this sale's cash flow specifically enables a vendor payment
        if pending_purchases:
            min_vendor_cost = min(p["total"] for p in pending_purchases)
            if cash < min_vendor_cost <= cash + sale["total"]:
                reward += 5.0  # This fulfillment directly unlocks best vendor

        return min(reward, 20.0)

    elif action_type == "PAY_VENDOR":
        if not transaction_id:
            return -8.0  # transaction_id is required for PAY_VENDOR
        vendor = next((p for p in pending_purchases if p["id"] == transaction_id), None)
        if not vendor:
            return -8.0  # Hallucinated transaction ID
        if vendor["total"] > cash:
            return -15.0  # Cannot afford — this action would fail in the env

        # Core ITC economics
        expected_itc = vendor["gst"] * vendor["reliability"]
        # ITC efficiency: how much expected ITC per rupee of cash spent
        itc_efficiency = expected_itc / max(1.0, vendor["total"])

        reward = 5.0                               # Base: paying vendors is always progress
        reward += expected_itc / 1_000.0           # Absolute ITC value (₹1K = +1)
        reward += itc_efficiency * 10_000.0        # Efficiency premium

        # Deadline urgency: paying vendors near deadline is more valuable
        # (each day lost = fewer chances to secure ITC before the 20th)
        if days_to_filing <= 3:
            reward += 10.0
        elif days_to_filing <= 7:
            reward += 5.0

        # Small penalty for paying very low-reliability vendors when better options exist
        best_reliability = max(
            (p["reliability"] for p in pending_purchases if p["total"] <= cash),
            default=0.0,
        )
        if vendor["reliability"] < best_reliability - 0.3:
            reward -= 3.0  # Chose a less reliable vendor over a much better one

        return min(reward, 35.0)

    elif action_type in ("DEFER_SALE", "DEFER_VENDOR"):
        # Explicit deferral penalizes relationships; only justified in edge cases
        if not transaction_id:
            return -3.0
        return -4.0

    return -1.0  # Unknown / catch-all


def gst_reward_fn(
    completions: List[str],
    prompts: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """
    GRPO reward function — analytically scores LLM action completions.

    WHY ANALYTICAL (not env-step):
      GRPO groups N completions (e.g., 8) for the SAME prompt and computes
      group-relative advantage = reward_i - mean(rewards). If all completions
      produce the same step reward (e.g., all output DO_NOTHING → all get -2.5),
      advantage = 0 → loss = 0 → no learning. This is the "zero loss" failure.

      The analytical scorer gives each action type a DIFFERENT reward based on
      the economic principles of the GST problem parsed from the prompt text.
      This guarantees reward variance within every GRPO group.

    REWARD RANGE:
      -20 (file too early / unaffordable action) to +35 (pay vendor near deadline)
      DO_NOTHING: -10 to -0.5
      Malformed JSON: -10
    """
    rewards: List[float] = []

    for i, completion in enumerate(completions):
        prompt = (prompts[i] if prompts and i < len(prompts) else "") or ""
        try:
            reward = _score_completion(completion, prompt)
        except Exception:
            reward = -10.0
        rewards.append(reward)

    return rewards


# ---------------------------------------------------------------------------
# Dataset builder for GRPO — diverse episode states
# ---------------------------------------------------------------------------

def _greedy_action(obs: GSTObservation) -> GSTAction:
    """Simple greedy policy used to generate realistic mid-episode states."""
    # Fulfill any pending sale first (generate cash)
    if obs.pending_sales:
        return GSTAction(action_type="FULFILL_SALE", transaction_id=obs.pending_sales[0].id)
    # Pay vendor with highest expected ITC that we can afford
    affordable = [
        p for p in obs.pending_purchases if obs.cash_balance >= p.total_amount
    ]
    if affordable:
        best = max(affordable, key=lambda p: p.gst_amount * p.vendor_gstr1_reliability)
        return GSTAction(action_type="PAY_VENDOR", transaction_id=best.id)
    # File if eligible and nothing else to do
    if obs.day >= 8:
        return GSTAction(action_type="FILE_GSTR3B")
    return GSTAction(action_type="DO_NOTHING")


def build_grpo_dataset(
    difficulty: str,
    num_episodes: int,
    seed_offset: int = 0,
) -> "Dataset":
    """
    Build a HuggingFace Dataset of (prompt, completion, reward) for GRPO.

    IMPORTANT IMPROVEMENT over naive DO_NOTHING rollout:
    Each episode is advanced using a MIXED policy (greedy + random) for the
    first half of steps, then captures snapshots. This produces diverse
    mid-episode states (cash partially spent, some vendors paid, deadline closer)
    that are far more informative training examples than always-Day-1 states.

    Each row is one step snapshot from one episode. The analytical reward
    function (gst_reward_fn) evaluates completions against the state encoded
    in the prompt text — so the prompt must be self-contained.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Install training dependencies: pip install '.[training]'")

    env = GSTEnvironment(difficulty=difficulty)
    records: List[Dict] = []

    for ep_idx in range(num_episodes):
        rng = random.Random(seed_offset + ep_idx)
        obs = env.reset(seed=seed_offset + ep_idx)
        done = False

        # Advance episode with mixed policy to get diverse states
        # 50% chance each step: greedy action vs DO_NOTHING
        # This ensures some vendors are paid, some sales fulfilled — realistic variety
        max_advance = obs.days_to_filing // 2  # Advance up to half the filing window
        advance_steps = rng.randint(0, max(0, max_advance))

        for _ in range(advance_steps):
            if done:
                break
            if rng.random() < 0.7:
                action = _greedy_action(obs)
            else:
                action = GSTAction(action_type="DO_NOTHING")
            obs = env.step(action)
            done = obs.done

        if done:
            # Episode ended during advance — restart cleanly
            obs = env.reset(seed=seed_offset + ep_idx)
            done = False

        # Capture snapshots from this point forward (up to episode end)
        episode_records: List[Dict] = []
        while not done:
            prompt_text = SYSTEM_PROMPT + "\n\n" + obs_to_prompt(obs)

            # Placeholder completion — replaced by model during GRPO generation
            placeholder = json.dumps({
                "action_type": "DO_NOTHING",
                "transaction_id": None,
            })

            episode_records.append({
                "prompt": prompt_text,
                "completion": placeholder,
                "obs_snapshot": obs_to_prompt(obs),
                "day": obs.day,
                "cash": obs.cash_balance,
                "episode_id": obs.episode_id,
                "difficulty": difficulty,
            })

            # Advance with greedy to get the next snapshot
            action = _greedy_action(obs)
            obs = env.step(action)
            done = obs.done

        # Tag each step with a base reward for context (actual rewards computed
        # analytically in gst_reward_fn based on the action the model outputs)
        terminal_reward = obs.reward or 0.0
        for rec in episode_records:
            rec["reward"] = terminal_reward

        records.extend(episode_records)

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Episode rollout (for reward signal)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStep:
    prompt: str
    response: str
    reward: float


def rollout_episode(
    env: GSTEnvironment,
    model_fn: Any,
    difficulty: str,
    seed: int,
) -> List[EpisodeStep]:
    """Run one episode and return (prompt, response, reward) triples for GRPO."""
    obs = env.reset(seed=seed)
    steps: List[EpisodeStep] = []
    done = False

    while not done:
        prompt = obs_to_prompt(obs)
        response = model_fn(SYSTEM_PROMPT + "\n\n" + prompt)
        action = parse_action(response, obs)

        obs = env.step(action)
        done = obs.done
        reward = obs.reward or 0.0

        steps.append(EpisodeStep(prompt=prompt, response=response, reward=reward))

    return steps


# ---------------------------------------------------------------------------
# Training loop (curriculum: L1 → L2 → L3 → L4)
# ---------------------------------------------------------------------------

def train(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    curriculum: List[str] = None,
    episodes_per_level: int = 200,
    output_dir: str = "./checkpoints/gst-grpo",
    max_new_tokens: int = 128,
    learning_rate: float = 2e-6,
    batch_size: int = 4,
    num_generations: int = 8,
) -> None:
    """
    Train Qwen2.5 on the GST environment using GRPO with curriculum learning.

    Curriculum strategy: start at L1 (cash-rich, simple) and progress to L4
    (cash-crisis, full complexity). Each level refines the policy learned at
    the previous level — stable curriculum prevents early catastrophic forgetting.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Install training dependencies: pip install '.[training]'")

    if curriculum is None:
        curriculum = ["L1", "L2", "L3", "L4"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    current_model = model_name

    for level_idx, difficulty in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"Curriculum Stage {level_idx + 1}/{len(curriculum)}: {difficulty}")
        print(f"{'='*60}")

        dataset = build_grpo_dataset(
            difficulty=difficulty,
            num_episodes=episodes_per_level,
            seed_offset=level_idx * 10000,
        )

        level_output_dir = os.path.join(output_dir, f"level_{difficulty}")

        config = GRPOConfig(
            output_dir=level_output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            max_prompt_length=1024,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_accumulation_steps=max(1, 16 // batch_size),
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
        )

        trainer = GRPOTrainer(
            model=current_model,
            args=config,
            train_dataset=dataset,
            reward_funcs=gst_reward_fn,
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(level_output_dir)
        print(f"Saved {difficulty} checkpoint to {level_output_dir}")

        current_model = level_output_dir

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\nTraining complete. Final model saved to {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GST advisor with GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--curriculum", default="L1,L2,L3,L4")
    parser.add_argument("--episodes-per-level", type=int, default=200)
    parser.add_argument("--output-dir", default="./checkpoints/gst-grpo")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train(
        model_name=args.model,
        curriculum=args.curriculum.split(","),
        episodes_per_level=args.episodes_per_level,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
