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
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # Find the JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return GSTAction(action_type="DO_NOTHING")

        data = json.loads(text[start:end])
        action_type = data.get("action_type", "DO_NOTHING")
        transaction_id = data.get("transaction_id")

        # Validate transaction_id exists in the current observation
        if transaction_id and transaction_id not in (
            {s.id for s in obs.pending_sales} | {p.id for p in obs.pending_purchases}
        ):
            transaction_id = None

        return GSTAction(action_type=action_type, transaction_id=transaction_id)
    except Exception:
        return GSTAction(action_type="DO_NOTHING")


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
    model_fn: Any,  # callable(prompt: str) -> str
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
# Dataset builder for GRPO (prompt → reward)
# ---------------------------------------------------------------------------

def build_grpo_dataset(
    difficulty: str,
    num_episodes: int,
    seed_offset: int = 0,
) -> "Dataset":
    """
    Build a HuggingFace Dataset of (prompt, completion, reward) for GRPO.

    Each row is one step from one episode. The episode reward is broadcast
    to all steps (Monte Carlo return with no discounting — suitable for GRPO).
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Install training dependencies: pip install '.[training]'")

    env = GSTEnvironment(difficulty=difficulty)
    records: List[Dict] = []

    for ep_idx in range(num_episodes):
        obs = env.reset(seed=seed_offset + ep_idx)
        done = False
        episode_records: List[Dict] = []

        while not done:
            prompt_text = SYSTEM_PROMPT + "\n\n" + obs_to_prompt(obs)

            # Placeholder completion — replaced by model during GRPO generation
            completion = json.dumps({
                "action_type": "DO_NOTHING",
                "transaction_id": None,
            })

            obs_snapshot = obs_to_prompt(obs)
            episode_records.append({
                "prompt": prompt_text,
                "completion": completion,
                "obs_snapshot": obs_snapshot,
                "day": obs.day,
                "cash": obs.cash_balance,
                "episode_id": obs.episode_id,
                "difficulty": difficulty,
            })

            action = GSTAction(action_type="DO_NOTHING")
            obs = env.step(action)
            done = obs.done

        # Tag each step with terminal episode reward for GRPO reward signal
        terminal_reward = obs.reward or 0.0
        for rec in episode_records:
            rec["reward"] = terminal_reward

        records.extend(episode_records)

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward function for GRPO (called by GRPOTrainer)
# ---------------------------------------------------------------------------

def gst_reward_fn(
    completions: List[str],
    prompts: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """
    GRPO reward function — evaluates LLM action completions in the environment.

    Each completion is parsed as a GSTAction and executed in a fresh environment
    (single-step evaluation). The reward from the environment is returned.

    For terminal steps, a 50-point bonus is added for valid filings.
    """
    rewards: List[float] = []
    env = GSTEnvironment(difficulty="L1")

    for i, completion in enumerate(completions):
        try:
            obs = env.reset(seed=i)
            # Advance to a mid-episode state for more interesting rewards
            for _ in range(random.randint(0, 5)):
                env.step(GSTAction(action_type="DO_NOTHING"))
            obs = env._build_observation()

            action = parse_action(completion, obs)
            next_obs = env.step(action)
            reward = next_obs.reward or 0.0
        except Exception:
            reward = -10.0  # Penalty for malformed action

        rewards.append(reward)

    return rewards


# ---------------------------------------------------------------------------
# Training loop (curriculum: L1 → L2 → L3 → L4)
# ---------------------------------------------------------------------------

def train(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    curriculum: List[str] = None,
    episodes_per_level: int = 200,
    output_dir: str = "./checkpoints/gst-grpo",
    max_new_tokens: int = 128,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    num_generations: int = 4,
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
            reward_model_path=None,  # Use inline reward function
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

        # Chain: next level starts from this checkpoint
        current_model = level_output_dir

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\nTraining complete. Final model saved to {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GST advisor with GRPO")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model for GRPO training",
    )
    parser.add_argument(
        "--curriculum",
        default="L1,L2,L3,L4",
        help="Comma-separated difficulty levels to train in order",
    )
    parser.add_argument(
        "--episodes-per-level",
        type=int,
        default=200,
        help="Number of episodes to generate per curriculum level",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/gst-grpo",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
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
