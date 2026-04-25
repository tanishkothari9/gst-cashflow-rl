"""
Evaluation Script — GST Cash Flow Optimization Environment.

Compares trained LLM agent against two baselines:
  - Random agent: chooses a valid random action each step
  - Greedy agent: fulfills sales, then pays most-reliable vendor, files on Day 8+

Usage:
    python training/evaluate.py --model ./checkpoints/gst-grpo/final --episodes 100
    python training/evaluate.py --baseline-only --difficulty L4 --episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gst_cashflow_env.models import GSTAction, GSTObservation
from server.gst_environment import GSTEnvironment
from training.train_grpo import SYSTEM_PROMPT, obs_to_prompt, parse_action


# ---------------------------------------------------------------------------
# Baseline agents
# ---------------------------------------------------------------------------

def random_agent(obs: GSTObservation) -> GSTAction:
    """Uniformly random valid action."""
    choices: List[GSTAction] = [GSTAction(action_type="DO_NOTHING")]

    for s in obs.pending_sales:
        choices.append(GSTAction(action_type="FULFILL_SALE", transaction_id=s.id))
    for p in obs.pending_purchases:
        if obs.cash_balance >= p.total_amount:
            choices.append(GSTAction(action_type="PAY_VENDOR", transaction_id=p.id))

    if obs.day >= 8:
        choices.append(GSTAction(action_type="FILE_GSTR3B"))

    return random.choice(choices)


def greedy_agent(obs: GSTObservation) -> GSTAction:
    """
    Greedy heuristic:
      1. Fulfill any pending sale (generate cash first)
      2. Pay the vendor with highest expected ITC (reliability × gst_amount)
         if cash is sufficient
      3. File on or after Day 8 when no pending sales remain
      4. DO_NOTHING otherwise
    """
    # Always fulfill pending sales to generate cash
    if obs.pending_sales:
        return GSTAction(action_type="FULFILL_SALE", transaction_id=obs.pending_sales[0].id)

    # Pay vendor with highest expected ITC that we can afford
    affordable_purchases = [
        p for p in obs.pending_purchases
        if obs.cash_balance >= p.total_amount
    ]
    if affordable_purchases:
        best = max(
            affordable_purchases,
            key=lambda p: p.gst_amount * p.vendor_gstr1_reliability,
        )
        return GSTAction(action_type="PAY_VENDOR", transaction_id=best.id)

    # File when eligible
    if obs.day >= 8:
        return GSTAction(action_type="FILE_GSTR3B")

    return GSTAction(action_type="DO_NOTHING")


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode_id: str
    difficulty: str
    total_reward: float
    net_gst_paid: float
    itc_utilization_pct: float
    filing_day: Optional[int]
    final_cash: float
    steps: int


def run_episode(
    env: GSTEnvironment,
    agent_fn: Callable[[GSTObservation], GSTAction],
    difficulty: str,
    seed: int,
    max_steps: int = 60,
) -> EpisodeResult:
    """Run one episode with the given agent and return key metrics."""
    obs = env.reset(seed=seed)
    done = False
    step_count = 0
    cumulative_reward = 0.0

    while not done and step_count < max_steps:
        action = agent_fn(obs)
        obs = env.step(action)
        done = obs.done
        cumulative_reward += obs.reward or 0.0
        step_count += 1

    gstr3b = obs.metadata.get("gstr3b", {}) if obs.metadata else {}
    return EpisodeResult(
        episode_id=obs.episode_id,
        difficulty=difficulty,
        total_reward=cumulative_reward,
        net_gst_paid=gstr3b.get("net_payable", 0.0),
        itc_utilization_pct=gstr3b.get("itc_utilization_pct", 0.0),
        filing_day=obs.metadata.get("filing_day") if obs.metadata else None,
        final_cash=obs.cash_balance,
        steps=step_count,
    )


# ---------------------------------------------------------------------------
# LLM agent wrapper
# ---------------------------------------------------------------------------

def build_llm_agent(model_path: str) -> Callable[[GSTObservation], GSTAction]:
    """Build an agent function from a trained checkpoint."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Install transformers and torch: pip install '.[training]'")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    def agent_fn(obs: GSTObservation) -> GSTAction:
        prompt = SYSTEM_PROMPT + "\n\n" + obs_to_prompt(obs)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return parse_action(response, obs)

    return agent_fn


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    agents: Dict[str, Callable],
    difficulty: str,
    num_episodes: int,
    seed_offset: int = 0,
) -> Dict[str, List[EpisodeResult]]:
    """Evaluate all agents on the same set of seeded episodes."""
    results: Dict[str, List[EpisodeResult]] = {name: [] for name in agents}
    env = GSTEnvironment(difficulty=difficulty)

    for ep_idx in range(num_episodes):
        seed = seed_offset + ep_idx
        for name, agent_fn in agents.items():
            result = run_episode(env, agent_fn, difficulty, seed=seed)
            results[name].append(result)

        if (ep_idx + 1) % 10 == 0:
            print(f"  Completed {ep_idx + 1}/{num_episodes} episodes")

    return results


def print_summary(results: Dict[str, List[EpisodeResult]], difficulty: str) -> None:
    """Print comparative evaluation summary."""
    print(f"\n{'='*70}")
    print(f"Evaluation Results — Difficulty {difficulty}")
    print(f"{'='*70}")
    print(f"{'Agent':<20} {'Reward':>10} {'Net GST':>12} {'ITC%':>8} {'Cash':>12}")
    print(f"{'-'*70}")

    for name, eps in results.items():
        avg_reward = sum(e.total_reward for e in eps) / len(eps)
        avg_gst = sum(e.net_gst_paid for e in eps) / len(eps)
        avg_itc = sum(e.itc_utilization_pct for e in eps) / len(eps) * 100
        avg_cash = sum(e.final_cash for e in eps) / len(eps)
        print(
            f"{name:<20} {avg_reward:>10.1f} ₹{avg_gst:>10,.0f} "
            f"{avg_itc:>7.1f}% ₹{avg_cash:>10,.0f}"
        )

    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GST advisor agents")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained model checkpoint (optional — skipped if not provided)",
    )
    parser.add_argument(
        "--difficulty",
        default="L1,L2,L3,L4",
        help="Comma-separated difficulty levels to evaluate on",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per difficulty level",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=9000,
        help="Seed offset (use different from training seeds to avoid data leakage)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate baseline agents (no LLM required)",
    )
    args = parser.parse_args()

    agents: Dict[str, Callable] = {
        "Random": random_agent,
        "Greedy": greedy_agent,
    }

    if not args.baseline_only and args.model:
        print(f"Loading model from {args.model}...")
        agents["LLM (GRPO)"] = build_llm_agent(args.model)

    difficulties = args.difficulty.split(",")

    for difficulty in difficulties:
        print(f"\nEvaluating on difficulty {difficulty} ({args.episodes} episodes)...")
        results = evaluate(
            agents=agents,
            difficulty=difficulty,
            num_episodes=args.episodes,
            seed_offset=args.seed_offset,
        )
        print_summary(results, difficulty)

        # Save raw results to JSON
        output_path = f"evaluation_{difficulty}.json"
        serializable = {
            name: [
                {
                    "episode_id": e.episode_id,
                    "total_reward": e.total_reward,
                    "net_gst_paid": e.net_gst_paid,
                    "itc_utilization_pct": e.itc_utilization_pct,
                    "filing_day": e.filing_day,
                    "final_cash": e.final_cash,
                    "steps": e.steps,
                }
                for e in eps
            ]
            for name, eps in results.items()
        }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Raw results saved to {output_path}")


if __name__ == "__main__":
    main()
