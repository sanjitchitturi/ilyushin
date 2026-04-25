"""
reward_fn.py — Environment interaction utilities for Ilyushin PPO training.

With PPO the reward comes directly from the environment after each step.
There is no keyword-based reward function anymore.

This module provides:
  - parse_action()     : safely parse model text output into a valid action dict
  - IlyushinRewardFn   : thin wrapper around the REST environment for
                         baseline evaluation and any standalone testing

The actual training rewards are collected inline in train.py's
collect_ppo_episode() by calling env_step() directly.
"""

import os
import json
import requests

BASE_URL       = os.getenv("ENV_BASE_URL", os.getenv("BASE_URL", "http://localhost:8000"))
VALID_ACTIONS  = [
    "read_logs", "check_metrics", "restart_service",
    "scale_up", "rollback", "page_oncall", "resolve",
]
VALID_SERVICES = ["web_server", "database", "cache", "queue", "api_gateway"]


def parse_action(text: str) -> dict:
    """
    Parse model output text into a valid action dict.

    Handles:
      - Clean JSON:       {"type": "restart_service", "target_service": "database"}
      - Markdown fences:  ```json\\n{...}\\n```
      - JSON embedded in text: "I will restart... {"type": ...}"
      - Invalid action types → falls back to read_logs
      - Invalid services   → sets target_service to None

    Always returns a dict with both "type" and "target_service" keys,
    so the environment never receives an incomplete action.
    """
    try:
        # Strip markdown fences
        if "```" in text:
            text = "\n".join(
                line for line in text.split("\n")
                if not line.strip().startswith("```")
            ).strip()

        # Extract first JSON object in the text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        action = json.loads(text)

        # Validate action type
        if action.get("type") not in VALID_ACTIONS:
            action["type"] = "read_logs"

        # Validate target service
        target = action.get("target_service")
        if target is not None and target not in VALID_SERVICES:
            action["target_service"] = None

        # Ensure the key always exists
        if "target_service" not in action:
            action["target_service"] = None

        return action

    except Exception:
        return {"type": "read_logs", "target_service": None}


def get_state_dict(data: dict) -> dict:
    """Extract the state dict from an environment API response."""
    return data.get("state", data)


class IlyushinRewardFn:
    """
    Thin wrapper around the Ilyushin REST environment.

    Intended for:
      - Baseline evaluation (random agent)
      - Standalone testing / debugging
      - NOT used during PPO training (train.py calls the env directly)

    The reward returned by step() is the real environment reward
    (healthy_services delta + bonuses/penalties from env/reward.py).
    This is a dense, meaningful signal — NOT a keyword heuristic.
    """

    def __init__(self, task_id: str = "easy"):
        self.task_id    = task_id
        self.session_id = f"reward_fn-{task_id}-{id(self)}"
        self.state      = None
        self.step_count = 0

    def reset(self) -> dict:
        """Reset the environment and return the initial state dict."""
        res = requests.post(
            f"{BASE_URL}/env/reset",
            json={"session_id": self.session_id, "task_id": self.task_id},
            timeout=15,
        )
        res.raise_for_status()
        data        = res.json()
        self.state  = get_state_dict(data)
        self.step_count = 0
        return self.state

    def step(self, action_text_or_dict) -> tuple:
        """
        Step the environment.

        Args:
            action_text_or_dict: either a raw model output string (will be parsed)
                                 or an already-parsed action dict.

        Returns:
            (state dict, reward float, done bool)

        The reward is the real environment reward — healthy service delta,
        completion bonuses, oncall penalties, etc.
        """
        if isinstance(action_text_or_dict, str):
            action = parse_action(action_text_or_dict)
        else:
            action = action_text_or_dict
            # Ensure target_service key exists
            if "target_service" not in action:
                action["target_service"] = None

        res = requests.post(
            f"{BASE_URL}/env/step",
            json={"session_id": self.session_id, "action": action},
            timeout=15,
        )
        res.raise_for_status()
        data        = res.json()
        self.state  = get_state_dict(data)
        self.step_count += 1

        # Handle reward whether env returns float or {"value": float}
        reward = data.get("reward", -0.1)
        if isinstance(reward, dict):
            reward = reward.get("value", -0.1)

        done = bool(data.get("done", self.state.get("done", False)))

        return self.state, float(reward), done

    def get_success_rate(self) -> float:
        """Compute current success rate from state."""
        if self.state is None:
            return 0.0
        healthy = self.state.get("healthy_services", 0)
        total   = self.state.get("total_services", 5)
        return healthy / max(total, 1)

    def get_breaker_status(self) -> dict:
        """Fetch the Breaker's current status from the env server."""
        try:
            res = requests.get(f"{BASE_URL}/breaker/status", timeout=5)
            if res.ok:
                return res.json()
        except Exception:
            pass
        return {}

    def send_feedback(self, performance: dict) -> dict:
        """
        Send episode performance to the Breaker so it can adapt difficulty.

        Args:
            performance: dict with keys:
                success_rate, avg_steps, avg_reward,
                failed_on_incidents, succeeded_on_incidents, unhealthy_services
        """
        try:
            res = requests.post(
                f"{BASE_URL}/feedback",
                json={"session_id": self.session_id, **performance},
                timeout=10,
            )
            if res.ok:
                return res.json().get("breaker_status", {})
        except Exception:
            pass
        return {}

    def close(self):
        """No-op. Kept for interface compatibility."""
        pass