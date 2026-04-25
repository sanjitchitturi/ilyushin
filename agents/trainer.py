import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = "meta-llama/Llama-3.3-70B-Instruct"  # hardcoded — always 70B regardless of MODEL_NAME env

SYSTEM_PROMPT = """You are a Trainer Agent in a production incident response training system.
Your job is to analyze completed episodes, identify Responder weaknesses, and generate
coaching instructions to improve performance.

You must return ONLY a JSON object with exactly these fields:
{
    "performance_rating": "poor" or "fair" or "good" or "excellent",
    "key_mistakes": ["mistake1", "mistake2"],
    "key_successes": ["success1", "success2"],
    "coaching_notes": "specific advice for improvement",
    "recommended_difficulty": "easier" or "same" or "harder",
    "focus_areas": ["area1", "area2"]
}

Return raw JSON only. No explanation. No markdown."""


class TrainerAgent:
    """
    Trainer Agent — always uses Llama-3.3-70B-Instruct.
    Analyzes completed episodes and identifies Responder weaknesses.
    """

    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.episode_logs        = []
        self.performance_history = []

    def analyze_episode(self, episode: dict) -> dict:
        prompt = f"""Analyze this completed incident response episode:

TASK: {episode.get('task_id', 'unknown')}
TOTAL STEPS: {episode.get('total_steps', 0)}
TOTAL REWARD: {episode.get('total_reward', 0)}
FINAL SCORE: {episode.get('final_score', 0)}
ONCALL PAGED: {episode.get('oncall_paged', False)}
DONE: {episode.get('done', False)}
HEALTHY SERVICES: {episode.get('healthy_services', 0)}/{episode.get('total_services', 5)}

ACTIONS TAKEN:
{json.dumps(episode.get('actions_taken', []), indent=2)}

REWARD BREAKDOWN PER STEP:
{json.dumps(episode.get('reward_breakdown', []), indent=2)}

Identify what went wrong, what went right, and how to improve."""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = "\n".join(
                    l for l in raw.split("\n")
                    if not l.strip().startswith("```")
                ).strip()
            analysis = json.loads(raw)
            self.episode_logs.append(episode)
            self.performance_history.append(analysis)
            return analysis
        except Exception as e:
            fallback = {
                "performance_rating":    "fair",
                "key_mistakes":          [],
                "key_successes":         [],
                "coaching_notes":        f"Trainer error: {str(e)}",
                "recommended_difficulty": "same",
                "focus_areas":           [],
            }
            self.performance_history.append(fallback)
            return fallback

    def get_responder_performance(self) -> dict:
        if not self.episode_logs:
            return {}

        total_steps  = [e.get("total_steps", 0) for e in self.episode_logs]
        scores       = [e.get("final_score", 0)  for e in self.episode_logs]
        oncall_count = sum(1 for e in self.episode_logs if e.get("oncall_paged", False))

        all_actions = []
        for e in self.episode_logs:
            all_actions.extend(e.get("actions_taken", []))

        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        common_actions = sorted(action_counts, key=action_counts.get, reverse=True)[:3]

        hardest = [
            e.get("task_id", "unknown")
            for e in self.episode_logs
            if e.get("final_score", 0) < 0.5
        ]

        return {
            "avg_steps":          round(sum(total_steps) / len(total_steps), 2),
            "success_rate":       round(sum(1 for s in scores if s == 1.0) / len(scores), 2),
            "avg_score":          round(sum(scores) / len(scores), 2),
            "oncall_rate":        round(oncall_count / len(self.episode_logs), 2),
            "common_actions":     common_actions,
            "hardest_incidents":  list(set(hardest)),
            "total_episodes":     len(self.episode_logs),
            "model":              MODEL_NAME,
        }

    def reset(self):
        self.episode_logs        = []
        self.performance_history = []
