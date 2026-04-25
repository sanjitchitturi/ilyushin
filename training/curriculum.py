import random
from collections import deque


PHASES = ["easy", "medium", "hard"]

GRADUATION_THRESHOLD = {
    "easy":   0.88,
    "medium": 0.84,
}

GRADUATION_WINDOW    = 50
REPLAY_RATIO         = 0.10
BREAKER_ESCALATION_INTERVAL = 10


class CurriculumManager:
    def __init__(self):
        self.current_phase     = "easy"
        self.phase_index       = 0
        self.episode_count     = 0
        self.total_episodes    = 0
        self.reward_history    = {phase: deque(maxlen=GRADUATION_WINDOW) for phase in PHASES}
        self.phase_episode_count = {phase: 0 for phase in PHASES}
        self.graduated_phases  = []
        self.breaker_level     = 1

    def get_task_id(self) -> str:
        if self.graduated_phases and random.random() < REPLAY_RATIO:
            return random.choice(self.graduated_phases)
        return self.current_phase

    def record_reward(self, reward: float, task_id: str):
        self.reward_history[task_id].append(reward)
        self.phase_episode_count[task_id] += 1
        self.episode_count  += 1
        self.total_episodes += 1

        if task_id == self.current_phase:
            if self.episode_count % BREAKER_ESCALATION_INTERVAL == 0:
                self.breaker_level = min(self.breaker_level + 1, 10)

        self._check_graduation()

    def _check_graduation(self):
        if self.current_phase == "hard":
            return

        history = self.reward_history[self.current_phase]
        if len(history) < GRADUATION_WINDOW:
            return

        avg_reward = sum(history) / len(history)
        threshold  = GRADUATION_THRESHOLD.get(self.current_phase, 0.85)

        if avg_reward >= threshold:
            print(f"\n[CURRICULUM] Graduating from {self.current_phase} "
                  f"(avg_reward={avg_reward:.3f} >= {threshold})")
            self.graduated_phases.append(self.current_phase)
            self.phase_index   = min(self.phase_index + 1, len(PHASES) - 1)
            self.current_phase = PHASES[self.phase_index]
            self.episode_count = 0
            self.breaker_level = max(1, self.breaker_level - 3)
            print(f"[CURRICULUM] Now training on: {self.current_phase}")

    def get_breaker_difficulty(self) -> dict:
        return {
            "level":              self.breaker_level,
            "max_incidents":      min(self.breaker_level, 4),
            "use_red_herrings":   self.breaker_level >= 3,
            "use_cascades":       self.breaker_level >= 5,
            "use_overload":       self.breaker_level >= 4,
            "simultaneous":       self.breaker_level >= 6,
        }

    def get_responder_weak_spots(self) -> list:
        weak = []
        history = list(self.reward_history[self.current_phase])
        if not history:
            return weak
        recent = history[-20:] if len(history) >= 20 else history
        avg    = sum(recent) / len(recent)
        if avg < 0.5:
            weak.append("basic_diagnosis")
        if avg < 0.7:
            weak.append("cascade_handling")
        if avg < 0.85:
            weak.append("overload_detection")
        return weak

    def summary(self) -> dict:
        return {
            "current_phase":    self.current_phase,
            "breaker_level":    self.breaker_level,
            "total_episodes":   self.total_episodes,
            "phase_episodes":   dict(self.phase_episode_count),
            "graduated_phases": self.graduated_phases,
            "avg_rewards": {
                phase: round(sum(h) / len(h), 3) if h else 0.0
                for phase, h in self.reward_history.items()
            }
        }