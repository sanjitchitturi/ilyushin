import os
import json
import random
from openai import OpenAI
from world.infrastructure import Infrastructure, SERVICE_NAMES
from world.incident_generator import IncidentGenerator, IncidentType, Severity

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = "meta-llama/Llama-3.3-70B-Instruct"  # hardcoded — always 70B regardless of MODEL_NAME env

SYSTEM_PROMPT = """You are a Breaker Agent in an adversarial training system.
Your job is to inject realistic production failures that will challenge the Responder Agent.

You get rewarded when the Responder fails to resolve incidents quickly.
You MUST learn from what worked and escalate your attacks.

You must return ONLY a JSON object with exactly these fields:
{
    "incident_type": "<type>",
    "target_services": ["service1", "service2"],
    "severity": "p1" or "p2" or "p3",
    "reasoning": "why this failure pattern will be hard to resolve"
}

Valid incident types:
- cpu_spike
- memory_leak
- service_crash
- high_latency
- cascade_failure
- red_herring

Valid services: web_server, database, cache, queue, api_gateway

CRITICAL RULES:
1. Analyze Responder's performance history
2. Target the services they struggle with most
3. Repeat the incident types that worked best before
4. Combine multiple incidents for cascading effects
5. Use red_herring to confuse the Responder
6. As Responder gets better, escalate: more incidents, harder combinations
7. Track what breaks them and use it again

Return raw JSON only. No explanation. No markdown."""

INCIDENT_TYPE_MAP = {
    "cpu_spike":       IncidentType.CPU_SPIKE,
    "memory_leak":     IncidentType.MEMORY_LEAK,
    "service_crash":   IncidentType.SERVICE_CRASH,
    "high_latency":    IncidentType.HIGH_LATENCY,
    "cascade_failure": IncidentType.CASCADE_FAILURE,
    "red_herring":     IncidentType.RED_HERRING,
}

SEVERITY_MAP = {
    "p1": Severity.P1,
    "p2": Severity.P2,
    "p3": Severity.P3,
}


class BreakerAgent:
    """
    Adaptive Breaker Agent that learns from Responder performance.
    Always uses Llama-3.3-70B-Instruct to generate strategic incidents based on:
    - What incidents worked before
    - What services Responder struggles with
    - Escalating difficulty as Responder improves
    """

    def __init__(self, infrastructure: Infrastructure, incident_generator: IncidentGenerator):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.infrastructure = infrastructure
        self.incident_generator = incident_generator

        self.episode_history = []
        self.responder_performance_history = []
        self.incident_effectiveness = {}   # Track which incidents worked best
        self.target_vulnerability = {}     # Track which services are vulnerable
        self.current_difficulty_level = 1
        self.total_responder_success_rate = 0.0

    def observe_responder_performance(self, performance_data: dict):
        """
        Observe Responder's performance and adapt strategy.

        Args:
            performance_data: {
                "success_rate": float (0-1),
                "avg_steps": int,
                "avg_reward": float,
                "failed_on_incidents": list,
                "succeeded_on_incidents": list,
                "healthy_services": int,
                "total_services": int,
            }
        """
        self.responder_performance_history.append(performance_data)
        self.total_responder_success_rate = performance_data.get("success_rate", 0.0)

        # Learn from failures: what incidents broke the Responder?
        failed_incidents = performance_data.get("failed_on_incidents", [])
        for incident in failed_incidents:
            incident_type = incident.get("type")
            targets = tuple(sorted(incident.get("targets", [])))
            key = f"{incident_type}_{targets}"
            self.incident_effectiveness[key] = self.incident_effectiveness.get(key, 0) + 1

        # Learn vulnerabilities: which services does Responder struggle with?
        if performance_data.get("healthy_services", 5) < 5:
            unhealthy = performance_data.get("unhealthy_services", [])
            for service in unhealthy:
                self.target_vulnerability[service] = self.target_vulnerability.get(service, 0) + 1

        # Adapt difficulty
        success_rate = performance_data.get("success_rate", 0.0)
        if success_rate > 0.85:
            self.current_difficulty_level = min(self.current_difficulty_level + 1, 10)
            print(f"[BREAKER] Responder success {success_rate:.1%} > 85% — ESCALATING to level {self.current_difficulty_level}")
        elif success_rate < 0.40:
            self.current_difficulty_level = max(self.current_difficulty_level - 1, 1)
            print(f"[BREAKER] Responder success {success_rate:.1%} < 40% — DE-ESCALATING to level {self.current_difficulty_level}")
        else:
            print(f"[BREAKER] Responder success {success_rate:.1%} — maintaining level {self.current_difficulty_level}")

    def break_system(self, responder_performance: dict = None) -> dict:
        """
        Generate a new incident strategy based on learning.
        Uses Llama-3.3-70B-Instruct to create adaptive, escalating attacks.
        """

        performance_context = ""
        if responder_performance:
            success_rate = responder_performance.get("success_rate", 0.0)
            avg_steps    = responder_performance.get("avg_steps", "unknown")
            performance_context = f"""
RESPONDER PERFORMANCE:
- Success rate: {success_rate:.1%}
- Average steps to resolve: {avg_steps}
- Responder is {'STRUGGLING' if success_rate < 0.5 else 'DOING WELL' if success_rate > 0.8 else 'ADAPTING'}

BREAKER LEARNING:
- Current difficulty level: {self.current_difficulty_level} / 10
- Incidents that worked best: {json.dumps(self._get_top_effective_incidents(3))}
- Services Responder struggles with: {json.dumps(self._get_vulnerable_services(3))}
"""

        history_context = ""
        if self.episode_history:
            recent = self.episode_history[-5:]
            history_context = f"\nRECENT ATTACKS:\n{json.dumps(recent, indent=2)}"

        escalation_instructions = self._get_escalation_instructions()

        prompt = f"""You are a Breaker Agent that learns and escalates attacks.
{performance_context}
{history_context}

ESCALATION STRATEGY (Level {self.current_difficulty_level}/10):
{escalation_instructions}

Current infrastructure services: {SERVICE_NAMES}

Generate the NEXT attack to challenge the Responder.
Target their vulnerabilities. Repeat what worked. Escalate complexity.
Return JSON only."""

        try:
            print(f"[BREAKER] Generating attack at difficulty level {self.current_difficulty_level} using {MODEL_NAME}...")
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.8,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = "\n".join(
                    l for l in raw.split("\n")
                    if not l.strip().startswith("```")
                ).strip()
            plan = json.loads(raw)
            self._inject(plan)
            self.episode_history.append(plan)
            print(f"[BREAKER] Injected: {plan['incident_type']} on {plan['target_services']}")
            return plan
        except Exception as e:
            print(f"[BREAKER] LLM error: {e}, using fallback")
            fallback = self._intelligent_fallback()
            self.episode_history.append(fallback)
            return fallback

    def _get_escalation_instructions(self) -> str:
        level = self.current_difficulty_level
        if level <= 2:
            return """- Single service failures
- Simple incident types (cpu_spike, service_crash)
- P2 or P3 severity"""
        elif level <= 4:
            return """- Target 2 services
- Combine incident types (cascade_failure)
- P1 or P2 severity
- Create dependency chains"""
        elif level <= 6:
            return """- Target 3 services simultaneously
- Mix incident types for confusion
- Include red_herring to mislead
- P1 severity dominant
- Force Responder to make hard choices"""
        elif level <= 8:
            return """- Target 4+ services
- Cascade failures that amplify
- Heavy use of red_herring
- Conflicting error signals
- All P1 severity
- Design incidents that break their patterns"""
        else:
            return """- Attack the entire system
- Multiple cascading failures
- Extreme red_herrings
- Conflict every metric
- All P1 severity
- Design scenarios that seem unresolvable
- Force complex coordination"""

    def _get_top_effective_incidents(self, top_n: int = 3) -> list:
        if not self.incident_effectiveness:
            return []
        sorted_incidents = sorted(
            self.incident_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [item[0] for item in sorted_incidents[:top_n]]

    def _get_vulnerable_services(self, top_n: int = 3) -> list:
        if not self.target_vulnerability:
            return []
        sorted_services = sorted(
            self.target_vulnerability.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [item[0] for item in sorted_services[:top_n]]

    def _inject(self, plan: dict):
        incident_type = INCIDENT_TYPE_MAP.get(
            plan.get("incident_type", "service_crash"),
            IncidentType.SERVICE_CRASH,
        )
        targets  = plan.get("target_services", [random.choice(SERVICE_NAMES)])
        severity = SEVERITY_MAP.get(plan.get("severity", "p2"), Severity.P2)

        valid_targets = [t for t in targets if t in SERVICE_NAMES]
        if not valid_targets:
            valid_targets = [random.choice(SERVICE_NAMES)]

        self.incident_generator.inject(
            incident_type=incident_type,
            target_services=valid_targets,
            severity=severity,
        )

    def _intelligent_fallback(self) -> dict:
        """Intelligent fallback when LLM fails — uses learned knowledge."""
        effective = self._get_top_effective_incidents(1)
        if effective and random.random() > 0.3:
            incident_type = effective[0].split("_")[0]
        else:
            incident_type = random.choice(list(INCIDENT_TYPE_MAP.keys()))

        vulnerable = self._get_vulnerable_services(1)
        if vulnerable and random.random() > 0.3:
            target = vulnerable[0]
        else:
            target = random.choice(SERVICE_NAMES)

        if self.current_difficulty_level >= 7:
            severity = "p1"
        elif self.current_difficulty_level >= 4:
            severity = random.choice(["p1", "p2"])
        else:
            severity = random.choice(["p2", "p3"])

        plan = {
            "incident_type":   incident_type,
            "target_services": [target],
            "severity":        severity,
            "reasoning":       f"Learned fallback: difficulty level {self.current_difficulty_level}",
        }
        self._inject(plan)
        return plan

    def get_status(self) -> dict:
        return {
            "difficulty_level":         self.current_difficulty_level,
            "responder_success_rate":   self.total_responder_success_rate,
            "total_episodes":           len(self.episode_history),
            "total_observations":       len(self.responder_performance_history),
            "effective_incidents":      self._get_top_effective_incidents(5),
            "vulnerable_services":      self._get_vulnerable_services(5),
            "recent_attacks":           self.episode_history[-3:] if self.episode_history else [],
            "model":                    MODEL_NAME,
        }

    def reset(self):
        self.episode_history = []
        self.responder_performance_history = []
        self.incident_effectiveness = {}
        self.target_vulnerability = {}
        self.current_difficulty_level = 1
        self.total_responder_success_rate = 0.0
