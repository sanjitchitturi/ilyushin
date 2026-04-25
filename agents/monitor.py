import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = "meta-llama/Llama-3.3-70B-Instruct"  # hardcoded — always 70B regardless of MODEL_NAME env

SYSTEM_PROMPT = """You are a Monitor Agent in a production incident response system.
Your job is to analyze infrastructure metrics and logs, classify the incident, and determine severity.

You must return ONLY a JSON object with exactly these fields:
{
    "incident_detected": true or false,
    "severity": "p1" or "p2" or "p3",
    "affected_services": ["service1", "service2"],
    "root_cause_hypothesis": "brief description of likely root cause",
    "recommended_first_action": "one of: read_logs, check_metrics, restart_service, scale_up, rollback"
}

Valid services: web_server, database, cache, queue, api_gateway
Return raw JSON only. No explanation. No markdown."""


class MonitorAgent:
    """
    Monitor Agent — always uses Llama-3.3-70B-Instruct.
    Analyzes infrastructure state and classifies incidents.
    """

    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def analyze(self, observation: dict) -> dict:
        infrastructure  = observation.get("infrastructure", {})
        active_incidents = observation.get("active_incidents", [])
        healthy = observation.get("healthy_services", 0)
        total   = observation.get("total_services", 5)

        prompt = f"""Analyze the following production infrastructure state:

HEALTHY SERVICES: {healthy}/{total}
ACTIVE INCIDENTS: {json.dumps(active_incidents, indent=2)}
INFRASTRUCTURE STATE:
{json.dumps(infrastructure, indent=2)}

Classify the incident and identify the root cause."""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = "\n".join(
                    l for l in raw.split("\n")
                    if not l.strip().startswith("```")
                ).strip()
            return json.loads(raw)
        except Exception as e:
            return {
                "incident_detected":        True,
                "severity":                 "p2",
                "affected_services":        [],
                "root_cause_hypothesis":    f"Monitor error: {str(e)}",
                "recommended_first_action": "read_logs",
            }
