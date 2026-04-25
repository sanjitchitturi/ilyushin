import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

SYSTEM_PROMPT = """You are a Verifier Agent in a production incident response system.
Your job is to confirm whether all services are running, report any lingering issues,
and give a clear post-incident health summary.

You must return ONLY a JSON object with exactly these fields:
{
    "resolution_confirmed": true or false,
    "services_still_unhealthy": ["service1", "service2"],
    "lingering_warnings": ["service.metric=value — explanation"],
    "verification_notes": "clear summary of current system state",
    "recommended_action": "resolved or continue_fixing or monitor"
}

HOW TO VERIFY:

STEP 1 — Check if services are running:
- Look at each service STATUS field
- status=healthy means the service IS running and IS resolved
- status=degraded means the service is running but impaired
- status=down means the service is not running at all
- Only add to services_still_unhealthy if status=degraded or status=down
- NEVER add a status=healthy service to services_still_unhealthy

STEP 2 — Check for lingering metric issues:
- Even healthy services can have elevated metrics after recovery
- This is normal post-incident behavior
- Check these thresholds for warnings:
  cpu above 75% = warning
  memory above 80% = warning
  latency above 500ms = warning
  error_rate above 5% = warning
- Add these to lingering_warnings as observations
- Lingering warnings do NOT make a service unhealthy
- Format: "service_name.metric=value — post-recovery noise, monitor"

STEP 3 — Check active incident logs:
- Incident logs may be stale after fixes are applied
- Do NOT use incident logs to determine if a service is unhealthy
- Only use incident logs to note what the original incident was
- A service with status=healthy is resolved even if an incident log still references it

STEP 4 — Set resolution_confirmed:
- true if ALL services show status=healthy
- false if ANY service shows status=degraded or status=down

STEP 5 — Set recommended_action:
- resolved — all services healthy, no critical metrics
- monitor — all services healthy but some lingering warnings above thresholds
- continue_fixing — one or more services still degraded or down

Return raw JSON only. No explanation. No markdown."""


class VerifierAgent:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def verify(self, observation: dict) -> dict:
        infrastructure = observation.get("infrastructure", {})
        healthy = observation.get("healthy_services", 0)
        total = observation.get("total_services", 5)
        active_incidents = observation.get("active_incidents", [])

        service_summary = []
        for svc, data in infrastructure.items():
            status = data.get("status", "unknown")
            metrics = data.get("metrics", {})
            cpu = metrics.get("cpu", {}).get("value", 0)
            mem = metrics.get("memory", {}).get("value", 0)
            lat = metrics.get("latency", {}).get("value", 0)
            err = metrics.get("error_rate", {}).get("value", 0)
            service_summary.append(
                f"{svc}: status={status} cpu={cpu}% mem={mem}% "
                f"latency={lat}ms error_rate={err}%"
            )

        prompt = f"""Verify the current production infrastructure state after incident response:

HEALTHY SERVICES: {healthy}/{total}

SERVICE STATUS SUMMARY (trust this above all else):
{chr(10).join(service_summary)}

FULL INFRASTRUCTURE STATE:
{json.dumps(infrastructure, indent=2)}

ACTIVE INCIDENT LOGS (may be stale after fixes):
{json.dumps(active_incidents, indent=2)}

Verify each service is running, report any lingering metric warnings,
and confirm whether the incident is fully resolved."""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
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
                "resolution_confirmed": healthy == total,
                "services_still_unhealthy": [],
                "lingering_warnings": [],
                "verification_notes": f"Verifier error: {str(e)}",
                "recommended_action": "resolved" if healthy == total else "continue_fixing"
            }