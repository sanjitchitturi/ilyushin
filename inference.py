"""
Inference Script — Ilyushin Incident Response Agent
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
from typing import Optional

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

from agents.monitor import MonitorAgent
from agents.responder import ResponderAgent
from agents.verifier import VerifierAgent
from agents.trainer import TrainerAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
BASE_URL     = os.getenv("ENV_BASE_URL", "ws://localhost:8000")
BENCHMARK    = "ilyushin-incident-response"

VALID_SERVICES = ["web_server", "database", "cache", "queue", "api_gateway"]

LINE = "=" * 64
DASH = "-" * 64
THIN = "." * 64


# ── display helpers ────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{LINE}")
    print(f"  {title}")
    print(f"{LINE}")

def print_sub(title: str):
    print(f"\n{DASH}")
    print(f"  {title}")
    print(f"{DASH}")

def print_row(label: str, value: str, indent: int = 2):
    print(f"{' ' * indent}{label:<28} {value}")

def print_service_table(infrastructure: dict):
    print(f"\n  {'SERVICE':<16} {'STATUS':<12} {'CPU':<10} {'MEM':<10} {'LATENCY':<12} {'ERR%':<8}")
    print(f"  {THIN}")
    for svc, data in infrastructure.items():
        status  = data.get("status", "unknown").upper()
        metrics = data.get("metrics", {})
        cpu     = metrics.get("cpu", {}).get("value", 0)
        mem     = metrics.get("memory", {}).get("value", 0)
        lat     = metrics.get("latency", {}).get("value", 0)
        err     = metrics.get("error_rate", {}).get("value", 0)
        print(f"  {svc:<16} {status:<12} {cpu:<10.1f} {mem:<10.1f} {lat:<12.1f} {err:<8.2f}")

def print_incidents(active_incidents: list):
    if not active_incidents:
        print("  No active incidents.")
        return
    for inc in active_incidents:
        print(f"  [{inc.get('severity','?').upper()}] {inc.get('type','?').upper()} "
              f"on {inc.get('targets', [])} — {inc.get('description','')}")

def print_monitor_report(report: dict):
    print_row("Incident detected",  str(report.get("incident_detected", "?")))
    print_row("Severity",           report.get("severity", "?").upper())
    print_row("Affected services",  str(report.get("affected_services", [])))
    print_row("Root cause",         report.get("root_cause_hypothesis", "?"))
    print_row("Recommended action", report.get("recommended_first_action", "?"))

def print_verifier_report(report: dict):
    print_row("Resolution confirmed", str(report.get("resolution_confirmed", "?")))
    print_row("Still unhealthy",      str(report.get("services_still_unhealthy", [])))
    print_row("Recommended",          report.get("recommended_action", "?"))
    warnings = report.get("lingering_warnings", [])
    if warnings:
        print(f"\n  Lingering warnings:")
        for w in warnings:
            print(f"    {w}")
    print_row("Notes", report.get("verification_notes", "?"))

def health_bar(healthy: int, total: int) -> str:
    if total == 0:
        return "[          ] 0/0"
    filled = int(10 * healthy / total)
    bar = "#" * filled + "." * (10 - filled)
    pct = int(100 * healthy / total)
    return f"[{bar}] {healthy}/{total} ({pct}%)"


# ── mandatory log format ───────────────────────────────────────────

def log_start(task: str, model: str):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    error_str = error if error else "null"
    done_str  = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── helpers ────────────────────────────────────────────────────────

def get_state_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "__dict__"):
        return obs.__dict__
    return {}

def get_down_services(state: dict) -> list:
    infrastructure = state.get("infrastructure", {})
    return [svc for svc, data in infrastructure.items() if data.get("status") == "down"]

def get_degraded_services(state: dict) -> list:
    infrastructure = state.get("infrastructure", {})
    return [svc for svc, data in infrastructure.items() if data.get("status") == "degraded"]

def get_unhealthy_services(state: dict) -> list:
    infrastructure = state.get("infrastructure", {})
    return [svc for svc, data in infrastructure.items() if data.get("status") != "healthy"]

def get_stuck_override(actions_taken: list, monitor_report: dict, state: dict) -> Optional[dict]:
    if len(actions_taken) < 3:
        return None

    last_three = actions_taken[-3:]
    if len(set(last_three)) == 1 and last_three[0] == "read_logs":
        down = get_down_services(state)
        degraded = get_degraded_services(state)
        candidates = down + degraded
        target = candidates[0] if candidates else "database"
        if target not in VALID_SERVICES:
            target = "database"
        return {"type": "restart_service", "target_service": target}

    if len(actions_taken) >= 4:
        last_four = actions_taken[-4:]
        if all(a == last_four[0] for a in last_four):
            down = get_down_services(state)
            degraded = get_degraded_services(state)
            candidates = down + degraded
            for svc in candidates:
                if svc not in last_four[0]:
                    return {"type": "restart_service", "target_service": svc}

    return None

def force_resolve_healthy(state: dict, resolved: list) -> Optional[dict]:
    infrastructure = state.get("infrastructure", {})
    for svc, data in infrastructure.items():
        if data.get("status") == "healthy" and svc not in resolved:
            return {"type": "resolve", "target_service": svc}
    return None

def compute_score(state: dict) -> float:
    infrastructure = state.get("infrastructure", {})
    if not infrastructure:
        return 0.0
    total   = len(infrastructure)
    healthy = sum(1 for d in infrastructure.values() if d.get("status") == "healthy")
    return round(healthy / total, 4) if total > 0 else 0.0


# ── episode runner ─────────────────────────────────────────────────

def run_episode(task_id: str, trainer: TrainerAgent) -> tuple:
    log_start(task=task_id, model=MODEL_NAME)
    print_section(f"TASK: {task_id.upper()}")

    monitor   = MonitorAgent()
    responder = ResponderAgent()
    verifier  = VerifierAgent()

    with GenericEnvClient(base_url=BASE_URL).sync() as env:
        result = env.reset(task_id=task_id)
        state  = get_state_dict(result.observation) if hasattr(result, "observation") else get_state_dict(result)

        print_sub("INITIAL INFRASTRUCTURE STATE")
        print_service_table(state.get("infrastructure", {}))
        print_sub("ACTIVE INCIDENTS")
        print_incidents(state.get("active_incidents", []))

        print_sub("MONITOR AGENT — Initial Analysis")
        monitor_report = monitor.analyze(state)
        print_monitor_report(monitor_report)

        all_rewards       = []
        total_reward      = 0.0
        step_count        = 0
        actions_taken     = []
        reward_breakdown  = []
        resolved_services = []

        while not state.get("done", False):
            steps_remaining = 20 - step_count
            down      = get_down_services(state)
            degraded  = get_degraded_services(state)
            unhealthy = get_unhealthy_services(state)

            print_sub(f"STEP {step_count + 1}  |  Steps remaining: {steps_remaining}  |  Health: {health_bar(state.get('healthy_services', 0), state.get('total_services', 5))}")
            print_row("Down services",     str(down)     if down     else "none")
            print_row("Degraded services", str(degraded) if degraded else "none")

            override = get_stuck_override(actions_taken, monitor_report, state)

            if override:
                action = override
                print_row("Stuck override", f"forcing {action['type']} on {action.get('target_service','')}")
            elif steps_remaining <= 3 and unhealthy:
                target = down[0] if down else degraded[0] if degraded else unhealthy[0]
                action = {"type": "restart_service", "target_service": target}
                print_row("Emergency action", f"restarting {target} with {steps_remaining} steps left")
            elif steps_remaining <= 2:
                resolve_action = force_resolve_healthy(state, resolved_services)
                action = resolve_action if resolve_action else {"type": "read_logs", "target_service": None}
            else:
                try:
                    action = responder.act(observation=state, monitor_report=monitor_report)
                except Exception as exc:
                    log_step(step_count + 1, "responder_error", 0.00, False, str(exc))
                    break

            action_str = action.get("type", "unknown")
            if action.get("target_service"):
                action_str += f"({action['target_service']})"

            print_row("Responder action", action_str)

            try:
                result = env.step(action)
                step_data = get_state_dict(result.observation) if hasattr(result, "observation") else get_state_dict(result)
                reward    = result.reward if hasattr(result, "reward") and result.reward is not None else step_data.get("reward", -0.1)
                done      = result.done if hasattr(result, "done") else step_data.get("done", False)
                state     = step_data
            except Exception as exc:
                log_step(step_count + 1, action_str, 0.00, True, str(exc))
                break

            step_count   += 1
            total_reward += reward
            all_rewards.append(reward)
            actions_taken.append(action.get("type", "unknown"))
            reward_breakdown.append({"step": step_count, "action": action_str, "reward": reward})

            if action.get("type") == "resolve" and action.get("target_service"):
                resolved_services.append(action["target_service"])

            print_row("Result",       state.get("last_action_result", ""))
            print_row("Reward",       f"{reward:+.2f}")
            print_row("Health after", health_bar(state.get("healthy_services", 0), state.get("total_services", 5)))

            log_step(step_count, action_str, reward, done)

            if done:
                break

            monitor_report = monitor.analyze(state)
            print_row("Monitor update", monitor_report.get("root_cause_hypothesis", ""))

            unhealthy_now = get_unhealthy_services(state)
            if not unhealthy_now:
                for svc in VALID_SERVICES:
                    if svc not in resolved_services and not state.get("done", False):
                        try:
                            result = env.step({"type": "resolve", "target_service": svc})
                            step_data = get_state_dict(result.observation) if hasattr(result, "observation") else get_state_dict(result)
                            reward    = result.reward if hasattr(result, "reward") and result.reward is not None else step_data.get("reward", -0.1)
                            done      = result.done if hasattr(result, "done") else step_data.get("done", False)
                            state     = step_data
                            step_count   += 1
                            total_reward += reward
                            all_rewards.append(reward)
                            actions_taken.append("resolve")
                            resolved_services.append(svc)
                            log_step(step_count, f"resolve({svc})", reward, done)
                            print_row("Auto-resolved", f"{svc}  reward={reward:+.2f}")
                            if done:
                                break
                        except Exception:
                            pass
                        break

        print_sub("VERIFIER AGENT — Final Check")
        verification = verifier.verify(state)
        print_verifier_report(verification)

        print_sub("INFRASTRUCTURE — Final State")
        print_service_table(state.get("infrastructure", {}))

        score   = compute_score(state)
        success = score == 1.0

        print_sub("EPISODE RESULT")
        print_row("Final score",  f"{score:.2f}")
        print_row("Total reward", f"{round(total_reward, 2):+.2f}")
        print_row("Steps taken",  str(step_count))
        print_row("Oncall paged", str(state.get("oncall_paged", False)))
        print_row("Outcome",      "SUCCESS" if success else "INCOMPLETE")

        log_end(success=success, steps=step_count, score=score, rewards=all_rewards)

        episode_summary = {
            "task_id":         task_id,
            "total_steps":     step_count,
            "total_reward":    round(total_reward, 4),
            "final_score":     score,
            "healthy_services": state.get("healthy_services", 0),
            "total_services":  state.get("total_services", 5),
            "oncall_paged":    state.get("oncall_paged", False),
            "done":            state.get("done", False),
            "actions_taken":   actions_taken,
            "reward_breakdown": reward_breakdown,
        }
        trainer.analyze_episode(episode_summary)

        return score, step_count, round(total_reward, 4)


# ── main ───────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is not set.")
        sys.exit(1)

    print_section("ILYUSHIN — INCIDENT RESPONSE AGENT")
    print_row("Model",     MODEL_NAME)
    print_row("API base",  API_BASE_URL)
    print_row("Tasks",     "easy / medium / hard")
    print_row("Max steps", "20 per episode")

    trainer  = TrainerAgent()
    task_ids = ["easy", "medium", "hard"]
    results  = {}

    for task_id in task_ids:
        score, steps, reward = run_episode(task_id=task_id, trainer=trainer)
        results[task_id] = {"score": score, "steps": steps, "reward": reward}

    print_section("FINAL RESULTS")
    print(f"\n  {'TASK':<12} {'SCORE':<10} {'STEPS':<10} {'REWARD':<12} {'OUTCOME'}")
    print(f"  {DASH}")
    total_score = 0.0
    for task_id, r in results.items():
        outcome = "SUCCESS" if r["score"] == 1.0 else "INCOMPLETE"
        print(f"  {task_id:<12} {r['score']:<10.2f} {r['steps']:<10} {r['reward']:<12.2f} {outcome}")
        total_score += r["score"]

    avg = total_score / len(results)
    print(f"  {THIN}")
    print(f"  {'AVERAGE':<12} {avg:<10.2f}")

    performance = trainer.get_responder_performance()
    print_sub("TRAINER AGENT — Performance Summary")
    for k, v in performance.items():
        print_row(k, str(v))

    print(f"\n{LINE}\n")


if __name__ == "__main__":
    main()