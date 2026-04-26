#!/usr/bin/env python3
"""
ILYUSHIN — GRPO ADVERSARIAL TRAINING
Responder (QLoRA+GRPO / Llama-3.2-3B) vs Breaker (Llama 70B)

Candidate scoring uses local reward estimation (snapshot approach):
  - At each step, snapshot the current state dict
  - Generate GROUP_SIZE candidate actions
  - Score each locally using compute_reward() — no extra env calls
  - Normalize scores within group → advantages for GRPO update
  - Advance the real episode with the best-scoring candidate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import time
import random
import requests
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

from dataset import SYSTEM_PROMPT, format_prompt
from curriculum import CurriculumManager
from env.state import EnvState
from env.reward import compute_reward

# ============================================================================
# CONFIG — tweak these to scale up or down
# ============================================================================

WORK_DIR        = Path(".")
CHECKPOINTS_DIR = WORK_DIR / "checkpoints"
LOGS_DIR        = WORK_DIR / "logs"
PLOTS_DIR       = WORK_DIR / "plots"

for d in [CHECKPOINTS_DIR, LOGS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ENV_URL    = os.getenv("BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

# ── GRPO hyperparameters ──────────────────────────────────────────
# GROUP_SIZE: number of candidate actions generated per state for ranking
#   Testing:     4   (fast, enough signal to verify loop works)
#   Real run:    4   (sweet spot for 3B model on single GPU)
#   More data:   8   (richer gradient, ~2x slower per update)
GROUP_SIZE = 4

# EPISODES_PER_PHASE: training episodes per curriculum phase (easy/medium/hard)
#   Testing:     10  (just enough to confirm no crashes, ~40 rollouts/phase)
#   Real run:    30  (120 rollouts/phase, comparable to old PPO 60-episode setup)
#   Full run:    60  (240 rollouts/phase, maximum learning signal)
EPISODES_PER_PHASE = 10

# GRPO_EPOCHS: gradient update passes over each collected group batch
#   Testing:     1
#   Real run:    2-3
GRPO_EPOCHS = 1

# LEARNING_RATE
#   Testing:     1e-5
#   Real run:    5e-6  (more stable for longer runs)
LEARNING_RATE = 1e-5

# ── Other constants ───────────────────────────────────────────────
MAX_STEPS         = 10
EVAL_EPISODES     = 10
BASELINE_EPISODES = 20
MAX_NEW_TOKENS    = 32
MIN_NEW_TOKENS    = 4
MAX_PROMPT_LENGTH = 512

VALID_ACTIONS  = ["read_logs", "check_metrics", "restart_service",
                  "scale_up", "rollback", "page_oncall", "resolve"]
VALID_SERVICES = ["web_server", "database", "cache", "queue", "api_gateway"]

# ============================================================================
# LOGGER
# ============================================================================

class AdversarialLogger:
    def __init__(self, logs_dir: Path):
        self.logs_dir        = logs_dir
        self.episodes_log    = []
        self.phases_log      = []
        self.baseline_log    = {}
        self.breaker_log     = []
        self.action_dist_log = defaultdict(lambda: defaultdict(int))

    def log_episode(self, episode, task_id, total_reward, total_steps, success_rate,
                    healthy_services, total_services, breaker_difficulty,
                    actions_taken=None, oncall_paged=False):
        self.episodes_log.append({
            "episode":            episode,
            "task_id":            task_id,
            "total_reward":       total_reward,
            "total_steps":        total_steps,
            "success_rate":       success_rate,
            "healthy_services":   healthy_services,
            "total_services":     total_services,
            "breaker_difficulty": breaker_difficulty,
            "oncall_paged":       oncall_paged,
            "timestamp":          datetime.now().isoformat(),
        })
        if actions_taken:
            for a in actions_taken:
                self.action_dist_log[task_id][a] += 1

    def log_phase(self, phase, start_episode, end_episode, avg_reward,
                  avg_success_rate, breaker_difficulty, avg_steps=0):
        self.phases_log.append({
            "phase":              phase,
            "start_episode":      start_episode,
            "end_episode":        end_episode,
            "avg_reward":         avg_reward,
            "avg_success_rate":   avg_success_rate,
            "avg_steps":          avg_steps,
            "breaker_difficulty": breaker_difficulty,
            "timestamp":          datetime.now().isoformat(),
        })

    def log_baseline(self, task_id, avg_reward, avg_success_rate, avg_steps, num_episodes):
        self.baseline_log[task_id] = {
            "avg_reward":       avg_reward,
            "avg_success_rate": avg_success_rate,
            "avg_steps":        avg_steps,
            "num_episodes":     num_episodes,
        }

    def log_breaker_status(self, phase, difficulty_level, effective_incidents,
                           vulnerable_services, responder_success_rate):
        self.breaker_log.append({
            "phase":                  phase,
            "difficulty_level":       difficulty_level,
            "effective_incidents":    effective_incidents,
            "vulnerable_services":    vulnerable_services,
            "responder_success_rate": responder_success_rate,
            "timestamp":              datetime.now().isoformat(),
        })

    def flush(self):
        with open(self.logs_dir / "episodes.json", "w") as f:
            json.dump(self.episodes_log, f, indent=2)
        with open(self.logs_dir / "phases.json", "w") as f:
            json.dump(self.phases_log, f, indent=2)
        with open(self.logs_dir / "baseline.json", "w") as f:
            json.dump(self.baseline_log, f, indent=2)
        with open(self.logs_dir / "breaker.json", "w") as f:
            json.dump(self.breaker_log, f, indent=2)
        with open(self.logs_dir / "action_dist.json", "w") as f:
            json.dump({k: dict(v) for k, v in self.action_dist_log.items()}, f, indent=2)
        print(f"[LOGS] Flushed to {self.logs_dir.resolve()}/")

# ============================================================================
# MEMORY UTILS
# ============================================================================

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

def print_gpu_memory(label=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    print(f"[GPU] {label} | allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

# ============================================================================
# ACTION PARSING
# ============================================================================

def parse_action(text: str, debug: bool = False) -> dict:
    """
    Parse model output into a valid action dict.
    Two-stage parsing:
    1. Try JSON parsing (ideal case)
    2. Fall back to extracting action/service from English text
    """
    original_text = text

    try:
        text  = text.strip()
        start = text.find("{")
        end   = text.find("}", start) + 1 if start != -1 else -1

        if start != -1 and end > start:
            candidate = text[start:end]
            action = json.loads(candidate)
        else:
            candidate = '{"type": "' + text
            start2 = candidate.find("{")
            end2   = candidate.find("}", start2) + 1
            if end2 > start2:
                candidate = candidate[start2:end2]
            action = json.loads(candidate)

        if action.get("type") not in VALID_ACTIONS:
            if debug:
                print(f"[DEBUG] Invalid type: {action.get('type')!r}")
            action["type"] = "read_logs"

        target = action.get("target_service")
        if target and target not in VALID_SERVICES:
            action["target_service"] = None
        if "target_service" not in action:
            action["target_service"] = None

        if debug:
            print(f"[DEBUG] Raw    : {repr(original_text[:120])}")
            print(f"[DEBUG] Parsed : {action}")

        return action

    except Exception as e:
        if debug:
            print(f"[DEBUG] Raw    : {repr(original_text[:120])}")
            print(f"[DEBUG] FAILED : {e} — trying English extraction")

        text_lower  = original_text.lower()
        action_type = "read_logs"
        if "restart" in text_lower:
            action_type = "restart_service"
        elif "scale up" in text_lower or "scale_up" in text_lower:
            action_type = "scale_up"
        elif "rollback" in text_lower:
            action_type = "rollback"
        elif "check metric" in text_lower or "check_metric" in text_lower:
            action_type = "check_metrics"
        elif "resolve" in text_lower:
            action_type = "resolve"
        elif "page" in text_lower and "oncall" in text_lower:
            action_type = "page_oncall"
        elif "read log" in text_lower or "read_log" in text_lower:
            action_type = "read_logs"

        target = None
        for svc in VALID_SERVICES:
            if svc in text_lower or svc.replace("_", " ") in text_lower:
                target = svc
                break

        if debug:
            print(f"[DEBUG] English extraction: type={action_type} target={target}")

        return {"type": action_type, "target_service": target}

# ============================================================================
# ENVIRONMENT HELPERS
# ============================================================================

def env_reset(env_url: str, session_id: str, task_id: str) -> dict:
    res = requests.post(
        f"{env_url}/env/reset",
        json={"session_id": session_id, "task_id": task_id},
        timeout=15,
    )
    res.raise_for_status()
    return res.json().get("state", res.json())


def env_step(env_url: str, session_id: str, action: dict) -> tuple:
    res = requests.post(
        f"{env_url}/env/step",
        json={"session_id": session_id, "action": action},
        timeout=15,
    )
    res.raise_for_status()
    data   = res.json()
    state  = data.get("state", data)
    reward = data.get("reward", -0.1)
    if isinstance(reward, dict):
        reward = reward.get("value", -0.1)
    done               = bool(data.get("done", state.get("done", False)))
    breaker_difficulty = int(data.get("breaker_status", {}).get("difficulty_level", 1))
    return state, float(reward), done, breaker_difficulty


def env_feedback(env_url: str, session_id: str, performance: dict):
    try:
        res = requests.post(
            f"{env_url}/env/feedback",
            json={"session_id": session_id, **performance},
            timeout=10,
        )
        if res.status_code == 404:
            return {}
        res.raise_for_status()
        return res.json().get("breaker_status", {})
    except Exception as e:
        print(f"[FEEDBACK] Error: {e}")
        return {}

# ============================================================================
# SNAPSHOT — convert state dict to EnvState for local reward estimation
# ============================================================================

def state_dict_to_envstate(state: dict, step_count: int, actions_taken: list) -> EnvState:
    """
    Convert the raw state dict returned by env_step into an EnvState
    so we can call compute_reward() locally without hitting the env.
    """
    return EnvState(
        task_id=state.get("task_id", "unknown"),
        step_count=step_count,
        done=state.get("done", False),
        infrastructure_snapshot=state.get("infrastructure", {}),
        active_incidents=state.get("active_incidents", []),
        actions_taken=list(actions_taken),
        services_resolved=state.get("services_resolved", []),
        total_services=state.get("total_services", 5),
        healthy_services=state.get("healthy_services", 0),
        oncall_paged=state.get("oncall_paged", False),
        last_action=state.get("last_action", "none"),
        last_action_result=state.get("last_action_result", ""),
        last_action_success=state.get("last_action_success", True),
    )


def predict_next_state(current_envstate: EnvState, action: dict) -> EnvState:
    """
    Locally predict the next EnvState for a candidate action.

    Intentionally simple — we only need good-enough predictions to
    rank candidates within a group, not perfect accuracy.

    Rules mirror the real env logic in environment.py:
      - restart_service / scale_up / rollback on DOWN or DEGRADED  -> healthy +1
      - restart_service / scale_up / rollback on HEALTHY           -> wrong action
      - resolve on a HEALTHY unresolved service                    -> clears incident
      - read_logs / check_metrics                                  -> no state change
      - page_oncall                                                -> oncall_paged=True
    """
    import copy

    action_type = action.get("type", "read_logs")
    target      = action.get("target_service")

    new_state = EnvState(
        task_id=current_envstate.task_id,
        step_count=current_envstate.step_count + 1,
        done=current_envstate.done,
        infrastructure_snapshot=copy.deepcopy(current_envstate.infrastructure_snapshot),
        active_incidents=list(current_envstate.active_incidents),
        actions_taken=list(current_envstate.actions_taken) + [action_type],
        services_resolved=list(current_envstate.services_resolved),
        total_services=current_envstate.total_services,
        healthy_services=current_envstate.healthy_services,
        oncall_paged=current_envstate.oncall_paged,
        last_action=action_type,
        last_action_result="predicted",
        last_action_success=True,
    )

    infra = new_state.infrastructure_snapshot

    if action_type in ("restart_service", "scale_up", "rollback") and target:
        svc_data = infra.get(target, {})
        status   = svc_data.get("status", "healthy")
        if status in ("down", "degraded"):
            infra[target] = dict(svc_data)
            infra[target]["status"] = "healthy"
            new_state.healthy_services = min(
                current_envstate.healthy_services + 1,
                current_envstate.total_services,
            )
            # Clear incidents targeting this service
            new_state.active_incidents = [
                inc for inc in current_envstate.active_incidents
                if target not in inc.get("targets", [])
            ]
        else:
            # Service already healthy — wrong action
            new_state.last_action_success = False

    elif action_type == "resolve" and target:
        svc_data = infra.get(target, {})
        status   = svc_data.get("status", "healthy")
        if status == "healthy" and target not in current_envstate.services_resolved:
            new_state.services_resolved = list(current_envstate.services_resolved) + [target]
            # Clear one incident referencing this service
            cleared   = False
            remaining = []
            for inc in current_envstate.active_incidents:
                if not cleared and target in inc.get("targets", []):
                    cleared = True
                else:
                    remaining.append(inc)
            new_state.active_incidents = remaining

    elif action_type == "page_oncall":
        new_state.oncall_paged = True

    # Mark done if all services healthy
    if new_state.healthy_services == new_state.total_services:
        new_state.done = True

    return new_state


def estimate_candidate_reward(
    current_envstate: EnvState,
    action: dict,
    actions_taken_so_far: list,
) -> float:
    """
    Estimate reward for a candidate action using local state prediction.
    Mirrors the read_logs spam penalty from the rollout loop.
    """
    predicted_next = predict_next_state(current_envstate, action)
    reward, _      = compute_reward(
        old_state=current_envstate,
        new_state=predicted_next,
        action_type=action.get("type", "read_logs"),
        action_success=predicted_next.last_action_success,
    )

    # read_logs spam penalty
    read_logs_count = actions_taken_so_far.count("read_logs")
    if action.get("type") == "read_logs" and read_logs_count >= 1:
        reward -= 0.5

    return round(reward, 4)

# ============================================================================
# MODEL LOADING — standard CausalLM + QLoRA (no value head needed for GRPO)
# ============================================================================

def load_model_and_tokenizer(model_name: str):
    print(f"[TRAIN] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"[TRAIN] Loading model with QLoRA: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(base_model, lora_config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[TRAIN] Gradient checkpointing enabled")

    print(f"[TRAIN] Model loaded on: {next(model.parameters()).device}")
    model.print_trainable_parameters()
    print_gpu_memory("after model load")
    return model, tokenizer

# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_prompt(tokenizer, state: dict) -> torch.Tensor:
    text = f"{SYSTEM_PROMPT}\n\n{format_prompt(state)}"
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    )
    return encoding["input_ids"].squeeze(0)

# ============================================================================
# GENERATION
# ============================================================================

def generate_action(model, tokenizer, query_tensor: torch.Tensor) -> torch.Tensor:
    device    = next(model.parameters()).device
    input_ids = query_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
            repetition_penalty=1.1,
        )

    return output[0][query_tensor.shape[0]:]

# ============================================================================
# BASELINE
# ============================================================================

def run_baseline(env_url: str, task_id: str, n_episodes: int = BASELINE_EPISODES) -> tuple:
    print(f"\n[BASELINE] Running {n_episodes} random episodes on task={task_id}...")
    total_reward  = 0.0
    total_success = 0.0
    total_steps   = 0

    for ep in range(n_episodes):
        session_id = f"baseline_{task_id}_{ep}_{int(time.time())}"
        try:
            state     = env_reset(env_url, session_id, task_id)
            ep_reward = 0.0
            ep_steps  = 0
            done      = False
            while not done and ep_steps < MAX_STEPS:
                action_type = random.choice(VALID_ACTIONS)
                action = {"type": action_type, "target_service": None}
                if action_type in ("restart_service", "scale_up", "rollback",
                                   "check_metrics", "resolve"):
                    action["target_service"] = random.choice(VALID_SERVICES)
                state, reward, done, _ = env_step(env_url, session_id, action)
                ep_reward += reward
                ep_steps  += 1
            success = state.get("healthy_services", 0) / max(state.get("total_services", 5), 1)
            total_reward  += ep_reward
            total_success += success
            total_steps   += ep_steps
        except Exception as e:
            print(f"[BASELINE] Episode {ep} error: {e}")
            continue

    n           = max(n_episodes, 1)
    avg_reward  = total_reward  / n
    avg_success = total_success / n
    avg_steps   = total_steps   / n
    print(f"[BASELINE] {task_id}: avg_reward={avg_reward:.2f} "
          f"avg_success={avg_success:.2%} avg_steps={avg_steps:.1f}")
    return avg_reward, avg_success, avg_steps

# ============================================================================
# GRPO LOSS
# ============================================================================

def compute_grpo_loss(
    model,
    tokenizer,
    optimizer,
    groups: list,
    debug: bool = False,
) -> float:
    """
    Manual GRPO update.

    For each group:
    - Normalize rewards within the group -> advantages
    - For each (query, response, advantage), compute log-probs of response
      tokens under current policy
    - Loss = -mean(advantage * log_prob) across all tokens and groups

    Pushes up log-prob of above-average responses, pushes down the rest.
    """
    if not groups:
        return 0.0

    device     = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_valid    = 0

    for group in groups:
        query_tensor = group["query_tensor"].to(device)
        responses    = group["responses"]    # list of tensors
        rewards      = group["rewards"]      # list of floats

        if len(rewards) < 2:
            continue

        # Normalize rewards within group -> advantages
        rewards_t  = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r     = rewards_t.mean()
        std_r      = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        for resp_tensor, adv in zip(responses, advantages):
            if resp_tensor.numel() == 0:
                continue

            # Full sequence: [prompt tokens] + [response tokens]
            full_ids = torch.cat([query_tensor, resp_tensor.to(device)], dim=0).unsqueeze(0)

            # Mask prompt tokens in labels
            labels = full_ids.clone()
            labels[0, :query_tensor.shape[0]] = -100

            outputs    = model(input_ids=full_ids, labels=labels)
            token_loss = outputs.loss  # mean NLL over response tokens

            # GRPO: high advantage -> increase log-prob -> decrease NLL
            total_loss = total_loss + (-adv * token_loss)
            n_valid   += 1

    if n_valid == 0:
        return 0.0

    loss = total_loss / n_valid

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

# ============================================================================
# GRPO ROLLOUT — snapshot-based candidate scoring
#
# At each step:
#   1. Snapshot current state -> EnvState (no extra env call)
#   2. Generate GROUP_SIZE candidate actions from the same prompt
#   3. Score each locally with predict_next_state + compute_reward
#   4. Store (query, responses, estimated_rewards) group for GRPO update
#   5. Execute the best-scoring candidate in the real env session
#   6. Use the real env reward/state to track episode progress
# ============================================================================

def collect_grpo_episode(
    env_url:   str,
    model,
    tokenizer,
    task_id:   str,
    episode:   int,
    debug:     bool = False,
) -> dict:
    session_id = f"grpo_{task_id}_{episode}_{int(time.time())}"

    all_groups         = []
    actions_taken      = []   # action type strings for spam penalty tracking
    total_reward       = 0.0
    step_count         = 0
    done               = False
    breaker_difficulty = 1
    oncall_paged       = False

    try:
        state = env_reset(env_url, session_id, task_id)
    except Exception as e:
        print(f"[GRPO] Reset failed for {session_id}: {e}")
        return {
            "all_groups": [], "total_reward": 0.0, "steps": 0,
            "success_rate": 0.0, "healthy_services": 0,
            "breaker_difficulty": 1, "actions_taken": [],
            "oncall_paged": False, "session_id": session_id,
        }

    while not done and step_count < MAX_STEPS:

        # Tokenize current state
        try:
            query_tensor = tokenize_prompt(tokenizer, state)
        except Exception as e:
            print(f"[GRPO] Tokenize error at step {step_count}: {e}")
            break

        # Snapshot current state for local reward estimation
        current_envstate = state_dict_to_envstate(state, step_count, actions_taken)

        # Generate GROUP_SIZE candidate actions from the same prompt
        group_responses       = []
        group_rewards         = []
        best_action           = None
        best_estimated_reward = -999.0

        for g in range(GROUP_SIZE):
            try:
                generated_ids = generate_action(model, tokenizer, query_tensor)
                action_text   = tokenizer.decode(generated_ids, skip_special_tokens=True)
            except Exception as e:
                if debug:
                    print(f"[GRPO] Generate error g={g}: {e}")
                continue

            action = parse_action(action_text, debug=debug)

            # Fill missing target for actions that need one
            if action["type"] in ("restart_service", "scale_up", "rollback", "resolve") \
                    and not action.get("target_service"):
                infra = state.get("infrastructure", {})
                for svc, data in infra.items():
                    if data.get("status") != "healthy":
                        action["target_service"] = svc
                        break
                if not action.get("target_service"):
                    action["target_service"] = VALID_SERVICES[0]

            # Score locally — no env call
            estimated_reward = estimate_candidate_reward(
                current_envstate=current_envstate,
                action=action,
                actions_taken_so_far=actions_taken,
            )

            group_responses.append(generated_ids.cpu())
            group_rewards.append(estimated_reward)

            if estimated_reward > best_estimated_reward:
                best_estimated_reward = estimated_reward
                best_action           = action

            if debug:
                print(f"[GRPO] g={g} action={action['type']}({action.get('target_service')}) "
                      f"est_reward={estimated_reward:.3f}")

        # Need at least 2 candidates for reward normalization
        if len(group_responses) >= 2:
            all_groups.append({
                "query_tensor": query_tensor.cpu(),
                "responses":    group_responses,
                "rewards":      group_rewards,
            })

        # Execute best candidate in real env
        if best_action is None:
            best_action = {"type": "read_logs", "target_service": None}

        try:
            state, real_reward, done, breaker_difficulty = env_step(
                env_url, session_id, best_action
            )
        except Exception as e:
            print(f"[GRPO] Env step error at step {step_count}: {e}")
            break

        action_str = best_action["type"]
        if best_action.get("target_service"):
            action_str += f"({best_action['target_service']})"

        actions_taken.append(best_action["type"])
        total_reward += real_reward
        step_count   += 1

        if best_action["type"] == "page_oncall":
            oncall_paged = True

        print(f"[GRPO] ep={episode} step={step_count} "
              f"best={action_str} "
              f"est={best_estimated_reward:.3f} real={real_reward:.3f} "
              f"group_est={[round(r, 2) for r in group_rewards]}")

    healthy_services = state.get("healthy_services", 0)
    total_services   = state.get("total_services", 5)

    return {
        "all_groups":         all_groups,
        "total_reward":       round(total_reward, 4),
        "steps":              step_count,
        "success_rate":       healthy_services / max(total_services, 1),
        "healthy_services":   healthy_services,
        "breaker_difficulty": breaker_difficulty,
        "actions_taken":      actions_taken,
        "oncall_paged":       oncall_paged,
        "session_id":         session_id,
    }

# ============================================================================
# EVAL EPISODE
# ============================================================================

def run_eval_episode(env_url, model, tokenizer, task_id, episode) -> dict:
    session_id = f"eval_{task_id}_{episode}_{int(time.time())}"
    try:
        state = env_reset(env_url, session_id, task_id)
    except Exception as e:
        print(f"[EVAL] Reset failed: {e}")
        return {"reward": 0.0, "steps": 0, "success": 0.0,
                "healthy_services": 0, "breaker_difficulty": 1,
                "actions_taken": [], "oncall_paged": False}

    total_reward       = 0.0
    step_count         = 0
    done               = False
    breaker_difficulty = 1
    actions_taken      = []
    oncall_paged       = False

    while not done and step_count < MAX_STEPS:
        try:
            query_tensor  = tokenize_prompt(tokenizer, state)
            generated_ids = generate_action(model, tokenizer, query_tensor)
            action_text   = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"[EVAL] Generate error: {e}")
            break

        action = parse_action(action_text, debug=False)
        if action["type"] in ("restart_service", "scale_up", "rollback", "resolve") \
                and not action.get("target_service"):
            infra = state.get("infrastructure", {})
            for svc, data in infra.items():
                if data.get("status") != "healthy":
                    action["target_service"] = svc
                    break
            if not action.get("target_service"):
                action["target_service"] = VALID_SERVICES[0]

        try:
            state, reward, done, breaker_difficulty = env_step(
                env_url, session_id, action
            )
        except Exception as e:
            print(f"[EVAL] Step error: {e}")
            break

        total_reward += reward
        step_count   += 1
        actions_taken.append(action["type"])
        if action["type"] == "page_oncall":
            oncall_paged = True

    healthy_services = state.get("healthy_services", 0)
    total_services   = state.get("total_services", 5)

    return {
        "reward":             round(total_reward, 4),
        "steps":              step_count,
        "success":            healthy_services / max(total_services, 1),
        "healthy_services":   healthy_services,
        "breaker_difficulty": breaker_difficulty,
        "actions_taken":      actions_taken,
        "oncall_paged":       oncall_paged,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("  ILYUSHIN — GRPO ADVERSARIAL TRAINING")
    print(f"  Responder (QLoRA+GRPO / {MODEL_NAME}) vs Breaker (Llama 70B)")
    print(f"  Group size: {GROUP_SIZE}  |  Episodes/phase: {EPISODES_PER_PHASE}")
    print(f"  Candidate scoring: local snapshot + compute_reward (no extra env calls)")
    print("=" * 80)
    print(f"[TRAIN] Checkpoints -> {CHECKPOINTS_DIR.resolve()}")
    print(f"[TRAIN] Logs        -> {LOGS_DIR.resolve()}")
    print(f"[TRAIN] Plots       -> {PLOTS_DIR.resolve()}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    print_gpu_memory("after optimizer init")

    logger     = AdversarialLogger(LOGS_DIR)
    curriculum = CurriculumManager()

    print("\n[BASELINE] Running random agent baselines...")
    for task_id in ["easy", "medium", "hard"]:
        avg_r, avg_s, avg_st = run_baseline(ENV_URL, task_id, n_episodes=BASELINE_EPISODES)
        logger.log_baseline(task_id, avg_r, avg_s, avg_st, BASELINE_EPISODES)

    global_episode = 0
    phase_episodes = defaultdict(list)

    for phase in ["easy", "medium", "hard"]:
        print(f"\n{'=' * 80}")
        print(f"  PHASE: {phase.upper()}")
        print(f"  Group size={GROUP_SIZE}  Episodes={EPISODES_PER_PHASE}  "
              f"Candidates/step={GROUP_SIZE} (scored locally, 0 extra env calls)")
        print(f"{'=' * 80}")

        phase_start      = global_episode
        phase_rewards    = []
        phase_successes  = []
        phase_steps_list = []

        for ep in range(EPISODES_PER_PHASE):
            print(f"\n[ROLLOUT] Phase={phase} Episode={ep+1}/{EPISODES_PER_PHASE} "
                  f"Global={global_episode}")

            debug_this_episode = (ep < 2)

            episode_data = collect_grpo_episode(
                env_url=ENV_URL,
                model=model,
                tokenizer=tokenizer,
                task_id=phase,
                episode=global_episode,
                debug=debug_this_episode,
            )

            all_groups = episode_data["all_groups"]
            n_groups   = len(all_groups)

            if n_groups == 0:
                print(f"[GRPO] No groups collected, skipping update")
                global_episode += 1
                free_memory()
                continue

            # GRPO update
            total_loss = 0.0
            for grpo_epoch in range(GRPO_EPOCHS):
                epoch_loss = compute_grpo_loss(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    groups=all_groups,
                    debug=debug_this_episode,
                )
                total_loss += epoch_loss

            avg_loss    = total_loss / max(GRPO_EPOCHS, 1)
            mean_reward = episode_data["total_reward"] / max(episode_data["steps"], 1)

            print(f"[GRPO] groups={n_groups} steps={episode_data['steps']} "
                  f"loss={avg_loss:.4f} mean_step_reward={mean_reward:.3f} "
                  f"success={episode_data['success_rate']:.2%} "
                  f"actions={episode_data['actions_taken']}")

            free_memory()

            logger.log_episode(
                episode=global_episode,
                task_id=phase,
                total_reward=episode_data["total_reward"],
                total_steps=episode_data["steps"],
                success_rate=episode_data["success_rate"],
                healthy_services=episode_data["healthy_services"],
                total_services=5,
                breaker_difficulty=episode_data["breaker_difficulty"],
                actions_taken=episode_data["actions_taken"],
                oncall_paged=episode_data["oncall_paged"],
            )

            phase_rewards.append(episode_data["total_reward"])
            phase_successes.append(episode_data["success_rate"])
            phase_steps_list.append(episode_data["steps"])
            phase_episodes[phase].append({
                "episode":            global_episode,
                "total_reward":       episode_data["total_reward"],
                "success_rate":       episode_data["success_rate"],
                "breaker_difficulty": episode_data["breaker_difficulty"],
            })

            curriculum.record_reward(episode_data["success_rate"], phase)

            env_feedback(ENV_URL, episode_data["session_id"], {
                "success_rate":           episode_data["success_rate"],
                "avg_steps":              episode_data["steps"],
                "avg_reward":             episode_data["total_reward"],
                "failed_on_incidents":    [],
                "succeeded_on_incidents": [],
                "unhealthy_services":     [],
            })

            global_episode += 1

            # Log breaker status every 5 episodes
            if (ep + 1) % 5 == 0:
                try:
                    breaker_res = requests.get(
                        f"{ENV_URL}/env/breaker/status/{episode_data['session_id']}",
                        timeout=5
                    )
                    if breaker_res.ok:
                        bs = breaker_res.json().get("breaker_status", breaker_res.json())
                        logger.log_breaker_status(
                            phase=phase,
                            difficulty_level=bs.get("difficulty_level", 1),
                            effective_incidents=bs.get("effective_incidents", []),
                            vulnerable_services=bs.get("vulnerable_services", []),
                            responder_success_rate=float(np.mean(phase_successes[-5:])),
                        )
                except Exception:
                    pass

        # Save checkpoint after each phase
        ckpt_path = CHECKPOINTS_DIR / f"checkpoint_{phase}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        print(f"[TRAIN] Checkpoint saved: {ckpt_path.resolve()}")

        # Eval
        print(f"\n[EVAL] Evaluating {phase} phase ({EVAL_EPISODES} episodes)...")
        eval_rewards, eval_successes, eval_steps_list, eval_breaker_difficulties = [], [], [], []

        for ev in range(EVAL_EPISODES):
            result = run_eval_episode(ENV_URL, model, tokenizer, phase, ev)
            eval_rewards.append(result["reward"])
            eval_successes.append(result["success"])
            eval_steps_list.append(result["steps"])
            eval_breaker_difficulties.append(result["breaker_difficulty"])
            print(f"[EVAL]   ep={ev+1} reward={result['reward']:.2f} "
                  f"success={result['success']:.2%} steps={result['steps']}")
            free_memory()

        avg_eval_reward  = float(np.mean(eval_rewards))
        avg_eval_success = float(np.mean(eval_successes))
        avg_eval_steps   = float(np.mean(eval_steps_list))
        avg_breaker_diff = float(np.mean(eval_breaker_difficulties))

        logger.log_phase(
            phase=phase,
            start_episode=phase_start,
            end_episode=global_episode,
            avg_reward=avg_eval_reward,
            avg_success_rate=avg_eval_success,
            avg_steps=avg_eval_steps,
            breaker_difficulty=int(avg_breaker_diff),
        )

        print(f"\n[PHASE {phase.upper()} COMPLETE]")
        print(f"  avg_reward={avg_eval_reward:.2f}  "
              f"avg_success={avg_eval_success:.2%}  "
              f"avg_steps={avg_eval_steps:.1f}  "
              f"breaker_level={avg_breaker_diff:.1f}")
        print_gpu_memory(f"end of {phase} phase")

    final_path = CHECKPOINTS_DIR / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n[TRAIN] Final model saved to {final_path.resolve()}")

    logger.flush()
    generate_plots(logger, phase_episodes, PLOTS_DIR)

    print(f"\n{'=' * 80}")
    print("  ADVERSARIAL GRPO TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"  Logs:        {LOGS_DIR.resolve()}")
    print(f"  Plots:       {PLOTS_DIR.resolve()}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR.resolve()}")

# ============================================================================
# PLOTTING
# ============================================================================

COLORS = {
    "baseline": "#d62728",
    "trained":  "#2ca02c",
    "easy":     "#1f77b4",
    "medium":   "#ff7f0e",
    "hard":     "#9467bd",
    "breaker":  "#e377c2",
}


def generate_plots(logger: AdversarialLogger, phase_episodes: dict, plots_dir: Path):
    print("\n[PLOTS] Generating visualizations...")
    phases        = ["easy", "medium", "hard"]
    baseline_data = logger.baseline_log
    trained_data  = {p["phase"]: p for p in logger.phases_log}

    # 1: Reward curve
    fig, ax = plt.subplots(figsize=(14, 6))
    for phase in phases:
        eps = phase_episodes.get(phase, [])
        if not eps:
            continue
        xs = [e["episode"] for e in eps]
        ys = [e["total_reward"] for e in eps]
        ax.plot(xs, ys, label=f"{phase.upper()} reward",
                color=COLORS[phase], linewidth=1.5, alpha=0.7)
        if len(ys) >= 5:
            smoothed = np.convolve(ys, np.ones(5) / 5, mode="valid")
            ax.plot(xs[4:], smoothed, color=COLORS[phase], linewidth=2.5,
                    linestyle="--", label=f"{phase.upper()} (smoothed)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Episode Reward", fontsize=12)
    ax.set_title("GRPO Learning Curve — Reward per Episode", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "01_reward_curve.png", dpi=150)
    plt.close()
    print("[PLOTS] Saved 01_reward_curve.png")

    # 2: Success rate
    fig, ax = plt.subplots(figsize=(14, 6))
    for phase in phases:
        eps = phase_episodes.get(phase, [])
        if not eps:
            continue
        xs = [e["episode"] for e in eps]
        ys = [e["success_rate"] * 100 for e in eps]
        ax.plot(xs, ys, color=COLORS[phase], linewidth=1.5, alpha=0.6,
                label=f"{phase.upper()}")
        if len(ys) >= 5:
            smoothed = np.convolve(ys, np.ones(5) / 5, mode="valid")
            ax.plot(xs[4:], smoothed, color=COLORS[phase], linewidth=2.5, linestyle="--")
    ax.axhline(y=100, color="green", linestyle=":", alpha=0.4, label="Perfect (100%)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("GRPO Learning Curve — Success Rate per Episode",
                 fontsize=14, fontweight="bold")
    ax.set_ylim([0, 110])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "02_success_rate.png", dpi=150)
    plt.close()
    print("[PLOTS] Saved 02_success_rate.png")

    # 3: Baseline vs Trained
    if baseline_data and trained_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        x     = np.arange(len(phases))
        width = 0.35
        b_success = [baseline_data.get(p, {}).get("avg_success_rate", 0) * 100 for p in phases]
        t_success = [trained_data.get(p,  {}).get("avg_success_rate", 0) * 100 for p in phases]
        bars1 = ax.bar(x - width/2, b_success, width, label="Random Baseline",
                       color=COLORS["baseline"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, t_success, width, label="Trained (GRPO+QLoRA)",
                       color=COLORS["trained"],  alpha=0.85, edgecolor="white")
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([p.upper() for p in phases], fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_xlabel("Task Difficulty", fontsize=12)
        ax.set_title("Before vs After GRPO Training — Success Rate",
                     fontsize=14, fontweight="bold")
        ax.set_ylim([0, 115])
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "03_baseline_vs_trained.png", dpi=150)
        plt.close()
        print("[PLOTS] Saved 03_baseline_vs_trained.png")

    # 4: Breaker escalation
    if logger.breaker_log:
        fig, ax = plt.subplots(figsize=(12, 5))
        xs = list(range(len(logger.breaker_log)))
        ys = [b["difficulty_level"] for b in logger.breaker_log]
        ax.plot(xs, ys, color=COLORS["breaker"], linewidth=2.5,
                marker="s", markersize=5, label="Breaker difficulty")
        ax.set_xlabel("Checkpoint", fontsize=11)
        ax.set_ylabel("Breaker Difficulty (1-10)", fontsize=11)
        ax.set_title("Adversarial Breaker Escalation Over Training",
                     fontsize=14, fontweight="bold")
        ax.set_ylim([0, 11])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "04_breaker_escalation.png", dpi=150)
        plt.close()
        print("[PLOTS] Saved 04_breaker_escalation.png")

    # 5: Action distribution
    action_dist = logger.action_dist_log
    if action_dist:
        fig, axes = plt.subplots(1, len(phases), figsize=(16, 5), sharey=False)
        if len(phases) == 1:
            axes = [axes]
        for ax, phase in zip(axes, phases):
            dist = action_dist.get(phase, {})
            if not dist:
                ax.set_title(f"{phase.upper()}\n(no data)")
                continue
            acts       = list(dist.keys())
            counts     = [dist[a] for a in acts]
            colors_bar = plt.cm.tab10(np.linspace(0, 1, len(acts)))
            ax.bar(range(len(acts)), counts, color=colors_bar, edgecolor="white")
            ax.set_xticks(range(len(acts)))
            ax.set_xticklabels(acts, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Action Count", fontsize=10)
            ax.set_title(f"{phase.upper()} Phase", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Trained Agent — Action Distribution by Phase",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plots_dir / "05_action_distribution.png", dpi=150)
        plt.close()
        print("[PLOTS] Saved 05_action_distribution.png")

    # 6: Steps to resolution
    if baseline_data and trained_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        x     = np.arange(len(phases))
        width = 0.35
        b_steps = [baseline_data.get(p, {}).get("avg_steps", MAX_STEPS) for p in phases]
        t_steps = [trained_data.get(p,  {}).get("avg_steps", MAX_STEPS) for p in phases]
        bars1 = ax.bar(x - width/2, b_steps, width, label="Random Baseline",
                       color=COLORS["baseline"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, t_steps, width, label="Trained (GRPO+QLoRA)",
                       color=COLORS["trained"],  alpha=0.85, edgecolor="white")
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        f"{h:.1f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([p.upper() for p in phases], fontsize=11)
        ax.set_ylabel("Average Steps to Resolution", fontsize=12)
        ax.set_xlabel("Task Difficulty", fontsize=12)
        ax.set_title("Before vs After — Steps to Resolution (fewer = better)",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=MAX_STEPS, color="red", linestyle="--", alpha=0.5,
                   label=f"Max steps ({MAX_STEPS})")
        plt.tight_layout()
        plt.savefig(plots_dir / "06_steps_to_resolution.png", dpi=150)
        plt.close()
        print("[PLOTS] Saved 06_steps_to_resolution.png")

    print(f"[PLOTS] All plots saved to {plots_dir.resolve()}/")


if __name__ == "__main__":
    main()
