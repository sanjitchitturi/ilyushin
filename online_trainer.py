#!/usr/bin/env python3
"""
online_trainer.py — Continuous Online Learning from Deployment
==============================================================

Runs alongside inference.py as a background process.
Watches inference.py stdout via a shared log file, reconstructs
episode trajectories from [START]/[STEP]/[END] log lines, and
runs GRPO weight updates after each completed episode.

Model improves continuously from real production incidents.

USAGE:
    # Terminal 1 — run inference and pipe stdout to shared log
    python inference.py 2>&1 | tee /tmp/ilyushin_inference.log

    # Terminal 2 — run online trainer watching that log
    python online_trainer.py

    # Or run both together:
    python inference.py 2>&1 | tee /tmp/ilyushin_inference.log &
    python online_trainer.py

ENV VARS:
    MODEL_NAME          model to load for weight updates
    HF_TOKEN / API_KEY  HuggingFace token
    INFERENCE_LOG       path to inference stdout log (default: /tmp/ilyushin_inference.log)
    CHECKPOINT_DIR      where to save updated weights   (default: ./online_checkpoints)
    UPDATE_EVERY        episodes between weight updates  (default: 1)
    MIN_REWARD_TO_LEARN only learn from episodes above this reward threshold (default: -999)
"""

import os
import sys
import gc
import json
import time
import copy
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

# ── Config ────────────────────────────────────────────────────────

MODEL_NAME     = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
INFERENCE_LOG  = os.getenv("INFERENCE_LOG", "/tmp/ilyushin_inference.log")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./online_checkpoints"))
UPDATE_EVERY   = int(os.getenv("UPDATE_EVERY", "1"))       # update after every N episodes
MIN_REWARD     = float(os.getenv("MIN_REWARD_TO_LEARN", "-999"))  # skip low-quality episodes

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LEARNING_RATE  = 1e-5
MAX_NEW_TOKENS = 32
MIN_NEW_TOKENS = 4
MAX_PROMPT_LENGTH = 512
POLL_INTERVAL  = 2.0   # seconds between log file checks

VALID_ACTIONS  = ["read_logs", "check_metrics", "restart_service",
                  "scale_up", "rollback", "page_oncall", "resolve"]
VALID_SERVICES = ["web_server", "database", "cache", "queue", "api_gateway"]

# Reward constants — mirrors env/reward.py
STEP_PENALTY            = -0.1
SERVICE_RECOVERED       =  3.0
ALL_RESOLVED_BONUS      = 10.0
FAST_RESOLUTION_BONUS   =  3.0
ONCALL_PENALTY          = -5.0
CASCADE_BONUS           =  2.0
FAST_THRESHOLD          = 10

LINE = "=" * 64
DASH = "-" * 64


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[TRAINER {ts}] {msg}", flush=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(model_name: str):
    log(f"Loading tokenizer: {model_name}")
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

    log(f"Loading model with QLoRA: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(base_model, lora_config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    log(f"Model on: {next(model.parameters()).device}")
    return model, tokenizer


# ============================================================================
# LOG PARSER
# Reconstructs episode trajectories from inference.py stdout lines.
#
# inference.py emits:
#   [START] task=<id> env=<benchmark> model=<name>
#   [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<str|null>
#   [END]   success=<bool> steps=<n> score=<float> rewards=<r1,r2,...>
# ============================================================================

class EpisodeParser:
    """
    Stateful parser that tails the inference log file and emits
    complete Episode objects whenever it sees a full START→STEP*→END block.
    """

    def __init__(self, log_path: str):
        self.log_path       = log_path
        self.file_handle    = None
        self.current        = None   # episode being assembled
        self.completed      = deque()

    def open(self):
        # Wait for the log file to appear
        while not Path(self.log_path).exists():
            log(f"Waiting for inference log at {self.log_path} ...")
            time.sleep(3)
        self.file_handle = open(self.log_path, "r")
        # Seek to end so we only process new lines going forward
        self.file_handle.seek(0, 2)
        log(f"Watching inference log: {self.log_path}")

    def poll(self):
        """Read any new lines and update parser state."""
        if self.file_handle is None:
            self.open()

        while True:
            line = self.file_handle.readline()
            if not line:
                break
            self._process_line(line.strip())

    def _process_line(self, line: str):
        if line.startswith("[START]"):
            self._parse_start(line)
        elif line.startswith("[STEP]"):
            self._parse_step(line)
        elif line.startswith("[END]"):
            self._parse_end(line)

    def _parse_start(self, line: str):
        # [START] task=easy env=ilyushin model=meta-llama/...
        parts = {}
        for token in line[len("[START]"):].strip().split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        self.current = {
            "task_id": parts.get("task", "unknown"),
            "model":   parts.get("model", MODEL_NAME),
            "steps":   [],
        }

    def _parse_step(self, line: str):
        if self.current is None:
            return
        # [STEP] step=1 action=restart_service(web_server) reward=17.90 done=false error=null
        parts = {}
        for token in line[len("[STEP]"):].strip().split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        try:
            self.current["steps"].append({
                "step":   int(parts.get("step", 0)),
                "action": parts.get("action", "unknown"),
                "reward": float(parts.get("reward", 0.0)),
                "done":   parts.get("done", "false") == "true",
                "error":  parts.get("error", "null"),
            })
        except Exception:
            pass

    def _parse_end(self, line: str):
        if self.current is None:
            return
        # [END] success=true steps=2 score=1.00 rewards=-0.10,17.90
        parts = {}
        for token in line[len("[END]"):].strip().split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v

        rewards_raw = parts.get("rewards", "")
        try:
            rewards = [float(r) for r in rewards_raw.split(",") if r]
        except Exception:
            rewards = []

        self.current["success"]      = parts.get("success", "false") == "true"
        self.current["total_steps"]  = int(parts.get("steps", 0))
        self.current["score"]        = float(parts.get("score", 0.0))
        self.current["total_reward"] = sum(rewards)
        self.current["rewards"]      = rewards

        self.completed.append(self.current)
        log(f"Episode parsed: task={self.current['task_id']} "
            f"score={self.current['score']:.2f} "
            f"reward={self.current['total_reward']:.2f} "
            f"steps={self.current['total_steps']} "
            f"success={self.current['success']}")
        self.current = None

    def pop_episode(self):
        """Return next completed episode or None."""
        if self.completed:
            return self.completed.popleft()
        return None

    def close(self):
        if self.file_handle:
            self.file_handle.close()


# ============================================================================
# TRAJECTORY BUILDER
# Reconstructs (prompt_tensor, response_tensor, reward) triples from
# episode data so we can run GRPO updates.
#
# Since we don't have the raw token tensors from inference time, we
# regenerate GROUP_SIZE responses for each action position and score
# them using the step rewards from the log. The best match to the
# actual action taken gets its reward assigned; others get estimated
# rewards via local prediction. This gives us a valid GRPO group.
# ============================================================================

from dataset import SYSTEM_PROMPT, format_prompt
from env.state import EnvState
from env.reward import compute_reward


def build_minimal_state(task_id: str, step_idx: int, action_str: str, reward: float) -> dict:
    """
    Build a minimal state dict from the action string and reward.
    We don't have the full infrastructure snapshot at inference time,
    so we reconstruct a plausible state from what we do know.
    """
    # Parse action string like "restart_service(web_server)"
    action_type   = action_str.split("(")[0] if "(" in action_str else action_str
    target_service = None
    if "(" in action_str and ")" in action_str:
        target_service = action_str.split("(")[1].rstrip(")")

    # Infer healthy_services from reward signal
    # SERVICE_RECOVERED_REWARD = 3.0 per service recovered
    # ALL_RESOLVED_BONUS = 10.0
    services_recovered = 0
    remaining_reward   = reward - STEP_PENALTY
    if remaining_reward >= ALL_RESOLVED_BONUS:
        remaining_reward -= ALL_RESOLVED_BONUS
        if remaining_reward >= FAST_RESOLUTION_BONUS:
            remaining_reward -= FAST_RESOLUTION_BONUS
    if remaining_reward > 0:
        services_recovered = round(remaining_reward / SERVICE_RECOVERED)

    # Build a plausible infrastructure state
    infra = {}
    for svc in VALID_SERVICES:
        if svc == target_service and services_recovered > 0:
            infra[svc] = {"status": "healthy", "metrics": {}, "overloaded": False}
        else:
            infra[svc] = {"status": "healthy", "metrics": {}, "overloaded": False}

    # If this was a corrective action on a specific service, mark it as
    # previously degraded to give the state some meaning
    if target_service and action_type in ("restart_service", "scale_up", "rollback"):
        infra[target_service] = {"status": "degraded", "metrics": {}, "overloaded": False}

    return {
        "task_id":            task_id,
        "step_count":         step_idx,
        "done":               False,
        "infrastructure":     infra,
        "active_incidents":   [],
        "healthy_services":   max(0, 5 - max(1, len([s for s in infra.values()
                                                      if s["status"] != "healthy"]))),
        "total_services":     5,
        "oncall_paged":       False,
        "last_action":        action_type,
        "last_action_result": "",
        "last_action_success": True,
        "services_resolved":  [],
    }


def tokenize_state(tokenizer, state: dict) -> torch.Tensor:
    text = f"{SYSTEM_PROMPT}\n\n{format_prompt(state)}"
    enc  = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    )
    return enc["input_ids"].squeeze(0)


def generate_response(model, tokenizer, query_tensor: torch.Tensor) -> torch.Tensor:
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


def tokenize_action(tokenizer, action_str: str) -> torch.Tensor:
    """Tokenize an action string as if it were the model's response."""
    action_type = action_str.split("(")[0] if "(" in action_str else action_str
    target      = None
    if "(" in action_str and ")" in action_str:
        target = action_str.split("(")[1].rstrip(")")

    json_str = json.dumps({"type": action_type, "target_service": target})
    enc = tokenizer(
        json_str,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return enc["input_ids"].squeeze(0)


def build_grpo_groups(episode: dict, model, tokenizer, group_size: int = 4) -> list:
    """
    Convert a parsed episode into GRPO training groups.

    For each step in the episode:
    1. Build a minimal state from the action + reward info we have
    2. Tokenize it as a prompt
    3. Use the actual action taken (tokenized) as one group member with its real reward
    4. Generate (group_size - 1) additional candidates with estimated rewards
    5. Return the group for GRPO update
    """
    groups = []

    for step_data in episode["steps"]:
        action_str = step_data["action"]
        real_reward = step_data["reward"]
        step_idx    = step_data["step"]

        # Skip steps with errors
        if step_data.get("error") and step_data["error"] != "null":
            continue

        # Build state and tokenize prompt
        state        = build_minimal_state(episode["task_id"], step_idx, action_str, real_reward)
        query_tensor = tokenize_state(tokenizer, state)

        # Member 0: the actual action taken with its real reward
        actual_response = tokenize_action(tokenizer, action_str)
        responses = [actual_response.cpu()]
        rewards   = [real_reward]

        # Members 1..group_size-1: generated candidates with estimated rewards
        for _ in range(group_size - 1):
            try:
                gen_ids     = generate_response(model, tokenizer, query_tensor)
                gen_text    = tokenizer.decode(gen_ids, skip_special_tokens=True)
                # Estimate reward: penalise read_logs spam, reward corrective actions
                est_reward  = STEP_PENALTY
                if any(a in gen_text for a in ("restart", "scale_up", "rollback")):
                    est_reward += SERVICE_RECOVERED * 0.5  # optimistic estimate
                elif "read_logs" in gen_text:
                    est_reward -= 0.5  # spam penalty
                responses.append(gen_ids.cpu())
                rewards.append(est_reward)
            except Exception:
                continue

        # Need at least 2 members for normalization
        if len(responses) >= 2:
            groups.append({
                "query_tensor": query_tensor.cpu(),
                "responses":    responses,
                "rewards":      rewards,
            })

    return groups


# ============================================================================
# GRPO UPDATE
# ============================================================================

def grpo_update(model, optimizer, groups: list, debug: bool = False) -> float:
    """
    Run one GRPO gradient update over the provided groups.
    Identical to the implementation in train.py.
    """
    if not groups:
        return 0.0

    device     = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_valid    = 0

    for group in groups:
        query_tensor = group["query_tensor"].to(device)
        responses    = group["responses"]
        rewards      = group["rewards"]

        if len(rewards) < 2:
            continue

        rewards_t  = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r     = rewards_t.mean()
        std_r      = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r

        for resp_tensor, adv in zip(responses, advantages):
            if resp_tensor.numel() == 0:
                continue

            full_ids = torch.cat(
                [query_tensor, resp_tensor.to(device)], dim=0
            ).unsqueeze(0)

            labels = full_ids.clone()
            labels[0, :query_tensor.shape[0]] = -100

            outputs    = model(input_ids=full_ids, labels=labels)
            token_loss = outputs.loss

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
# CHECKPOINT
# ============================================================================

def save_checkpoint(model, tokenizer, episode_count: int):
    ckpt = CHECKPOINT_DIR / f"online_ep{episode_count:04d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    log(f"Checkpoint saved: {ckpt.resolve()}")

    # Also overwrite 'latest' symlink / folder
    latest = CHECKPOINT_DIR / "latest"
    if latest.exists():
        import shutil
        shutil.rmtree(latest)
    model.save_pretrained(str(latest))
    tokenizer.save_pretrained(str(latest))


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# STATS TRACKER
# ============================================================================

class OnlineStats:
    def __init__(self):
        self.episodes_processed = 0
        self.episodes_skipped   = 0
        self.total_updates      = 0
        self.total_groups       = 0
        self.losses             = []
        self.rewards            = []
        self.scores             = []

    def record_episode(self, episode: dict, loss: float, n_groups: int):
        self.episodes_processed += 1
        self.total_updates      += 1
        self.total_groups       += n_groups
        self.losses.append(loss)
        self.rewards.append(episode["total_reward"])
        self.scores.append(episode["score"])

    def print_summary(self):
        if not self.losses:
            return
        recent = self.losses[-10:]
        print(f"\n{DASH}")
        print(f"  ONLINE TRAINER — Stats")
        print(f"  Episodes processed : {self.episodes_processed}")
        print(f"  Episodes skipped   : {self.episodes_skipped}")
        print(f"  Total updates      : {self.total_updates}")
        print(f"  Avg loss (last 10) : {sum(recent)/len(recent):.4f}")
        print(f"  Avg score (all)    : {sum(self.scores)/len(self.scores):.2f}")
        print(f"  Avg reward (all)   : {sum(self.rewards)/len(self.rewards):.2f}")
        print(f"{DASH}\n")


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print(f"\n{LINE}")
    print(f"  ILYUSHIN — ONLINE TRAINER")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Watching    : {INFERENCE_LOG}")
    print(f"  Checkpoints : {CHECKPOINT_DIR.resolve()}")
    print(f"  Update every: {UPDATE_EVERY} episode(s)")
    print(f"  Min reward  : {MIN_REWARD}")
    print(f"{LINE}\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Parser and stats
    parser = EpisodeParser(INFERENCE_LOG)
    stats  = OnlineStats()

    # Episode buffer — accumulate UPDATE_EVERY episodes before updating
    episode_buffer = []

    log("Online trainer running. Waiting for inference episodes...")
    log("(Make sure inference.py stdout is being piped to the log file)")

    try:
        while True:
            # Poll log file for new lines
            parser.poll()

            # Process any completed episodes
            while True:
                episode = parser.pop_episode()
                if episode is None:
                    break

                # Skip low-quality episodes
                if episode["total_reward"] < MIN_REWARD:
                    log(f"Skipping episode (reward={episode['total_reward']:.2f} < {MIN_REWARD})")
                    stats.episodes_skipped += 1
                    continue

                episode_buffer.append(episode)
                log(f"Buffered episode {len(episode_buffer)}/{UPDATE_EVERY} "
                    f"(task={episode['task_id']} reward={episode['total_reward']:.2f})")

                # Run update when buffer is full
                if len(episode_buffer) >= UPDATE_EVERY:
                    log(f"Running GRPO update on {len(episode_buffer)} episode(s)...")

                    all_groups = []
                    for ep in episode_buffer:
                        groups = build_grpo_groups(ep, model, tokenizer, group_size=4)
                        all_groups.extend(groups)
                        log(f"  Built {len(groups)} groups from episode "
                            f"task={ep['task_id']} steps={ep['total_steps']}")

                    if all_groups:
                        loss = grpo_update(model, optimizer, all_groups)
                        log(f"Update complete: loss={loss:.4f} groups={len(all_groups)}")

                        for ep in episode_buffer:
                            stats.record_episode(ep, loss, len(all_groups))
                    else:
                        log("No groups built, skipping update")

                    free_memory()

                    # Save checkpoint
                    save_checkpoint(model, tokenizer, stats.episodes_processed)

                    # Print stats
                    stats.print_summary()

                    # Clear buffer
                    episode_buffer.clear()

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        log("Shutting down online trainer...")
        stats.print_summary()

        # Save final checkpoint
        if stats.episodes_processed > 0:
            save_checkpoint(model, tokenizer, stats.episodes_processed)
            log("Final checkpoint saved.")

        parser.close()
        log("Done.")


if __name__ == "__main__":
    # Ensure workspace root is on path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
