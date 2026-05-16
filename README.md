# Ilyushin: Multi-Agent Adversarial Incident Escalation for Autonomous Recovery

An OpenEnv-compatible reinforcement learning environment where AI agents learn to resolve production infrastructure incidents under adversarial pressure. A Responder agent is trained via GRPO to fix failing services while a Breaker agent continuously generates harder, more deceptive failures.

## Overview

Production incidents are expensive and time-sensitive. Ilyushin trains LLMs to autonomously diagnose and resolve infrastructure failures across five interconnected services. What makes this environment distinct is the adversarial dynamic: a second LLM (the Breaker) observes the Responder's performance and escalates its attack strategy accordingly, injecting cascades, red herrings, and simultaneous multi-service failures as the Responder improves.

This creates a genuine multi-agent training loop where both sides are forced to adapt.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Server starts at `http://localhost:8000`

### Docker

```bash
docker build -t ilyushin .
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  -e API_KEY=your_token \
  ilyushin
```

### Environment Variables

```bash
export BASE_URL="http://localhost:8000"
export HF_TOKEN="your_huggingface_token"
export API_KEY="your_huggingface_token"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/env/reset` | Start a new episode |
| POST | `/env/step` | Take an action |
| GET | `/env/state/{session_id}` | Get current state |
| POST | `/env/feedback` | Send performance feedback to Breaker |
| GET | `/env/breaker/status/{session_id}` | Get Breaker difficulty status |
| GET | `/tasks` | List all tasks |
| POST | `/grader/` | Grade a completed episode |
| GET | `/baseline` | Run random baseline agent |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `infrastructure` | dict | All 5 services with CPU, memory, latency, error rate, RPS metrics |
| `active_incidents` | list | Currently active incidents with type, severity, targets |
| `healthy_services` | int | Number of currently healthy services |
| `total_services` | int | Total services (always 5) |
| `last_action_result` | string | Outcome of the last action taken |
| `last_action_success` | bool | Whether the last action succeeded |
| `oncall_paged` | bool | Whether on-call engineer has been paged |
| `step_count` | int | Steps taken in this episode |
| `done` | bool | Whether the episode has ended |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | One of: `read_logs`, `check_metrics`, `restart_service`, `scale_up`, `rollback`, `page_oncall`, `resolve` |
| `target_service` | string or null | One of: `web_server`, `database`, `cache`, `queue`, `api_gateway` |

```json
{"type": "restart_service", "target_service": "database"}
{"type": "read_logs"}
{"type": "scale_up", "target_service": "web_server"}
```

## Reward System

| Event | Reward |
|-------|--------|
| Each step | -0.1 |
| Service recovered | +3.0 per service |
| Service degraded | -1.0 per service |
| Wrong action on a service | -0.5 |
| Incident cleared | +2.0 per incident |
| All services resolved | +10.0 bonus |
| Fast resolution (under 10 steps) | +3.0 bonus |
| Paging on-call | -5.0 |

## Tasks

| Task | Description |
|------|-------------|
| Easy | Single service crash. Identify and restart the failed cache service. No cascading effects. |
| Medium | Two-service failure with dependency chain. Database crash cascading into API gateway degradation. |
| Hard | Multi-service cascade with red herrings. Simultaneous P1 incidents across 3+ services with misleading metrics. |

## Agents

**Responder** — the agent being trained. Receives infrastructure state and outputs a JSON action each step.

**Breaker** — adversarial agent (Llama 3.3-70B) that generates incidents at the start of each episode. Observes Responder performance via the `/env/feedback` endpoint and escalates difficulty (1-10) as the Responder improves. Learns which incident types and target services the Responder struggles with most.

**Monitor** — analyzes infrastructure state and provides diagnostic context to the Responder each step.

## Training

Training uses GRPO on the Responder model. The Breaker adapts difficulty in real time based on Responder performance.

```bash
cd training
export BASE_URL="http://localhost:8000"
export HF_TOKEN="your_token"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
python train.py
```

The training script runs three curriculum phases (easy, medium, hard), saves checkpoints after each phase, and generates reward and performance plots on completion.

## Inference

```bash
export HF_TOKEN="your_token"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
python inference.py
```

Runs the trained agent through one episode each of easy, medium, and hard difficulty, with Monitor analysis at each step.

## Links

- HuggingFace Space: https://huggingface.co/spaces/akkiisfrommars/ilyushin
- GitHub: https://github.com/vedkde/Ilyushin
