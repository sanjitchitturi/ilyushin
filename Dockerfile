# ── Ilyushin — Incident Response Environment ──────────────────────
# HuggingFace Spaces Docker deployment
# Serves the FastAPI environment server on port 7860 (HF requirement)
# ──────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# HF Spaces runs containers as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# Install dependencies as user
COPY --chown=user requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-server.txt

# Copy project source
COPY --chown=user . .

# HF Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
