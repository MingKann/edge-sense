# ─── Stage 1: Base ──────────────────────────────────────
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Stage 2: Dependencies ──────────────────────────────
# 利用 Docker 层缓存：requirements.txt 不变则复用层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Stage 3: App ───────────────────────────────────────
COPY src/    src/
COPY web/    web/

# OLLAMA_URL 可通过环境变量覆盖（默认指向宿主机 Ollama）
ENV OLLAMA_URL=http://host.docker.internal:11434

EXPOSE 8000

CMD ["python", "src/server.py"]
