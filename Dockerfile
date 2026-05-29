# ─── Stage 1: Builder ──────────────────────────────────
# 在虚拟环境中安装所有 Python 依赖，编译 C 扩展
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 创建隔离虚拟环境，避免与系统 Python 包冲突
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# ─── Stage 2: Runtime ──────────────────────────────────
FROM python:3.12-slim AS runtime

# 仅安装运行时系统依赖（无编译工具链，镜像显著缩小）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制完整虚拟环境（保留 entry points 与 .so 文件）
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY src/    src/
COPY web/    web/

# OLLAMA_URL 可通过环境变量覆盖（默认指向宿主机 Ollama）
ENV OLLAMA_URL=http://host.docker.internal:11434
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "src/server.py"]
