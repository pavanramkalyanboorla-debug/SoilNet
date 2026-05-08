# syntax=docker/dockerfile:1

# =====================================================
# STAGE 1: Dependency installation
# =====================================================
FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS deps

WORKDIR /app

# System dependencies required for PIL/OpenCV/TensorFlow runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (better layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into .venv
RUN --mount=from=uv-stage,source=/uv,target=/bin/uv \
    uv sync --no-dev --frozen


# =====================================================
# STAGE 2: Runtime image
# =====================================================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from deps stage
COPY --from=deps /app/.venv /app/.venv

# -----------------------------------------------------
# Runtime artifacts only
# -----------------------------------------------------
COPY artifacts/model.keras ./artifacts/model.keras
COPY artifacts/class_names.pkl ./artifacts/class_names.pkl

# -----------------------------------------------------
# Application code only
# -----------------------------------------------------
COPY app/ app/
COPY src/pipeline/ src/pipeline/
COPY src/utils/ src/utils/
COPY src/constants.py src/constants.py
COPY src/__init__.py src/__init__.py

# Streamlit config
COPY .streamlit/ .streamlit/

# Environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# =====================================================
# Launch Streamlit
# =====================================================
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]