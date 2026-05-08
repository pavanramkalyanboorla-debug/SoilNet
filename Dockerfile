# syntax=docker/dockerfile:1

# =====================================================
# STAGE 1: Dependencies (cached until pyproject.toml / uv.lock change)
# =====================================================
FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS deps

WORKDIR /app

# Copy only the files needed to install dependencies
COPY pyproject.toml uv.lock README.md ./

# Install the project in a virtual environment
RUN --mount=from=uv-stage,source=/uv,target=/bin/uv \
    uv sync --no-dev --frozen


# =====================================================
# STAGE 2: Final lightweight image
# =====================================================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy the ready‑to‑use virtual environment (no build tools)
COPY --from=deps /app/.venv /app/.venv

# Pre‑trained artifacts – required at build time
# (upload them via Hugging Face UI for the Space, or keep locally for development)
COPY artifacts/ ./artifacts/

# Application source code (changes frequently – this layer is cheap to rebuild)
COPY src/ src/
COPY app/ app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]