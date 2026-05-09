# syntax=docker/dockerfile:1

# ------------------------------------------------------------
# Stage 1: Dependencies (cached until pyproject.toml / uv.lock change)
# ------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS deps
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=from=uv-stage,source=/uv,target=/bin/uv \
    uv sync --no-dev --frozen

# ------------------------------------------------------------
# Stage 2: Final lightweight image
# ------------------------------------------------------------
FROM python:3.10-slim AS runtime
WORKDIR /app

# Copy the ready-to-use virtual environment
COPY --from=deps /app/.venv /app/.venv

# Pre‑trained artifacts – only the files actually needed at inference
COPY artifacts/model_weights.weights.h5 ./artifacts/model_weights.weights.h5
COPY artifacts/class_names.pkl ./artifacts/class_names.pkl

# Application source code
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false", \
     "--server.enableXsrfProtection=false"]