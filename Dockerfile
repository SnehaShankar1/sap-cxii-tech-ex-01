# syntax=docker/dockerfile:1
#
# Multi-stage build for the product similarity service.
#
# Changes from the Dockerfile supplied with the exercise, and why:
#
#   1. requirements.txt is copied and installed BEFORE the source. The
#      original `COPY . /app` preceded `pip install`, so any source edit
#      invalidated the pip layer and reinstalled every dependency - a
#      multi-minute rebuild on every one-line change.
#
#   2. `ENV NAME ProductSimilarityApp` used the space-separated form, which
#      Docker has deprecated. Now `ENV KEY=value`.
#
#   3. Runs as a non-root user. Many Kubernetes clusters enforce
#      `runAsNonRoot: true` via Pod Security Standards and will refuse to
#      schedule a root container outright.
#
#   4. Multi-stage: build tools (needed to compile hnswlib) stay in the
#      builder and never reach the runtime image.
#
#   5. HEALTHCHECK added, matching the Kubernetes liveness probe.

# ---------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# hnswlib ships a C++ extension and needs a compiler to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NAME=ProductSimilarityApp \
    DATASET_PATH=/app/data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson

COPY --from=builder /opt/venv /opt/venv

# System user with no login shell and no home directory to write to.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/

USER appuser

EXPOSE 8000

# Hits the liveness endpoint, which stays 200 even while the index builds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Single worker on purpose: each worker builds its own ~140 MB feature matrix
# and HNSW graph, so N workers cost N times the memory and N times the ~30s
# startup. Scale with replicas rather than workers, and let the index be
# shared per-pod. See README for the alternative (prebuilt index on a volume).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
