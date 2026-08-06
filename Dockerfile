# MAGNETS queue API — slim image with only the API + scheduling deps.
FROM python:3.12-slim

# Astropy/pandas build wheels are prebuilt for slim; no compiler needed.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=./data/queue.db

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the pieces the API actually needs. core/ is included so the orchestrator's
# guarded `from core.magellan_planning import RANKING_PROFILES` resolves; without
# sncosmo/matplotlib installed that import fails and is caught (profile tau falls
# back to the config default) — the scheduling path is unaffected.
COPY api/ ./api/
COPY orchestrator/ ./orchestrator/
COPY core/ ./core/
COPY ref/ ./ref/
COPY web/ ./web/

# SQLite file + demo allocations live here; mount a persistent disk at /app/data.
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
