# Stage 1: Build React frontend
FROM node:22-slim AS frontend
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --production=false
COPY ui/ ./
RUN npm run build

# Stage 2: Python backend + built frontend
FROM python:3.12-slim AS backend
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# Copy built frontend static files (vite outputs to src/llm_forge/ui/static)
COPY --from=frontend /app/src/llm_forge/ui/static src/llm_forge/ui/static/

# Install Python package with UI dependencies
RUN pip install --no-cache-dir ".[ui]"

# Create data directory for JSON stores
RUN mkdir -p /app/data

EXPOSE 8888

ENV FORGE_CORS_ORIGINS="http://localhost:8888"
ENV FORGE_AUTH_ENABLED="false"

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8888/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "llm_forge.ui.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8888"]
