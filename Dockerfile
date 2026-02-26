# =============================================================================
# LLM Guardrail Studio - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized image size
# Supports both CPU and GPU deployments
# =============================================================================

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt


# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GUARDRAIL_LOG_FILE=/app/logs/guardrails.log \
    GUARDRAIL_DATABASE_URL=sqlite:////app/data/guardrails.db

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# Development stage (with additional tools)
# =============================================================================
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    mypy \
    pre-commit

USER appuser

# Enable hot reload
ENV RELOAD=true

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# =============================================================================
# Dashboard stage (Streamlit)
# =============================================================================
FROM production as dashboard

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
