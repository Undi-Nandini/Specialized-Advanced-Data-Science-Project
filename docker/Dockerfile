# ─── Stage 1: dependency builder ────────────────────────────────────────────
FROM python:3.11-alpine AS builder

RUN apk add --no-cache gcc musl-dev
WORKDIR /deps
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: lean runtime image ────────────────────────────────────────────
FROM python:3.11-alpine

LABEL org.opencontainers.image.title="Review Sentiment API"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="GRU-based NLP sentiment analysis service"

WORKDIR /service

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application source
COPY src/  ./src/
COPY data/ ./data/

# Create runtime directories
RUN mkdir -p src/network/saved monitoring/logs logs \
    && adduser -D -u 1001 svcuser \
    && chown -R svcuser:svcuser /service

USER svcuser

# Configuration via environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_DIR=src/network/saved
ENV HOST=0.0.0.0
ENV PORT=8080
ENV WORKERS=2

EXPOSE 8080

# Startup health check
HEALTHCHECK --interval=25s --timeout=8s --start-period=45s --retries=3 \
    CMD wget -qO- http://localhost:8080/status || exit 1

# Entrypoint
CMD ["sh", "-c", "uvicorn src.server.api:app --host $HOST --port $PORT --workers $WORKERS"]
