FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (supports vendored wheels for air-gap)
COPY requirements.txt .
# If vendored wheels exist, use them (air-gapped); otherwise pip install normally
COPY vendor/ /app/vendor/
RUN if [ -d /app/vendor ] && [ "$(ls -A /app/vendor 2>/dev/null)" ]; then \
      pip install --no-index --find-links=/app/vendor -r requirements.txt; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application
COPY . .

# Expose harness + indexer ports
EXPOSE 8091 9090

# Run harness server
CMD ["python", "server.py", "--port", "8091", "--host", "0.0.0.0"]
