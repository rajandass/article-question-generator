FROM python:3.9-slim AS base

# Set environment variables for model downloads
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface \
    HF_DATASETS_CACHE=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# Install system dependencies
FROM base AS builder
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM builder AS final
WORKDIR /app

# Copy application files
COPY download_models.py .
COPY app_docker.py .
COPY startup.sh .
RUN chmod +x startup.sh

# Create cache directory
RUN mkdir -p /root/.cache/huggingface

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Command to run when container starts
CMD ["./startup.sh"]