FROM python:3.9-slim

WORKDIR /app

# Combine system deps and pip install in single layer to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    # Remove apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages without caching wheels
# --no-cache-dir reduces image size by not storing package cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/*

# Set environment variables for model downloads
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# Copy only necessary files (changes most frequently)
COPY download_models.py .
COPY app_docker.py .

# Create cache directory and download models
RUN mkdir -p /root/.cache/huggingface && \
    python download_models.py

COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 8501

# Run startup script instead of direct streamlit command
CMD ["./startup.sh"]
