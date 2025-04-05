FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /root/.cache/huggingface

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app_v2.py", "--server.address", "0.0.0.0"]
