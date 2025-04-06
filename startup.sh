#!/bin/bash
set -e

# Ensure cache directory exists
mkdir -p /root/.cache/huggingface

# Check if we should download models
if [ "${DOWNLOAD_MODELS}" = "true" ]; then
  echo "Model download enabled. Downloading required models..."
  # Set transformers to online mode for downloads
  export TRANSFORMERS_OFFLINE=0
  python download_models.py || {
    echo "Model download failed. Check network connection and try again."
    exit 1
  }
fi

# Set offline mode after downloads complete
export TRANSFORMERS_OFFLINE=1
echo "Setting application to offline mode. Using cached models only."

# Start the application
echo "Starting Streamlit application..."
streamlit run app_docker.py --server.address 0.0.0.0