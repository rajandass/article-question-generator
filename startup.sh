#!/bin/bash
set -e

# Ensure cache directory exists
mkdir -p /root/.cache/huggingface

# Function to check if required models are available
check_models() {
  python -c "
from pathlib import Path
required_models = [
  'facebook/bart-large-cnn', 
  'mrm8488/t5-base-finetuned-question-generation-ap', 
  'facebook/bart-large-xsum'
]
missing = [m for m in required_models if not Path(f'/root/.cache/huggingface/{m}/config.json').exists()]
if missing:
    print(f'Missing models: {missing}')
    exit(1)
else:
    print('All required models are available')
    exit(0)
  "
}

# Check if we should download models
if [ "${DOWNLOAD_MODELS}" = "true" ]; then
  echo "Model download enabled. Downloading required models..."
  # Set transformers to online mode for downloads
  export TRANSFORMERS_OFFLINE=0
  
  # Try to download models
  if python download_models.py; then
    echo "Model download completed successfully."
  else
    echo "Model download failed. Checking if required models are already cached..."
    if check_models; then
      echo "All required models are available despite download failure. Proceeding."
    else
      echo "Required models missing. Exiting."
      exit 1
    fi
  fi
else
  echo "Model download disabled. Checking if required models are already cached..."
  export TRANSFORMERS_OFFLINE=1
  
  if check_models; then
    echo "All required models are available. Proceeding."
  else
    echo "Required models missing and downloads disabled. Exiting."
    exit 1
  fi
fi

# Set offline mode after downloads complete
export TRANSFORMERS_OFFLINE=1
echo "Setting application to offline mode. Using cached models only."

# Start the application
echo "Starting Streamlit application..."
streamlit run app_docker.py --server.address 0.0.0.0