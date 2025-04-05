#!/bin/bash
set -e

# Ensure cache directory exists
mkdir -p /root/.cache/huggingface

# Download models if not present
python download_models.py

# Start the application
streamlit run app_docker.py --server.address 0.0.0.0
