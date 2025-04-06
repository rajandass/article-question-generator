import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
from pathlib import Path
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_model_cache(model_name: str, cache_dir: str) -> bool:
    """Verify if model files exist in cache and are valid"""
    model_path = Path(cache_dir) / model_name
    return (model_path / "config.json").exists()

def test_connection(test_url="https://huggingface.co", timeout=5):
    """Test connection to specified URL with proxy handling"""
    proxies = {}
    
    # Check for proxy environment variables
    for var in ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy']:
        if var in os.environ and os.environ[var]:
            proxy_val = os.environ[var]
            logger.info(f"Found proxy setting: {var}={proxy_val}")
            protocol = var.lower().split('_')[0]
            proxies[protocol] = proxy_val
    
    try:
        logger.info(f"Testing connection to {test_url}...")
        response = requests.get(test_url, timeout=timeout, proxies=proxies)
        logger.info(f"✓ Connection successful: Status code {response.status_code}")
        return True
    except Exception as e:
        logger.warning(f"✗ Connection error: {e}")
        return False

def download_model_with_retry(model_name, cache_dir, model_type="model", max_retries=3):
    """Download model with retry logic"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Downloading {model_name} to {cache_dir} (device: {device})")
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Use appropriate tokenizer class
            if "t5" in model_name.lower():
                tokenizer = T5Tokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    local_files_only=False
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            # Force save to specific location
            model_path = Path(cache_dir) / model_name
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(str(model_path))
            model.save_pretrained(str(model_path))
            
            elapsed_time = time.time() - start_time
            
            # Verify download
            if verify_model_cache(model_name, cache_dir):
                logger.info(f"✓ Successfully cached {model_name} in {elapsed_time:.1f}s")
                return True
            else:
                logger.error(f"✗ Failed to verify cache for {model_name}")
                
        except Exception as e:
            logger.error(f"✗ Error downloading {model_name} (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
            
    logger.error(f"✗ Failed to download {model_name} after {max_retries} attempts")
    return False

def download_models():
    logger.info("Starting model download process...")
    
    # Configure HF settings
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['TRANSFORMERS_OFFLINE'] = "0"
    
    # Test connection
    connection_ok = test_connection()
    
    # Skip downloads if offline and cached
    if not connection_ok:
        logger.warning("No internet connection available. Will use cached models if available.")
        if os.environ.get('REQUIRE_DOWNLOADS', 'false').lower() == 'true':
            logger.error("Internet connection required but not available. Exiting.")
            return 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Ensure cache directory exists
    cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE', "/root/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    # List of required models
    models = {
        "summarization": [
            "facebook/bart-large-cnn",
            "facebook/bart-large-xsum"
        ],
        "question": [
            "mrm8488/t5-base-finetuned-question-generation-ap"
        ]
    }
    
    # Optional models - only download if specifically requested
    if os.environ.get('DOWNLOAD_ALL_MODELS', 'false').lower() == 'true':
        models["summarization"].append("google/pegasus-xsum")
        models["question"].append("t5-small")
    
    total_models = sum(len(model_list) for model_list in models.values())
    models_processed = 0
    models_succeeded = 0
    required_models_succeeded = 0
    required_models_total = 0
    
    for category, model_list in models.items():
        for model_name in model_list:
            models_processed += 1
            is_required = (category == "summarization" and model_name in ["facebook/bart-large-cnn", "facebook/bart-large-xsum"]) or \
                         (category == "question" and model_name == "mrm8488/t5-base-finetuned-question-generation-ap")
            
            if is_required:
                required_models_total += 1
                
            logger.info(f"\n[{models_processed}/{total_models}] Processing {model_name}...")
            
            # Skip download if already cached properly and not forced
            if verify_model_cache(model_name, cache_dir) and not os.environ.get('FORCE_DOWNLOAD', 'false').lower() == 'true':
                logger.info(f"✓ Model {model_name} already cached")
                models_succeeded += 1
                if is_required:
                    required_models_succeeded += 1
                continue
            
            # Skip download if offline
            if not connection_ok:
                logger.warning(f"Skipping download of {model_name} - no internet connection")
                continue
            
            # Download with retry
            if download_model_with_retry(model_name, cache_dir):
                models_succeeded += 1
                if is_required:
                    required_models_succeeded += 1
    
    # Report overall success
    if models_succeeded == total_models:
        logger.info(f"\n✅ All {total_models} models successfully downloaded and cached")
        return 0
    elif required_models_succeeded == required_models_total:
        logger.info(f"\n⚠️ {models_succeeded}/{total_models} models successfully cached")
        logger.info(f"✅ All {required_models_succeeded} required models are available")
        return 0
    else:
        logger.error(f"\n❌ {total_models - models_succeeded} models failed, including required models")
        return 1

if __name__ == "__main__":
    exit_code = download_models()
    sys.exit(exit_code)