import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
from pathlib import Path
import requests
import time

def verify_model_cache(model_name: str, cache_dir: str) -> bool:
    """Verify if model files exist in cache and are valid"""
    model_path = Path(cache_dir) / model_name
    return (model_path / "config.json").exists()

def download_models():
    print("Starting model download process...")
    # Configure HF settings
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['TRANSFORMERS_OFFLINE'] = "0"
    
    # Test connection
    proxy_set = False
    connection_ok = False
    
    try:
        print("Testing connection to Hugging Face...")
        response = requests.get("https://huggingface.co", timeout=5)
        print("✓ Connection to Hugging Face successful")
        connection_ok = True
    except Exception as e:
        print(f"✗ Connection error: {e}")
        
        # Check if proxy environment variable is set
        if 'HTTPS_PROXY' in os.environ and os.environ['HTTPS_PROXY']:
            proxy_set = True
            print(f"Using proxy from environment: {os.environ['HTTPS_PROXY']}")
            
        # If no proxy is set, we could try a default one
        if not proxy_set and not connection_ok:
            print("No proxy set. You may need to configure a proxy if behind a firewall.")
    
    if not connection_ok and not proxy_set:
        print("Warning: No internet connection and no proxy configured.")
        print("Models will only be used if they are already cached.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ensure cache directory exists
    cache_dir = "/root/.cache/huggingface"
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
    
    for category, model_list in models.items():
        for model_name in model_list:
            models_processed += 1
            print(f"\n[{models_processed}/{total_models}] Processing {model_name}...")
            
            try:
                # Skip if already cached properly
                if verify_model_cache(model_name, cache_dir):
                    print(f"✓ Model {model_name} already cached")
                    models_succeeded += 1
                    continue
                
                # Download and cache model
                print(f"Downloading {model_name} to {cache_dir}")
                
                # Use appropriate tokenizer class
                start_time = time.time()
                
                if "t5" in model_name.lower():
                    tokenizer = T5Tokenizer.from_pretrained(
                        model_name, 
                        cache_dir=cache_dir,
                        local_files_only=False,
                        force_download=True
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        force_download=True
                    )
                
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    force_download=True
                )
                
                # Force save to specific location
                model_path = Path(cache_dir) / model_name
                os.makedirs(model_path, exist_ok=True)
                tokenizer.save_pretrained(str(model_path))
                model.save_pretrained(str(model_path))
                
                # Calculate download time
                elapsed_time = time.time() - start_time
                
                # Verify download
                if verify_model_cache(model_name, cache_dir):
                    print(f"✓ Successfully cached {model_name} in {elapsed_time:.1f}s")
                    models_succeeded += 1
                else:
                    print(f"✗ Failed to verify cache for {model_name}")
                    
            except Exception as e:
                print(f"✗ Error processing {model_name}: {str(e)}")
                continue
    
    # Report overall success
    if models_succeeded == total_models:
        print(f"\n✅ All {total_models} models successfully downloaded and cached")
        return 0
    else:
        print(f"\n⚠️ {models_succeeded}/{total_models} models successfully cached")
        print(f"❌ {total_models - models_succeeded} models failed")
        return 1

if __name__ == "__main__":
    exit_code = download_models()
    sys.exit(exit_code)