import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
from pathlib import Path
import requests
from huggingface_hub import HfFolder

def verify_model_cache(model_name: str, cache_dir: str) -> bool:
    """Verify if model files exist in cache"""
    model_path = Path(cache_dir) / model_name
    return (model_path / "config.json").exists()

def download_models():
    # Configure HF settings
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['TRANSFORMERS_OFFLINE'] = "0"
    
    # Test connection
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        print("Connection to Hugging Face successful")
    except Exception as e:
        print(f"Connection error: {e}")
        print("Checking proxy settings...")
        # Try with proxy if direct connection fails
        os.environ['HTTPS_PROXY'] = "http://proxy.example.com:8080"  # Adjust proxy if needed
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ensure cache directory exists
    cache_dir = "/root/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    
    models = {
        "summarization": [
            "facebook/bart-large-cnn",
            "google/pegasus-xsum",
            "facebook/bart-large-xsum"
        ],
        "question": [
            "mrm8488/t5-base-finetuned-question-generation-ap",
            "t5-small"
        ]
    }
    
    for category, model_list in models.items():
        for model_name in model_list:
            print(f"Processing {model_name}...")
            try:
                # Skip if already cached properly
                if verify_model_cache(model_name, cache_dir):
                    print(f"Model {model_name} already cached")
                    continue
                
                # Add download options
                download_kwargs = {
                    "cache_dir": cache_dir,
                    "local_files_only": False,
                    "use_auth_token": False,
                    "force_download": True,
                    "proxies": None if 'HTTPS_PROXY' not in os.environ else {
                        'https': os.environ['HTTPS_PROXY']
                    },
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                # Download and cache model
                print(f"Downloading {model_name} to {cache_dir}")
                if "t5" in model_name.lower():
                    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
                
                # Force save to specific location
                model_path = Path(cache_dir) / model_name
                tokenizer.save_pretrained(str(model_path))
                model.save_pretrained(str(model_path))
                
                # Verify download
                if os.path.exists(os.path.join(cache_dir, model_name)):
                    print(f"Successfully cached {model_name}")
                else:
                    raise Exception(f"Failed to cache {model_name}")
                    
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
                continue

if __name__ == "__main__":
    download_models()
