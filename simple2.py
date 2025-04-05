import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set cache directory before any other imports
os.environ['TRANSFORMERS_CACHE'] = "D:/huggingface_cache"
os.environ['HF_HOME'] = "D:/huggingface_cache"
# Token loaded from .env file
if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    raise ValueError("Please set HUGGING_FACE_HUB_TOKEN in .env file")

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate


# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
device = 0 if torch.cuda.is_available() else -1

model = pipeline(
    task="text-generation",
    model="microsoft/phi-2",
    device=device,
    max_length=256,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    trust_remote_code=True,
    truncation=True
    )

llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template("Explain {topic} in the detail for a {age} year old to understand.")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")
response = chain.invoke({"topic": topic, "age": age})
print(response)