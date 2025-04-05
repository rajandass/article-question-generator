import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set cache directory before any other imports
os.environ['TRANSFORMERS_CACHE'] = "D:/huggingface_cache"
os.environ['HF_HOME'] = "D:/huggingface_cache"
# Token now loaded from .env file automatically
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
device = 0 if torch.cuda.is_available() else -1

summarization_pipeline = pipeline(task="summarization", model="facebook/bart-large-cnn",device=device)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline(task="summarization", model="facebook/bart-large", device=device)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline(task="question-answering",model="deepset/roberta-base-squad2", device=device)

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way: \n\n {text}")

summarization_chain = summary_template | summarizer | refiner

text_to_summarize = input("\n Enter text to summarize:\n")
length= input("\n Enter the length (short/medium/long): ")
length_map = { "short": 50, "medium": 150, "long": 300}
max_length = length_map.get(length.lower(),150)

summary = summarization_chain.invoke({"text": text_to_summarize, "length": max_length})

print("\n **Generated Summary:** ")
print(summary)
while True:
    question = input("\n Ask a Question about the summary( or type 'exit' to stop): \n")
    if question.lower == "exit":
        break
    qa_result= qa_pipeline(question=question, context=summary)

    print("\n ** Answer: **")
    print(qa_result["answer"])




