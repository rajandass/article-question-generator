import os
# Set cache directory before any other imports
os.environ['TRANSFORMERS_CACHE'] = "D:/huggingface_cache"
os.environ['HF_HOME'] = "D:/huggingface_cache"
import torch
from transformers import pipeline

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
device = 0 if torch.cuda.is_available() else -1

model = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=device  # Use GPU if available, otherwise fallback to CPU
    )

response = model('New Delhi:The BJP has demanded an apology from Congress leader Sonia Gandhi over her remarks on the Waqf Bill. Mrs Gandhi termed the Waqf Bill a "brazen assault" on the Constitution, accusing the BJP of seeking to keep society in a state of "permanent polarisation." The Lok Sabha passed the Waqf amendment bill late on Thursday, leading to strong reactions from both the ruling party and the opposition. Speaking at a Congress Parliamentary Party meeting in Samvidhan Sadan, Mrs Gandhi claimed that the Bill was "bulldozed" through the lower house. She also criticised the proposed One Nation, One Election Bill, calling it a subversion of the Constitution, and vowed that the Congress would strongly oppose it."Whether it is education, civil rights and liberties, our federal structure, or the conduct of elections, the Modi government is dragging the country into an abyss where our Constitution will remain on paper," she said. She further accused the government of attempting to turn India into a "surveillance state." Congress President Mallikarjun Kharge and Leader of Opposition in Lok Sabha Rahul Gandhi were among those present at the meeting. Mrs Gandhi alleged that Prime Minister Narendra Modi had rebranded and marketed initiatives from the 2004-2014 Congress rule as his own, and called for a public outreach campaign to counter such claims.')
print(response)