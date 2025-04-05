# Article Question Generator

An AI-powered application that automatically generates relevant questions from articles using state-of-the-art NLP models.

## Features

- Article summarization using BART/Pegasus models
- Intelligent question generation from article content
- Two summary refinement methods: manual (rule-based) and model-based
- Configurable number of questions (1-10)
- Sample articles included for demonstration
- Downloadable question output
- Progress tracking and status updates
- Mobile-friendly UI with custom styling

## Requirements

- Python 3.8+
- streamlit
- transformers
- langchain
- torch

## Installation

```bash
git clone <repository-url>
cd langchain-huggingface
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser to the displayed URL (typically http://localhost:8501)

3. Either:
   - Select a sample article from the dropdown
   - Paste your own article text

4. Configure options in the sidebar:
   - Choose refinement method (manual/model)
   - Set number of questions
   - Advanced: Select specific models

5. Click "Generate Questions" and wait for results

## Models Used

- Summarization: 
  - facebook/bart-large-cnn (default)
  - google/pegasus-xsum
  - facebook/bart-large-xsum

- Question Generation:
  - mrm8488/t5-base-finetuned-question-generation-ap (default)
  - t5-small

- Summary Refinement:
  - facebook/bart-large-xsum (default)
  - google/pegasus-xsum

## First Run

Note: On first run, the application will download the required models which may take several minutes depending on your internet connection.

## Architecture

The application uses a pipeline architecture:
1. Article summarization
2. Summary refinement (manual or model-based)
3. Question generation

Each step is processed with progress tracking and error handling.

## License

[Your chosen license]
