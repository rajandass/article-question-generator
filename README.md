# Article Question Generator

AI-powered application that generates questions from articles using Hugging Face models and LangChain.

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

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

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

## Development

To create a new feature branch:
```bash
git checkout -b feature/your-feature-name
```

## License

[Your chosen license]
