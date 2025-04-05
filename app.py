import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from typing import List, Dict, Literal
import time

# Set page configuration
st.set_page_config(
    page_title="Article Question Generator",
    page_icon="❓",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .question {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading models... (This may take a minute)")
def initialize_pipelines(summarization_model="facebook/bart-large-cnn", 
                         question_model="mrm8488/t5-base-finetuned-question-generation-ap",
                         refinement_model="facebook/bart-large-xsum"):
    """Initialize all required pipelines with caching for efficiency"""
    # Summarization pipeline
    sum_tokenizer = AutoTokenizer.from_pretrained(summarization_model)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model)
    summarization_pipeline = HuggingFacePipeline(
        pipeline=pipeline(
            "summarization",
            model=sum_model,
            tokenizer=sum_tokenizer,
            max_length=150,
            min_length=30,
            do_sample=False
        )
    )
    
    # Question generation pipeline - Using explicit T5Tokenizer
    from transformers import T5Tokenizer  # Add this import at the top of your file
    q_tokenizer = T5Tokenizer.from_pretrained(question_model)
    q_model = AutoModelForSeq2SeqLM.from_pretrained(question_model)
    question_pipeline = HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",
            model=q_model,
            tokenizer=q_tokenizer,
            max_length=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    )
    
    # Refinement pipeline (for model-based refinement)
    refine_tokenizer = AutoTokenizer.from_pretrained(refinement_model)
    refine_model = AutoModelForSeq2SeqLM.from_pretrained(refinement_model)
    refinement_pipeline = HuggingFacePipeline(
        pipeline=pipeline(
            "summarization",
            model=refine_model,
            tokenizer=refine_tokenizer,
            max_length=100,
            min_length=30,
            do_sample=True,
            temperature=0.4  # Lower temperature for more focused refinement
        )
    )
    
    return summarization_pipeline, question_pipeline, refinement_pipeline

def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Break text into chunks that fit within model context limits"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n\n'
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def summarize_article(article_text: str, summarization_pipeline):
    """Generate a concise summary of the article"""
    # Handle longer texts by chunking and summarizing each chunk
    chunks = chunk_text(article_text)
    chunk_summaries = []
    
    for chunk in chunks:
        summary = summarization_pipeline.invoke(chunk)
        chunk_summaries.append(summary)
    
    # Combine chunk summaries
    full_summary = " ".join(chunk_summaries)
    
    # If the combined summary is still long, summarize again
    if len(full_summary.split()) > 150:
        full_summary = summarization_pipeline.invoke(full_summary)
        
    return full_summary

def refine_summary_manually(summary: str) -> str:
    """Extract key points and refine the summary using rule-based approaches"""
    # Split into sentences
    sentences = summary.split('. ')
    
    # Keep only sentences with substantial content (at least 5 words)
    refined_sentences = [s for s in sentences if len(s.split()) >= 5]
    
    # Rejoin and ensure proper punctuation
    refined_summary = '. '.join(refined_sentences)
    if not refined_summary.endswith('.'):
        refined_summary += '.'
        
    return refined_summary

def refine_summary_with_model(summary: str, refinement_pipeline) -> str:
    """Refine the summary using a model for a more focused extraction of key points"""
    # Add a prompt to guide the refinement
    prompt = f"Extract key points from this summary: {summary}"
    
    # Use the refinement model to focus and clarify the summary
    refined_summary = refinement_pipeline.invoke(prompt)
    
    # Ensure proper formatting and punctuation
    if not refined_summary.endswith('.'):
        refined_summary += '.'
        
    return refined_summary

def generate_questions(text: str, question_pipeline, num_questions: int = 5) -> List[str]:
    """Generate questions based on the provided text"""
    # Split text into sentences to generate questions on key points
    sentences = text.split('. ')
    questions = []
    
    # Process each sentence to generate a potential question
    for sentence in sentences:
        if len(sentence.split()) >= 5:  # Only use substantial sentences
            # T5 question generation models typically expect "answer: context" format
            prompt = f"answer: {sentence}"
            question = question_pipeline.invoke(prompt)
            
            # Clean up the generated question
            clean_question = question.strip()
            if not clean_question.endswith('?'):
                clean_question += '?'
            
            questions.append(clean_question)
            
            if len(questions) >= num_questions:
                break
    
    return questions

def process_article(article_text: str, 
                   refinement_method: Literal["manual", "model"],
                   num_questions: int,
                   pipelines):
    """Process an article with progress tracking for Streamlit"""
    summarization_pipeline, question_pipeline, refinement_pipeline = pipelines
    
    # Create placeholders for progress updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Summarize the article
    status_text.text("Summarizing article...")
    summary = summarize_article(article_text, summarization_pipeline)
    progress_bar.progress(33)
    
    # Step 2: Refine the summary
    status_text.text(f"Refining summary using {refinement_method} method...")
    if refinement_method == "manual":
        refined_summary = refine_summary_manually(summary)
    else:
        refined_summary = refine_summary_with_model(summary, refinement_pipeline)
    progress_bar.progress(66)
    
    # Step 3: Generate questions
    status_text.text("Generating questions...")
    questions = generate_questions(refined_summary, question_pipeline, num_questions)
    progress_bar.progress(100)
    
    # Clear status indicators
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return {
        "summary": summary,
        "refined_summary": refined_summary,
        "questions": questions,
        "refinement_method": refinement_method
    }

# Main application
def main():
    # Display header
    st.markdown('<p class="main-header">Article Question Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Generate relevant questions from any article using AI</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    refinement_method = st.sidebar.radio(
        "Summary Refinement Method",
        ["manual", "model"],
        help="Manual uses rule-based extraction, Model uses an AI model to extract key points"
    )
    
    num_questions = st.sidebar.slider(
        "Number of Questions to Generate",
        min_value=1,
        max_value=10,
        value=5,
        help="Select how many questions to generate from the article"
    )
    
    # Models selection (advanced options)
    with st.sidebar.expander("Advanced Model Options"):
        summarization_model = st.selectbox(
            "Summarization Model",
            ["facebook/bart-large-cnn", "google/pegasus-xsum", "facebook/bart-large-xsum"],
            index=0,
            help="Choose the model for article summarization"
        )
        
        question_model = st.selectbox(
            "Question Generation Model",
            ["mrm8488/t5-base-finetuned-question-generation-ap", "t5-small"],
            index=0,
            help="Choose the model for generating questions"
        )
        
        refinement_model = st.selectbox(
            "Refinement Model",
            ["facebook/bart-large-xsum", "google/pegasus-xsum"],
            index=0,
            help="Choose the model for refining the summary (only used with model-based refinement)"
        )
    
    # Load models when app starts
    with st.spinner("Loading models... This may take a minute on first run"):
        pipelines = initialize_pipelines(
            summarization_model=summarization_model,
            question_model=question_model,
            refinement_model=refinement_model
        )
    
    # Sample article options
    st.markdown('<p class="sub-header">Article Input</p>', unsafe_allow_html=True)
    sample_articles = {
        "None - Enter my own article": "",
        "AI Overview": """Artificial intelligence (AI) is transforming industries across the global economy. From healthcare to transportation, AI applications are changing how we live and work. 
        
In healthcare, AI algorithms are helping doctors diagnose diseases from medical images with greater accuracy. Machine learning models trained on vast datasets of X-rays, MRIs, and CT scans can detect patterns that might be missed by human eyes. This is particularly useful for detecting early-stage cancers and rare conditions.

The transportation sector is being revolutionized by autonomous driving technologies. Self-driving cars use a combination of sensors, cameras, and AI to navigate roads safely. Companies like Tesla, Waymo, and Cruise are making significant advances in this area, though full autonomy in all conditions remains a challenge.

Natural language processing, another branch of AI, is powering virtual assistants like Siri, Alexa, and Google Assistant. These systems can understand and respond to human speech, making technology more accessible to everyone. They're becoming increasingly sophisticated, handling complex queries and performing a wide range of tasks.

Despite these advances, AI also raises important ethical considerations. Issues of privacy, bias in algorithms, and potential job displacement need careful attention. Policymakers around the world are working to create regulatory frameworks that foster innovation while protecting public interests.

As AI continues to evolve, collaboration between technologists, ethicists, policymakers, and the public will be essential to ensure these powerful tools benefit humanity.""",
        "Climate Change": """Climate change is one of the most pressing challenges facing humanity today. It refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities, especially the burning of fossil fuels.

The science is clear: global carbon dioxide emissions have increased dramatically since the industrial revolution, leading to a greenhouse effect that warms the planet. The Intergovernmental Panel on Climate Change (IPCC) has established that human-induced warming reached approximately 1°C above pre-industrial levels in 2017 and is increasing at about 0.2°C per decade.

This warming has far-reaching consequences. Rising sea levels threaten coastal communities and island nations. Changing precipitation patterns affect agriculture and food security. Extreme weather events like hurricanes, floods, and wildfires are becoming more frequent and intense.

Addressing climate change requires both mitigation strategies to reduce emissions and adaptation measures to deal with unavoidable impacts. Renewable energy technologies like solar and wind power are increasingly cost-competitive with fossil fuels. Electric vehicles, energy-efficient buildings, and sustainable agricultural practices can further reduce our carbon footprint.

International cooperation is essential, with frameworks like the Paris Agreement establishing goals to limit warming to well below 2°C. Many countries, cities, businesses, and individuals are taking action, but current pledges are insufficient to prevent dangerous levels of warming.

The economic transition needed is significant but also presents opportunities for innovation, job creation, and sustainable development. Climate justice considerations ensure that both the costs and benefits of climate action are shared equitably."""
    }
    
    selected_article = st.selectbox("Choose a sample article or enter your own", list(sample_articles.keys()))
    
    article_text = st.text_area(
        "Paste your article text here:",
        height=300,
        value=sample_articles[selected_article]
    )
    
    # Process article when button is clicked
    if st.button("Generate Questions", type="primary", disabled=not article_text.strip()):
        if len(article_text.strip()) < 100:
            st.error("Please enter a longer article (at least 100 characters)")
        else:
            with st.spinner("Processing article..."):
                result = process_article(
                    article_text, 
                    refinement_method, 
                    num_questions,
                    pipelines
                )
            
            # Display results
            st.markdown('<p class="sub-header">Generated Content</p>', unsafe_allow_html=True)
            
            # Summary section
            with st.expander("Article Summary", expanded=True):
                st.markdown(f"<div class='highlight'>{result['summary']}</div>", unsafe_allow_html=True)
            
            # Refined summary section
            with st.expander("Refined Summary", expanded=True):
                st.markdown(f"<p class='info-text'>Refinement method: <b>{result['refinement_method']}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<div class='highlight'>{result['refined_summary']}</div>", unsafe_allow_html=True)
            
            # Questions section
            st.markdown("### Generated Questions")
            for i, question in enumerate(result["questions"], 1):
                st.markdown(f"<div class='question'><b>Q{i}:</b> {question}</div>", unsafe_allow_html=True)
            
            # Allow export of questions
            questions_text = "\n".join([f"Q{i}: {q}" for i, q in enumerate(result["questions"], 1)])
            st.download_button(
                label="Download Questions",
                data=questions_text,
                file_name="generated_questions.txt",
                mime="text/plain"
            )
    
    # Add information about the app
    with st.expander("About this App"):
        st.markdown("""
        This application uses state-of-the-art NLP models to:
        
        1. **Summarize** articles to extract their core content
        2. **Refine** summaries using either rule-based approaches or additional AI models
        3. **Generate relevant questions** based on the article content
        
        It's useful for:
        - Educators creating reading comprehension materials
        - Content creators developing FAQs
        - Students testing their understanding of complex texts
        - Researchers extracting key questions from papers
        
        The app uses the following models from Hugging Face:
        - Summarization: BART or Pegasus models
        - Question generation: T5-based models
        
        Built with Streamlit, Transformers, and LangChain.
        """)

if __name__ == "__main__":
    main()