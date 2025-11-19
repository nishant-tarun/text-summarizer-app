import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
import nltk
from io import StringIO
import time

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize NLP models (with caching)
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_abstractive_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

nlp = load_spacy_model()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SpacyTokenizer:
    def to_sentences(self, text):
        doc = nlp(text)
        return [str(sent).strip() for sent in doc.sents]

    def to_words(self, sentence):
        doc = nlp(sentence)
        return [token.text for token in doc if not token.is_space]

def extractive_summary(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, SpacyTokenizer())
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary)

def abstractive_summary(text, min_length=30, max_length=120):
    summarizer = load_abstractive_model()
    # Split text if too long (BART has 1024 token limit)
    max_chunk = 1000
    if len(text.split()) > max_chunk:
        text = ' '.join(text.split()[:max_chunk])
    
    result = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
    return result[0]['summary_text']

def get_text_stats(text):
    words = len(text.split())
    chars = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    return words, chars, sentences

def main():
    # Header
    st.title("📝 AI Text Summarizer")
    st.markdown("### Powered by Transformers & NLP")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        method = st.selectbox(
            "Summarization Method",
            ["extractive", "abstractive", "hybrid"],
            help="Extractive: Selects key sentences. Abstractive: Generates new sentences. Hybrid: Combines both."
        )
        
        st.markdown("---")
        
        if method in ["extractive", "hybrid"]:
            sentences_count = st.slider(
                "Number of Sentences",
                min_value=1,
                max_value=10,
                value=3,
                help="For extractive summarization"
            )
        else:
            sentences_count = 3
        
        if method in ["abstractive", "hybrid"]:
            min_length = st.slider("Min Length", 20, 100, 30)
            max_length = st.slider("Max Length", 50, 300, 120)
        else:
            min_length, max_length = 30, 120
        
        st.markdown("---")
        st.info("💡 **Tip:** Upload a .txt file or paste text directly!")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Input Text")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=["txt"],
            help="Upload a .txt file to summarize"
        )
        
        # Text input
        if uploaded_file:
            text_input = uploaded_file.read().decode("utf-8")
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        else:
            text_input = ""
        
        text_input = st.text_area(
            "Or paste your text here:",
            value=text_input,
            height=400,
            placeholder="Enter the text you want to summarize..."
        )
        
        # Show input stats
        if text_input:
            words, chars, sentences = get_text_stats(text_input)
            st.markdown(f"""
            <div class="stats-box">
                <b>Input Statistics:</b><br>
                📊 Words: {words} | Characters: {chars} | Sentences: {sentences}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("✨ Summary Output")
        
        # Summarize button
        if st.button("🚀 Summarize", use_container_width=True):
            if not text_input.strip():
                st.warning("⚠️ Please provide input text or upload a file.")
            else:
                try:
                    with st.spinner(f"Generating {method} summary..."):
                        start_time = time.time()
                        
                        if method == "extractive":
                            summary = extractive_summary(text_input, sentence_count=sentences_count)
                        elif method == "abstractive":
                            summary = abstractive_summary(text_input, min_length=min_length, max_length=max_length)
                        elif method == "hybrid":
                            extract = extractive_summary(text_input, sentence_count=sentences_count)
                            summary = abstractive_summary(extract, min_length=min_length, max_length=max_length)
                        
                        elapsed_time = time.time() - start_time
                        
                        # Display summary
                        st.markdown(f"""
                        <div class="summary-box">
                            {summary}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Summary stats
                        sum_words, sum_chars, sum_sentences = get_text_stats(summary)
                        compression_ratio = (1 - sum_words/words) * 100 if words > 0 else 0
                        
                        st.markdown(f"""
                        <div class="stats-box">
                            <b>Summary Statistics:</b><br>
                            📊 Words: {sum_words} | Characters: {sum_chars} | Sentences: {sum_sentences}<br>
                            📉 Compression: {compression_ratio:.1f}% | ⏱️ Time: {elapsed_time:.2f}s
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
                        
                        st.success("✅ Summary generated successfully!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👈 Configure settings and click 'Summarize' to generate summary")

    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About the Methods"):
        st.markdown("""
        - **Extractive Summarization**: Selects the most important sentences from the original text using TextRank algorithm.
        - **Abstractive Summarization**: Generates new sentences that capture the essence using BART (Facebook AI).
        - **Hybrid Summarization**: First extracts key sentences, then rephrases them abstractively for better coherence.
        """)

if __name__ == "__main__":
    main()