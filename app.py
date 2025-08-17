import streamlit as st
from utils import create_qa_chain, answer_question
import time
import os

# Token handling (works both locally and on Spaces)
HF_TOKEN = os.environ.get('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not HF_TOKEN:
    st.error("Missing Hugging Face token! Add it in Space settings or .env file.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# App configuration (keep your exact UI)
st.set_page_config(
    page_title="Simple Q&A System nashat",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_qa_system():
    with st.spinner("Loading GPT-2 Medium (this may take 1-2 minutes)..."):
        return create_qa_chain()

try:
    qa_chain, embeddings, text_splitter = load_qa_system()
except Exception as e:
    st.error(f"Failed to load: {str(e)}")
    st.stop()

# Your exact UI text and styling
st.title("🤖 Hi! Ask Me a Question")
st.caption("Using GPT-2 Medium with Wikipedia/Web context (please be careful writing your question, im not the smartest :))")

question = st.text_input("What would you like to know?", 
                        placeholder="Whats the capital city of France?")

if question:
    with st.spinner("Generating answer..."):
        start_time = time.time()
        result = answer_question(qa_chain, embeddings, text_splitter, question)
        elapsed = time.time() - start_time
    
    st.subheader("Answer")
    st.write(result["answer"])
    
    if result["sources"]:
        st.caption(f"📚 Sources: {result['sources']} (Generated in {elapsed:.1f}s)")
    
    if "error" in result["answer"].lower():
        st.warning("Try rephrasing your question")