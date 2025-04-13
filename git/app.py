import streamlit as st
from transformers import pipeline
from keybert import KeyBERT
import fitz  # PyMuPDF
import docx
import nltk
nltk.download('stopwords')


# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
kw_model = KeyBERT()

# Extract text
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def generate_summary(text):
    return summarizer(text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']

def extract_keywords(text, top_n=5):
    return kw_model.extract_keywords(text, top_n=top_n)

# Streamlit UI
st.title("📄 AI Document Summarizer & Keyword Extractor")

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file:
    file_text = ""
    if uploaded_file.name.endswith(".pdf"):
        file_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        file_text = extract_text_from_docx(uploaded_file)

    file_text = file_text[:3000]  # Optional trim

    st.subheader("📋 Summary")
    st.write(generate_summary(file_text))

    st.subheader("🔑 Keywords")
    keywords = extract_keywords(file_text, top_n=5)
    for kw, _ in keywords:
        st.markdown(f"- {kw}")
