import streamlit as st
import PyPDF2
import os
#from dotenv import load_dotenv
import google.generativeai as genai


# Load API key (use st.secrets on Streamlit Cloud)
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API key not found. Make sure it's set in Streamlit Secrets.")
    st.stop()


genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

# Text chunking function
def split_text(text, chunk_size=3000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Load PDF content
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Get relevant response
def get_answer(question, chunks):
    context = "\n\n".join(chunks[:3])  # Just using top 3 chunks for now
    prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {question}"
    res = model.generate_content(prompt)
    return res.text

# UI starts
st.set_page_config("📄 Document Q&A-BOT")
st.title("📝 Get your answers in a click")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# question = st.text_input("Ask a question related to the document:")
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("Ask a question related to the document:")
    submit_question = st.form_submit_button("Ask")

if uploaded_file:
    pdf_text = read_pdf(uploaded_file)
    chunks = split_text(pdf_text)

    if submit_question and question.strip():
        with st.spinner("Thinking..."):
            answer = get_answer(question, chunks)
            st.success(answer)
