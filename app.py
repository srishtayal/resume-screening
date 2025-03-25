import streamlit as st
import os
import docx
import PyPDF2
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
svc_model = pickle.load(open('clf.pkl', 'rb')) 
tfidf = pickle.load(open('tfidf.pkl', 'rb')) 
le = pickle.load(open('encoder.pkl', 'rb'))  

# Function to extract text from different file formats
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode('utf-8', errors='ignore')

def handle_file_upload(uploaded_file):
    """Extracts text based on file type."""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.strip()

# Function to predict resume category
def pred(input_resume):
    cleaned_text = clean_text(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Function to compute similarity between resume and job description
def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=3000)
    vectors = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_desc)])
    similarity_score = cosine_similarity(vectors)[0, 1]
    return round(similarity_score * 1000, 2)

# Extract keywords
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=50)
    ngrams = vectorizer.fit_transform([clean_text(text)])
    return set(vectorizer.get_feature_names_out())

# Suggest missing keywords
def suggest_keywords(resume_text, job_desc):
    resume_keywords = extract_keywords(resume_text)
    job_desc_keywords = extract_keywords(job_desc)
    missing_keywords = job_desc_keywords - resume_keywords
    return list(missing_keywords)[:10]  # Limit to top 10 suggestions

# Streamlit UI
def main():
    st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")
    st.title("📄 Resume Matching & Job Fit Analyzer")

    # Job description input
    job_desc = st.text_area("💼 Paste the Job Description Here:")

    # File uploader (Multiple Resumes)
    uploaded_files = st.file_uploader("📂 Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # Toggle button to show/hide career prediction
    show_career_category = st.checkbox("Show Predicted Career Category")

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            resume_text = handle_file_upload(uploaded_file)

            if not resume_text.strip():
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                continue

            category = pred(resume_text) if show_career_category else None
            similarity_score = compute_similarity(resume_text, job_desc) if job_desc else None
            missing_keywords = suggest_keywords(resume_text, job_desc) if job_desc else []

            results.append({
                "filename": uploaded_file.name,
                "category": category,
                "similarity_score": similarity_score,
                "missing_keywords": missing_keywords
            })

        # Display Results
        if results:
            results.sort(key=lambda x: x["similarity_score"], reverse=True)  # Sort by similarity
            for res in results:
                st.subheader(f"📌 {res['filename']}")
                
                if show_career_category:
                    st.write(f"**Predicted Career Category:** {res['category']}")

                if job_desc:
                    st.write(f"**Resume Match Score:** {res['similarity_score']}%")
                    if res["similarity_score"] < 50:
                        st.warning("⚠️ Your resume needs improvements.")
                    else:
                        st.success("✅ Good match!")

                    if res["missing_keywords"]:
                        st.write(f"🔍 **Suggested Keywords:** {', '.join(res['missing_keywords'])}")

if __name__ == "__main__":
    main()
