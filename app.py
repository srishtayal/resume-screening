import streamlit as st
import pickle
import docx  
import PyPDF2  
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
svc_model = pickle.load(open('clf.pkl', 'rb')) 
tfidf = pickle.load(open('tfidf.pkl', 'rb')) 
le = pickle.load(open('encoder.pkl', 'rb'))  

# Define custom stopwords
custom_stop_words = set(["the", "is", "in", "and", "to", "of", "for", "a", "on", "with", "at", "by", "from"])

def clean_text(text):
    """Basic text cleaning: lowercase, remove URLs, mentions, hashtags, special characters."""
    text = text.lower()  
    text = re.sub(r'http\S+\s', ' ', text)  
    text = re.sub(r'[@#]\S+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  
    return text

def remove_common_words(text):
    """Remove common words like 'the', 'is', etc."""
    text = re.sub(r'\b(the|is|in|and|to|of|for|a|on|with|at|by|from|this|that|as|an|it|we|you|our)\b', ' ', text)
    return text

def extract_relevant_sections(text):
    """Extract relevant sections like skills, experience."""
    sections = ["skills", "experience", "projects", "technologies", "tools", "education"]
    extracted_text = "\n".join([line for line in text.split("\n") if any(sec in line.lower() for sec in sections)])
    return extracted_text if extracted_text else text

def cleanResume(text):
    """Full resume cleaning pipeline."""
    text = clean_text(text)
    text = remove_common_words(text)
    text = re.sub(r'\s+', ' ', text).strip()  
    return extract_relevant_sections(text)

def extract_text_from_pdf(file):
    """Extract text from PDF."""
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX."""
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file):
    """Extract text from TXT."""
    try:
        return file.read().decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        return file.read().decode('latin-1', errors='ignore')

def handle_file_upload(uploaded_file):
    """Handle file upload based on file type."""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def pred(input_resume):
    """Predict career category."""
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()  
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

def compute_similarity(resume_text, job_desc):
    """Compute similarity between resume and job description."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=3000)
    vectors = vectorizer.fit_transform([cleanResume(resume_text), cleanResume(job_desc)])
    
    similarity_score = cosine_similarity(vectors)[0, 1]
    return round(similarity_score * 100, 2)

def extract_keywords(text):
    """Extract keywords from text."""
    technical_stopwords = {"project", "work", "experience", "intern", "team", "development"}
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=50)
    ngrams = vectorizer.fit_transform([cleanResume(text)])
    keywords = vectorizer.get_feature_names_out()
    
    # Filter out generic terms, keep only skills/tools
    filtered_keywords = [kw for kw in keywords if kw not in technical_stopwords and len(kw) > 4]
    
    return set(filtered_keywords)

def suggest_keywords(resume_text, job_desc):
    """Suggest keywords to improve resume."""
    resume_keywords = extract_keywords(resume_text)
    job_desc_keywords = extract_keywords(job_desc)

    missing_keywords = job_desc_keywords - resume_keywords  # Find missing skills

    # Prioritize domain-specific skills
    prioritized_keywords = [kw for kw in missing_keywords if kw in job_desc_keywords]

    return set(prioritized_keywords[:10])  # Limit to top 10 suggestions

def main():
    st.set_page_config(page_title="Resume Analysis & Job Match", page_icon="📄", layout="wide")
    st.title("📄 Resume Category Prediction & Job Match Score")
    
    job_desc = st.text_area("💼 Paste the Job Description Here:")
    uploaded_file = st.file_uploader("📂 Upload Your Resume", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.subheader("Predicted Career Category")
            category = pred(resume_text)
            st.write(f"**{category}**")

            if job_desc:
                similarity_score = compute_similarity(resume_text, job_desc)
                st.subheader("Resume Match Score")
                st.write(f"**{similarity_score}% match**")

                if similarity_score < 50:
                    st.warning("⚠️ Your resume needs improvements to match this job better.")
                else:
                    st.success("✅ Great! Your resume is a good match for this job.")

                missing_keywords = suggest_keywords(resume_text, job_desc)
                if missing_keywords:
                    st.subheader("🔍 Suggested Keywords to Improve Your Resume")
                    st.write(f"Consider adding these keywords to better match the job description:")
                    st.write(f" **{', '.join(missing_keywords)}**")
                else:
                    st.success("✅ Your resume already contains relevant keywords for this job.")
        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
