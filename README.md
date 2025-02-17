# Resume Screening App

This web application is designed for automated resume screening using machine learning models to help employers quickly filter and shortlist candidates based on their resumes. The app analyzes resumes, extracts relevant information, and presents a summary or categorization to assist recruiters in making decisions.

## Features

- **Upload Resume**: Users can upload resumes in various formats such as PDF, DOCX, etc.
- **Resume Analysis**: The app uses NLP models to extract key information from resumes like skills, experience, education, and more.
- **User Interface**: Simple, clean, and intuitive interface using Streamlit.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.7+
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Spacy
- PyPDF2
- Other dependencies as specified in the `requirements.txt` file

### Steps to Set Up Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/srishtayal/resume-screening.git
   ```

2. Navigate to the project folder:
   ```bash
   cd resume-screening
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open the app in your browser at:
   ```
   http://localhost:8501
   ```

## Usage

- Upload a resume by clicking the "Upload Resume" button.
- Once uploaded, the app will process the resume and display suggested profession.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, with libraries such as Pandas, NumPy, and SpaCy for text processing and analysis.
- **Machine Learning**: Custom models for NLP-based resume analysis and scoring.
- **Data Visualization**: Matplotlib, Seaborn, and other plotting libraries to visualize extracted resume data.

  ---

❤️ Srishti Tayal
