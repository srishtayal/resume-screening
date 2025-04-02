# **Resume Screening App**  

A machine learning-powered web application that automates resume screening, helping employers efficiently filter and shortlist candidates. The app analyzes resumes, extracts key details, and categorizes candidates based on job relevance.  

## **Features**  

- **Resume Upload** – Supports multiple formats (PDF, DOCX, etc.).  
- **Automated Analysis** – Extracts skills, experience, education, and other key details using NLP.  
- **Job Matching** – Suggests relevant professions based on resume content.  
- **User-Friendly UI** – Built with Streamlit for an intuitive and interactive experience.  

##  **Installation & Setup**  

### **Prerequisites**  
Ensure you have the following installed:  
- Python **3.7+**  
- **pip** package manager  
- Required dependencies (listed in `requirements.txt`)  

### **Setup Instructions**  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/srishtayal/resume-screening.git
   cd resume-screening
   ```  
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Run the application**  
   ```bash
   streamlit run app.py
   ```  
4. **Access the app**  
   Open your browser and go to: [http://localhost:8501](http://localhost:8501)  

##  **Technology Stack**  

**Frontend:** Streamlit  
**Backend:** Python (Pandas, NumPy, SpaCy)  
**Machine Learning:** NLP-based models for resume parsing and job matching  
**Data Visualization:** Matplotlib, Seaborn for insights  

##  **Usage**  

1️⃣ **Upload** a resume using the "Upload Resume" button.  
2️⃣ The app **processes the resume** and extracts key information.  
3️⃣ Get an **instant recommendation** on job suitability based on extracted data.  

