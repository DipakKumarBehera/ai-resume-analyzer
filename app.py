import streamlit as st
import PyPDF2
import re
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# -----------------------------
# Skill Detection
# -----------------------------
def detect_skills(text):
    skills_list = [
        "python", "java", "c++", "machine learning",
        "data science", "deep learning", "sql",
        "html", "css", "javascript", "react"
    ]

    detected = []
    text = text.lower()

    for skill in skills_list:
        if re.search(skill, text):
            detected.append(skill)

    return detected


# -----------------------------
# Resume Score
# -----------------------------
def calculate_score(skills):
    return min(len(skills) * 10, 100)


# -----------------------------
# REAL ATS SCORE
# -----------------------------
def calculate_ats_score(resume_text, job_description):
    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    ats_score = round(similarity[0][0] * 100, 2)

    return ats_score


# -----------------------------
# Find Missing Keywords
# -----------------------------
def find_missing_keywords(resume_text, job_description):
    # Clean text
    resume_text = re.sub(r'[^\w\s]', '', resume_text.lower())
    job_description = re.sub(r'[^\w\s]', '', job_description.lower())

    resume_words = set(resume_text.split())
    job_words = set(job_description.split())

    # Common English words to ignore
    stop_words = {
        "the", "and", "with", "for", "are", "you", "your",
        "this", "that", "have", "from", "will", "should",
        "looking", "candidate", "experience", "knowledge",
        "role", "job", "work"
    }

    missing = job_words - resume_words

    important_missing = [
        word for word in missing
        if word not in stop_words and len(word) > 3
    ]

    return important_missing[:10]



# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(score, ats_score, skills):
    file_name = "AI_Resume_Report.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Resume Analysis Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Resume Score: {score}/100", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"ATS Score: {ats_score}/100", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Detected Skills:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for skill in skills:
        elements.append(Paragraph(f"- {skill}", styles["Normal"]))

    doc.build(elements)
    return file_name


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🚀 AI Resume Analyzer - Advanced Version")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here")

if uploaded_file is not None and job_description != "":
    resume_text = extract_text_from_pdf(uploaded_file)

    detected_skills = detect_skills(resume_text)
    score = calculate_score(detected_skills)
    ats_score = calculate_ats_score(resume_text, job_description)
    missing_keywords = find_missing_keywords(resume_text, job_description)

    st.success("Analysis Completed ✅")

    # Resume Score
    st.subheader("📊 Resume Score")
    st.write(f"{score}/100")

    # ATS Score
    st.subheader("🤖 Real ATS Score")
    st.write(f"{ats_score}/100")

    # Detected Skills
    st.subheader("🧠 Detected Skills")
    for skill in detected_skills:
        st.write("-", skill)

    # Skill Chart
    if detected_skills:
        st.subheader("📈 Skill Distribution Chart")
        skill_counts = [1] * len(detected_skills)

        fig, ax = plt.subplots()
        ax.bar(detected_skills, skill_counts)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Missing Keywords
    st.subheader("⚠ Missing Important Keywords")
    if missing_keywords:
        for word in missing_keywords:
            st.write("-", word)
    else:
        st.write("Great! No major keywords missing.")

    # Download PDF
    if st.button("📥 Download PDF Report"):
        pdf_file = generate_pdf(score, ats_score, detected_skills)

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Click Here to Download",
                data=f,
                file_name=pdf_file,
                mime="application/pdf"
            )
