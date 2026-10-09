import re
import streamlit as st
import fitz  # PyMuPDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Website settings
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload your resume and compare it with a job description."
)

st.info(
    "This tool provides an estimated text similarity score "
    "and keyword analysis. It does not predict hiring outcomes."
)


# 2. Skills and keywords to check
SKILLS = [
    "python", "sql", "excel", "power bi", "tableau",
    "machine learning", "deep learning", "data analysis",
    "data visualization", "statistics", "pandas", "numpy",
    "scikit-learn", "matplotlib", "seaborn",
    "natural language processing", "nlp",
    "artificial intelligence", "generative ai",
    "streamlit", "django", "flask", "html", "css",
    "javascript", "java", "c++", "git", "github",
    "aws", "azure", "docker", "mongodb", "mysql",
    "data cleaning", "data preprocessing",
    "communication", "problem solving"
]


# 3. Extract text from PDF
def extract_resume_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pages = []

        for page in pdf:
            pages.append(page.get_text())

    return "\n".join(pages).strip()


# 4. Find skills in text
def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        # Match whole words or phrases, not partial words
        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            found.append(skill)

    return sorted(found)


# 5. Calculate text similarity
def calculate_similarity(resume_text, job_text):
    documents = [resume_text, job_text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(documents)

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(score * 100, 2)


# 6. Upload resume and enter job description
st.subheader("Step 1: Upload your resume")

uploaded_file = st.file_uploader(
    "Choose a PDF resume",
    type=["pdf"]
)

st.subheader("Step 2: Add the job description")

job_description = st.text_area(
    "Paste the job description here",
    height=220,
    placeholder=(
        "Example: We are looking for a Python Developer "
        "with Python, SQL, problem-solving, and Git skills."
    )
)


# 7. Analyze button
if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:
        st.error("Please upload your resume PDF.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    else:
        try:
            resume_text = extract_resume_text(uploaded_file)

            if len(resume_text) < 30:
                st.error(
                    "Very little text was found. "
                    "Your PDF may be scanned or image-only. "
                    "Try a text-based PDF."
                )

            else:
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)

                matching_skills = sorted(
                    set(resume_skills) & set(job_skills)
                )

                missing_skills = sorted(
                    set(job_skills) - set(resume_skills)
                )

                similarity = calculate_similarity(
                    resume_text,
                    job_description
                )

                # 8. Display results
                st.divider()
                st.header("Analysis Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Text Similarity",
                    f"{similarity}%"
                )

                col2.metric(
                    "Matching Skills",
                    len(matching_skills)
                )

                col3.metric(
                    "Potential Missing Skills",
                    len(missing_skills)
                )

                st.progress(
                    int(max(0, min(100, similarity)))
                )

                st.caption(
                    "Text similarity measures overlap in text patterns. "
                    "It is not an ATS score or hiring probability."
                )

                # 9. Matching skills
                st.subheader("✅ Matching Skills")

                if matching_skills:
                    st.write(", ".join(matching_skills))
                else:
                    st.write(
                        "No matching skills from the built-in "
                        "skill list were found."
                    )

                # 10. Missing skills
                st.subheader("📌 Potential Missing Keywords")

                if missing_skills:
                    st.write(", ".join(missing_skills))
                    st.caption(
                        "Check these against your actual experience. "
                        "Only add skills you genuinely have."
                    )
                else:
                    st.write(
                        "No missing keywords were detected "
                        "from the built-in skill list."
                    )

                # 11. Recommendations
                st.subheader("💡 Suggestions")

                if missing_skills:
                    st.write(
                        "Review the job requirements and your resume. "
                        "If you genuinely have any of the listed skills, "
                        "describe them clearly in your resume."
                    )

                if "projects" not in resume_text.lower():
                    st.write(
                        "Consider including relevant academic projects "
                        "if you have completed them."
                    )

                if "education" not in resume_text.lower():
                    st.write(
                        "Check that your education details are included."
                    )

                st.success("Resume analysis completed!")

        except Exception as error:
            st.error(
                "Could not analyze this PDF. Please try another "
                "text-based PDF. Technical details: "
                + str(error)
            )


# 12. Footer
st.divider()
st.caption(
    "Beginner project built with Python, Streamlit, "
    "PyMuPDF, and Scikit-learn."
)