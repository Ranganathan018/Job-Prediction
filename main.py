import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load models
st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text-generation", model="gpt2", device=-1, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=generator)

# Extract PDF text
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Generate embedding
def generate_embeddings(text):
    return st.session_state.model.encode(text)

# Match resume to jobs
def match_resume_to_jobs(resume_embedding, job_embeddings, job_descriptions):
    similarities = cosine_similarity([resume_embedding], job_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    return [(job_descriptions[i], similarities[i]) for i in ranked_indices]

# LangChain explanation
def explain_match(resume_text, job_description):
    prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template="Explain why this job is a good match for the resume:\nResume: {resume}\nJob: {job_description}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(resume=resume_text, job_description=job_description)

# Extract skills
skills_list = ['python', 'machine learning', 'deep learning', 'tensorflow', 'data analysis', 'java', 'sql', 'project management']
def extract_skills_from_resume(text):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

# Streamlit UI
st.title("Job Match Predictor from Resume")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Resume Text")
    st.write(resume_text)

    st.subheader("Matching Jobs...")
    job_descriptions = [
        "Software Engineer with experience in Python and machine learning.",
        "Data Scientist with expertise in data analysis and AI.",
        "Product Manager with a focus on agile development.",
        "DevOps Engineer with experience in cloud infrastructure.",
    ]

    resume_skills = extract_skills_from_resume(resume_text)
    resume_embedding = generate_embeddings(resume_text)
    job_embeddings = [generate_embeddings(jd) for jd in job_descriptions]
    ranked_jobs = match_resume_to_jobs(resume_embedding, job_embeddings, job_descriptions)

    for job, similarity in ranked_jobs[:5]:
        relevant_skills = [skill for skill in resume_skills if skill in job.lower()]
        explanation = f"This job matches because your resume includes: {', '.join(relevant_skills)}" if relevant_skills else "This job may be relevant based on general background."
        st.markdown(f"**Job:** {job}")
        st.markdown(f"**Similarity:** {similarity:.2f}")
        st.markdown(f"**Explanation:** {explanation}")
        st.markdown("---")
