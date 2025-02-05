import os
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Initialize Hugging Face LLM
generator = pipeline(
    "text-generation",  # Task
    model="gpt2",       # Use GPT-2 or any other open-source model
    device="cpu",       # Use "cuda" if you have a GPU
    max_new_tokens=100      # Adjust as needed
)
llm = HuggingFacePipeline(pipeline=generator)

# Create FastAPI app
app = FastAPI()

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Generate embeddings
def generate_embeddings(text):
    return model.encode(text)

# Step 3: Match resume to jobs
def match_resume_to_jobs(resume_embedding, job_embeddings, job_descriptions):
    similarities = cosine_similarity([resume_embedding], job_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_jobs = [(job_descriptions[i], similarities[i]) for i in ranked_indices]
    return ranked_jobs

# Step 4: Explain matches using LangChain
def explain_match(resume_text, job_description):
    prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template="Explain why this job is a good match for the resume:\nResume: {resume}\nJob: {job_description}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(resume=resume_text, job_description=job_description)
# Sample list of skills you want to extract
skills_list = [
    'python', 'machine learning', 'deep learning', 'tensorflow', 'data analysis', 'java', 'sql', 'project management'
]

def extract_skills_from_resume(resume_text):
    resume_text = resume_text.lower()  # Convert to lowercase for better matching
    found_skills = [skill for skill in skills_list if skill.lower() in resume_text]
    return found_skills


# Full pipeline
def job_match_pipeline(resume_path, job_descriptions):
    # Step 1: Extract resume text
    resume_text = extract_text_from_pdf(resume_path)
    # Step 2: Extract skills from the resume
    resume_skills = extract_skills_from_resume(resume_text)
    
    # Step 2: Generate embeddings
    resume_embedding = generate_embeddings(resume_text)
    job_embeddings = [generate_embeddings(job) for job in job_descriptions]
    
    # Step 3: Match jobs
    ranked_jobs = match_resume_to_jobs(resume_embedding, job_embeddings, job_descriptions)
    
    # Step 4: Explain matches
    results = []
    for job, similarity in ranked_jobs[:5]:  # Top 5 matches
    # Extract relevant skills from the resume that match the job title
        relevant_skills = [skill for skill in resume_skills if skill.lower() in job.lower()]
    
        explanation = (
            f"This job matches because the resume mentions skills like {', '.join(relevant_skills)}."
            if relevant_skills else 
            "This job may be relevant based on your background."
        )
    
        results.append({
            "job": job,
            "similarity": float(similarity),
            "explanation": explanation
        })

    
    return results

# FastAPI endpoint
@app.post("/match-jobs/")
async def match_jobs(resume: UploadFile = File(...)):
    # Save uploaded file temporarily
    if not os.path.exists("temp"):
        os.makedirs("temp")
    resume_path = f"temp/{resume.filename}"
    with open(resume_path, "wb") as buffer:
        buffer.write(resume.file.read())
    
    # Replace with your job descriptions database
    job_descriptions = [
        "Software Engineer with experience in Python and machine learning.",
        "Data Scientist with expertise in data analysis and AI.",
        "Product Manager with a focus on agile development.",
        "DevOps Engineer with experience in cloud infrastructure.",
    ]
    
    # Run the pipeline
    results = job_match_pipeline(resume_path, job_descriptions)
    
    # Clean up temporary file
    os.remove(resume_path)
    
    return JSONResponse(content=results)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)