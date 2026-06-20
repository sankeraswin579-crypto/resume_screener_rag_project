import json
import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



def run_screening_pipeline(all_chunks, job_description):
  

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": min(15, len(all_chunks))}
    )

    relevant_docs = retriever.invoke(job_description)

    context = ""

    for doc in relevant_docs:
        candidate = doc.metadata.get(
            "candidate_name",
            "Unknown"
        )

        context += f"""
Candidate: {candidate}

{doc.page_content}

----------------------------------
"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = f"""
You are an expert HR recruiter.

JOB DESCRIPTION:
{job_description}

RESUME DATA:
{context}

Evaluate every candidate.

Return ONLY valid JSON.

[
  {{
    "Candidate":"Name",
    "Match_Score":90,
    "Pros":["Point1","Point2"],
    "Cons":["Point1","Point2"],
    "Verdict":"Summary"
  }}
]
"""

    response = llm.invoke(prompt)

    output = response.content.strip()

    if "```json" in output:
        output = output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(output)
    except Exception:
        print("LLM Returned:")
        print(output)
        raise