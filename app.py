import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

from utils import process_uploaded_resumes
from rag_pipeline import run_screening_pipeline

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("🎯 Enterprise RAG Resume Screener & Ranker")
st.write(
    "Upload resumes, compare them against a job description, and rank candidates."
)

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

jd_input = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the job description here..."
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Screen Candidates"):

    if not uploaded_files:
        st.warning("Please upload resumes.")
        st.stop()

    if not jd_input:
        st.warning("Please enter a Job Description.")
        st.stop()

    with st.spinner("Processing resumes..."):

        chunks = process_uploaded_resumes(uploaded_files)

        results = run_screening_pipeline(
            chunks,
            jd_input
        )

        df = pd.DataFrame(results)

        df = df.sort_values(
            by="Match_Score",
            ascending=False
        ).reset_index(drop=True)

        st.success("Analysis Complete")

        st.subheader("Candidate Rankings")

        st.dataframe(
            df[["Candidate", "Match_Score", "Verdict"]],
            use_container_width=True
        )

        st.subheader("Detailed Analysis")

        for idx, row in df.iterrows():

            with st.expander(
                f"Rank {idx+1}: {row['Candidate']} ({row['Match_Score']}/100)"
            ):

                st.write(f"**Verdict:** {row['Verdict']}")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("### Strengths")
                    for p in row["Pros"]:
                        st.write(f"✅ {p}")

                with col2:
                    st.write("### Weaknesses")
                    for c in row["Cons"]:
                        st.write(f"⚠️ {c}")