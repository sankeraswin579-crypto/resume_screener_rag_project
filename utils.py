import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_uploaded_resumes(uploaded_files):
    """
    Takes a list of Streamlit uploaded files, extracts text using PyPDFLoader,
    attaches metadata, chunks the text, and cleans up temporary files.
    """
    all_chunks = []
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for uploaded_file in uploaded_files:
        # Create a unique temporary path for PyPDFLoader to read
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load and parse PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            # Inject metadata to track which candidate this text belongs to
            candidate_name = uploaded_file.name.replace(".pdf", "")
            for doc in docs:
                doc.metadata["candidate_name"] = candidate_name
            
            # Split into semantic chunks
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            
        finally:
            # Ensure the file is deleted even if parsing fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    return all_chunks