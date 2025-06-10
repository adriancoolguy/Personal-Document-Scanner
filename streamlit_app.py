import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from data_loader import load_data, scan_for_files
from embeddings.vector_store import VectorStore
import openai

# Constants
CACHE_DIR = Path.home() / ".personal_document_scanner"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.json"
FILE_METADATA_CACHE = CACHE_DIR / "file_metadata.json"

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_cached_embeddings():
    """Load cached embeddings if they exist."""
    if EMBEDDINGS_CACHE.exists():
        with open(EMBEDDINGS_CACHE, 'r') as f:
            return json.load(f)
    return None

def save_embeddings(embeddings_data):
    """Save embeddings to cache."""
    with open(EMBEDDINGS_CACHE, 'w') as f:
        json.dump(embeddings_data, f)

def get_file_metadata():
    """Get metadata about scanned files."""
    if FILE_METADATA_CACHE.exists():
        with open(FILE_METADATA_CACHE, 'r') as f:
            return json.load(f)
    return {}

def save_file_metadata(metadata):
    """Save file metadata to cache."""
    with open(FILE_METADATA_CACHE, 'w') as f:
        json.dump(metadata, f)

def get_gpt_answer(question: str, relevant_data: list) -> str:
    """Get GPT's analysis of the relevant data."""
    # Format the relevant data with source information
    context = "\n".join([
        f"Data point {i+1}:\n" + "\n".join([
            f"{k}: {v}" for k, v in row.items()
            if not pd.isna(v) and str(v).strip()  # Skip empty/null values
        ])
        for i, row in enumerate(relevant_data)
    ])
    
    # Create a more general-purpose prompt
    prompt = f"""Based on the following data from various documents, please answer this question: {question}

Relevant data:
{context}

Instructions:
1. Answer the question based only on the provided data
2. If you find multiple relevant pieces of information, include all of them
3. If the data doesn't contain enough information to answer the question, say so
4. If you find information that's related but not exactly what was asked, mention it
5. Be specific about what information you found and what you didn't find

Please provide a clear, concise answer based only on the data provided."""

    # Get GPT's response
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on document content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # Balanced temperature for natural but focused answers
    )
    
    return response.choices[0].message.content

# Initialize Streamlit
st.set_page_config(
    page_title="Personal Document Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Personal Document Scanner")
st.markdown("Ask questions about your Excel and CSV files in Downloads folder.")

# OpenAI API key input
api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    openai.api_key = api_key
    st.session_state.api_key = api_key
elif 'api_key' in st.session_state:
    api_key = st.session_state.api_key
    openai.api_key = api_key

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}
if 'indexing_complete' not in st.session_state:
    st.session_state.indexing_complete = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Ensure cache directory exists
ensure_cache_dir()

# Load cached data
cached_embeddings = load_cached_embeddings()
file_metadata = get_file_metadata()

# Debug output for UI state
st.write("DEBUG:", {
    "api_key": bool(api_key),
    "vector_store": st.session_state.vector_store is not None,
    "indexing_complete": st.session_state.indexing_complete,
    "cached_embeddings": cached_embeddings is not None,
})

# Robust UI state handling
if st.session_state.vector_store and st.session_state.indexing_complete:
    st.subheader("Ask a Question")
    query = st.text_input("Type your question about any document in your Downloads folder:")
    if query:
        with st.spinner("Searching and analyzing..."):
            results = st.session_state.vector_store.query(query, top_k=5)
            if results:
                # Extract just the data rows (without scores)
                relevant_data = [row for row, _ in results]
                
                # Get GPT's analysis
                answer = get_gpt_answer(query, relevant_data)
                
                # Display GPT's answer
                st.subheader("Answer")
                st.write(answer)
                
                # Display the relevant data
                st.subheader("Relevant Data")
                results_df = pd.DataFrame([
                    {**row, 'similarity_score': 1 / (1 + score)}
                    for row, score in results
                ])
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No matches found. Try rephrasing your question or check if the documents have been properly indexed.")
elif not st.session_state.indexing_complete:
    if not cached_embeddings and api_key:
        with st.spinner("Scanning Downloads folder and building index..."):
            try:
                st.write("DEBUG: Scanning for files...")
                files = scan_for_files()
                st.write(f"DEBUG: Found {len(files)} files.")
                if not files:
                    st.warning("No Excel or CSV files found in Downloads folder.")
                else:
                    all_data = []
                    progress_bar = st.progress(0)
                    total_files = len(files)
                    # Only process the first 5 Excel files
                    excel_files = [f for f in files if f.suffix.lower() in [".xls", ".xlsx"]]
                    files_to_process = excel_files[:5]
                    for idx, file in enumerate(files):
                        if file not in files_to_process:
                            st.write(f"DEBUG: Skipped {file} (not in first 5 Excel files)")
                            continue
                        st.write(f"DEBUG: Loading file {file}")
                        if str(file) not in st.session_state.processed_files:
                            try:
                                df = load_data(str(file.parent))
                                all_data.append(df)
                                file_metadata[str(file)] = {
                                    "last_modified": os.path.getmtime(file),
                                    "size": os.path.getsize(file)
                                }
                                st.session_state.processed_files.add(str(file))
                                st.write(f"DEBUG: Loaded {file}")
                            except Exception as e:
                                st.error(f"Error processing {file}: {str(e)}")
                        progress = (idx + 1) / total_files
                        progress_bar.progress(progress)
                    if all_data:
                        st.write("DEBUG: Combining all data...")
                        combined_df = pd.concat(all_data, ignore_index=True)
                        # Cap the total number of rows to 1000 for proof of concept
                        combined_df = combined_df.head(1000)
                        st.write(f"DEBUG: Using {len(combined_df)} rows for embedding (capped at 1000)")
                        st.write("DEBUG: Initializing vector store...")
                        vector_store = VectorStore(api_key=api_key)
                        text_chunks = []
                        original_rows = []
                        for _, row in combined_df.iterrows():
                            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                            text_chunks.append(row_text)
                            original_rows.append(row.to_dict())
                        st.write("DEBUG: Building index...")
                        embedding_progress = st.progress(0)
                        embedding_status = st.empty()
                        def progress_callback(done, total, status=""):
                            embedding_progress.progress(done / total)
                            embedding_status.write(f"{status} ({done}/{total})")
                        vector_store.build_index(text_chunks, original_rows, progress_callback=progress_callback)
                        embedding_progress.progress(1.0)
                        embedding_status.write("Indexing complete!")
                        st.write("DEBUG: Index built successfully.")
                        st.session_state.vector_store = vector_store
                        embeddings_data = {
                            "text_chunks": text_chunks,
                            "original_rows": original_rows
                        }
                        save_embeddings(embeddings_data)
                        save_file_metadata(file_metadata)
                        st.success(f"Successfully indexed {len(files)} files!")
                        st.session_state.indexing_complete = True
                        st.experimental_rerun()
                    else:
                        st.error("No data could be loaded from the found files.")
            except Exception as e:
                st.error(f"Critical error during indexing: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    elif cached_embeddings and api_key and st.session_state.vector_store is None:
        with st.spinner("Loading cached embeddings..."):
            vector_store = VectorStore(api_key=api_key)
            vector_store.build_index(
                cached_embeddings["text_chunks"],
                cached_embeddings["original_rows"]
            )
            st.session_state.vector_store = vector_store
            st.session_state.indexing_complete = True
            st.success("Loaded cached embeddings!")
            st.experimental_rerun()
    else:
        st.info("Please enter your OpenAI API key to begin.")
else:
    st.error("Unexpected state. Try refreshing the page or restarting the app.") 