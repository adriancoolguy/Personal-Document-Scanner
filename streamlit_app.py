import streamlit as st
import pandas as pd
import json
from pathlib import Path
from data_loader import load_file, scan_for_files
import openai
from datetime import datetime
from typing import Dict, Any

# Constants
MAX_FILE_SIZE_MB = 10  # Maximum file size to process

# Available GPT models
GPT_MODELS = {
    "GPT-4 Turbo": "gpt-4-1106-preview",
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-3.5 Turbo 16K": "gpt-3.5-turbo-16k"
}

def get_gpt_answer(query: str, df: pd.DataFrame, model: str = "gpt-4-1106-preview") -> str:
    """Get an answer from GPT based on relevant data."""
    if df.empty:
        return "I couldn't find any relevant information to answer your question."
    
    try:
        # Convert DataFrame to a more compact format with proper type handling
        def clean_value(val):
            if pd.isna(val):
                return "N/A"
            if isinstance(val, (int, float)):
                return str(val)
            return str(val)

        # Clean sample data
        sample_data = []
        for _, row in df.head(100).iterrows():
            cleaned_row = {str(k): clean_value(v) for k, v in row.items()}
            sample_data.append(cleaned_row)

        # Clean summary statistics
        summary_stats = {}
        for col in df.describe().columns:
            summary_stats[str(col)] = {
                str(k): clean_value(v)
                for k, v in df.describe()[col].items()
            }

        data_summary = {
            "columns": [str(col) for col in df.columns.tolist()],
            "row_count": str(len(df)),
            "sample_data": sample_data,
            "summary_stats": summary_stats
        }
        
        # Format the prompt
        prompt = f"""Question: {query}

Data Summary:
- Number of rows: {data_summary['row_count']}
- Columns: {', '.join(data_summary['columns'])}
- Sample data: {json.dumps(data_summary['sample_data'], indent=2, ensure_ascii=False)}
- Summary statistics: {json.dumps(data_summary['summary_stats'], indent=2, ensure_ascii=False)}

Please analyze this data and provide a clear, concise answer to the question. Focus on the most relevant information."""

        # Get response from GPT
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analyst. Provide clear, concise answers based on the data provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"

# Initialize Streamlit
st.set_page_config(
    page_title="Personal Document Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Personal Document Scanner")
st.write("Search and analyze your Excel and CSV files in Downloads folder.")

# OpenAI API key input
api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    openai.api_key = api_key
    st.session_state.api_key = api_key
elif 'api_key' in st.session_state:
    api_key = st.session_state.api_key
    openai.api_key = api_key

# Model selection
selected_model = st.selectbox(
    "Select GPT Model",
    options=list(GPT_MODELS.keys()),
    index=0,  # Default to GPT-4 Turbo
    help="Choose the GPT model to use for analysis."
)

# File selection
st.write("🔍 Select a File to Analyze")
downloads_dir = Path.home() / "Downloads"
excel_files = list(downloads_dir.glob("*.xlsx"))
csv_files = list(downloads_dir.glob("*.csv"))
all_files = excel_files + csv_files

if not all_files:
    st.warning("No Excel or CSV files found in Downloads folder.")
else:
    selected_file = st.selectbox(
        "Select a file to analyze:",
        options=all_files,
        format_func=lambda x: x.name
    )
    
    if selected_file:
        st.write(f"💬 Ask Questions About Your Document")
        st.write(f"Currently analyzing: {selected_file.name}")
        
        # Load and process the file
        try:
            df = load_file(str(selected_file))
            if df is not None and not df.empty:
                # Question input
                question = st.text_input("Type your question about the document:")
                
                if question:
                    with st.spinner("Analyzing your question..."):
                        answer = get_gpt_answer(question, df, GPT_MODELS[selected_model])
                        st.write("Answer")
                        st.write(answer)
            else:
                st.error("No data could be loaded from the selected file.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")