# Personal-Document-Scanner

A simple, secure tool to ask natural-language questions about local Excel and CSV files — no re-uploading, no cloud storage, and no technical know-how required.

🔍 Overview
AskMyFiles is a local-first application that allows users to:

Index all .csv and .xlsx files from a selected folder on their desktop.

Ask natural-language questions like “How much did I spend on my kids in January 2023?”

Get accurate answers by using local embeddings + semantic search + a language model.

Avoid the cloud — no files are uploaded or sent externally unless explicitly configured.

🎯 Use Case
Your boss has dozens (or hundreds) of Excel spreadsheets scattered across folders. Instead of manually opening and searching each one, they want a simple tool where they can:

Choose a folder once

Ask: “How much did I spend on travel in Q4 2023?”

Get a response in seconds.

No re-uploading. No manual filtering. All offline.

💻 Tech Stack
Component	Tool/Library	Purpose
UI	Streamlit	Simple local web UI
Data Parsing	pandas	Reads .csv and .xlsx files
Embeddings	sentence-transformers	Local embedding of text rows
Vector Search	FAISS	Fast local similarity search
LLM (Optional)	OpenAI or llama-cpp	Processes the final question and generates a human-readable answer

📁 Folder Structure
perl
Copy
Edit
ask-my-files/
├── streamlit_app.py        # Main app logic
├── data_loader.py          # Loads and parses spreadsheets
├── vector_store.py         # Embeds and indexes data with FAISS
├── qa_engine.py            # Handles question answering using embeddings + LLM
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
⚙️ How It Works
File Selection

User selects a local folder

All .csv and .xlsx files in that folder are automatically parsed

Indexing

Each row (e.g. each transaction) is embedded using a pre-trained sentence transformer

Embeddings are stored locally in a FAISS index

Question Answering

User types a question in the UI

The system retrieves the most relevant rows

A local (or cloud-based) language model summarizes the answer

Output

A plain-English response appears under the input box

🛠 Setup Instructions
Clone the Repo

bash
Copy
Edit
git clone https://github.com/yourname/ask-my-files.git
cd ask-my-files
Install Dependencies
(Use a virtual environment if needed)

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
streamlit run streamlit_app.py
🧪 Example Questions You Can Ask
“What were my total expenses in January 2023?”

“How much did I spend on my kids last summer?”

“Which vendors charged me over $1,000 last quarter?”

“Show transactions related to education in 2022.”

🔐 Privacy First
All files stay on your machine.

No uploading or cloud syncing unless you opt-in to OpenAI.

You can swap in a fully local LLM (e.g. via llama-cpp-python) to keep all computation offline.

📦 Future Features (Optional)
PDF/Docx file support

Caching for faster re-launch

Built-in categorization/tagging

Export filtered data directly from UI

Keyword aliases (e.g. "kids" = ["school", "childcare", "toys"])