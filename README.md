A full Retrieval-Augmented Generation (RAG) pipeline to query and summarize PDF documents. This project uses LangChain for orchestration, ChromaDB for vector storage, HuggingFace embeddings for semantic representation, and Llama 3 (via Groq API) for natural language understanding. Users can ask questions or request summaries from PDFs, enabling efficient semantic search over unstructured documents.

## Features

- Load and process PDF documents dynamically
- Split PDFs into chunks for efficient retrieval
- Generate embeddings using HuggingFace models
- Store and query embeddings in Chroma vector database
- Answer questions or summarize PDFs with Llama 3 (Groq)
- Supports semantic search and contextual Q&A

---

## Tech Stack

- **Python**
- **LangChain** – RAG orchestration
- **ChromaDB** – vector database
- **HuggingFace Embeddings** – semantic representation
- **Llama 3 (Groq API)** – large language model
- **PDF Processing** – unstructured document handling

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install --upgrade langchain
pip install langchain_community
pip install langchain_text_splitter
pip install langchain_chroma
pip install langchain_groq
pip install requests

A Streamlit web app that lets you:

- Upload PDFs (local or via URL) and ask questions about their content.
- Send direct questions to the Groq LLM.
- Send custom prompts to the Groq API.

Built with LangChain, HuggingFace embeddings, and Chroma for vector search.

Features
1. PDF Upload/URL
 - Upload one or multiple PDFs.
 - Split content into chunks, create embeddings, and perform QA.
 - Maintains chat history for better interaction.
2. Direct Question
 - Ask any question directly to the LLM without PDF context.
3. API Prompt
 - Send any prompt to the Groq API and get a response.

Dependencies
 - streamlit
 - langchain
 - langchain_community
 - langchain_chroma
 - langchain_groq
 - unstructured
 - requests

Install all dependencies with pip install -r requirements.txt.

Folder Structure
pdf-chat-qa-groq/
│
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ README.md              # This file
├─ vector_db/             # Chroma vector DB (auto-generated)
└─ uploaded_<filename>.pdf # Uploaded PDFs (saved temporarily)
