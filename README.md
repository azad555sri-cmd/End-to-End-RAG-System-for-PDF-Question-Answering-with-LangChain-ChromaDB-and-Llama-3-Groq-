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
