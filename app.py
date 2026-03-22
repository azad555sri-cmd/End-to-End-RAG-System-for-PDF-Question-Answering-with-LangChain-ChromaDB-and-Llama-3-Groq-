import streamlit as st
import os
import requests
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "groq_api_key_here"

st.title("📄 PDF Q&A with LangChain + Groq")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
use_sample = st.checkbox("Use sample PDF from URL", value=False)

if use_sample:
    url = "https://dspmuranchi.ac.in/pdf/Blog/Python%20Built-In%20Functions.pdf"
    response = requests.get(url)
    with open("sample.pdf", "wb") as f:
        f.write(response.content)
    pdf_path = "sample.pdf"
elif uploaded_file is not None:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
else:
    st.info("Please upload a PDF or select the sample PDF.")
    st.stop()

# Load document
loader = UnstructuredFileLoader(pdf_path)
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
texts = text_splitter.split_documents(documents)

# Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings()
persist_directory = "vector_db"
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

# Retriever
retriever = vectordb.as_retriever()

# LLM from Groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# User query
query = st.text_input("Enter your question about the PDF:")

if st.button("Get Answer") and query:
    with st.spinner("Generating answer..."):
        response = qa_chain.invoke({"query": query})
        st.subheader("Answer:")
        st.write(response["result"])

        st.subheader("Source Document:")
        for doc in response["source_documents"]:
            st.write(doc.metadata.get("source", "Unknown"))