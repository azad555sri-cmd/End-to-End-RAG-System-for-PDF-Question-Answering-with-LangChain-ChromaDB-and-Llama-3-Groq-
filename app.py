import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
import os
import requests

# --- Setup API key ---
os.environ["GROQ_API_KEY"] = "groq_api_key_here"

st.set_page_config(page_title="PDF Chat QA", page_icon="📄")
st.title("📄 PDF Chat QA with Groq LLM")

# --- Sidebar: Select input type ---
option = st.sidebar.radio(
    "Choose input type:",
    ["PDF Upload/URL", "Direct Question", "API Prompt"]
)

# --- PDF Upload/URL Option ---
if option == "PDF Upload/URL":
    st.header("Upload a PDF or provide URL")
    
    pdf_file_path = None
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    pdf_url = st.text_input("Or enter PDF URL:")

    # Save uploaded file to disk
    if uploaded_file:
        pdf_file_path = f"uploaded_{uploaded_file.name}"
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("PDF uploaded successfully!")
    elif pdf_url:
        response = requests.get(pdf_url)
        pdf_file_path = "temp.pdf"
        with open(pdf_file_path, "wb") as f:
            f.write(response.content)
        st.success("PDF downloaded successfully!")

    if pdf_file_path:
        # --- Load and split PDF ---
        loader = UnstructuredFileLoader(pdf_file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        texts = text_splitter.split_documents(documents)

        # --- Create embeddings & retriever ---
        embeddings = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="vector_db")
        retriever = vectordb.as_retriever()

        # --- Setup LLM and QA chain ---
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask a question about the PDF:")

        if st.button("Send") and query:
            answer = qa_chain.run(query)
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", answer))

        # Display chat history
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")

# --- Direct Question Option ---
elif option == "Direct Question":
    st.header("Ask a direct question")
    question = st.text_input("Enter your question:")
    if st.button("Ask") and question:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(question)  # returns AIMessage
        st.markdown(f"**Answer:** {response.content}")  # access text with .content

# --- API Prompt Option ---
elif option == "API Prompt":
    st.header("Send a prompt to Groq API")
    prompt = st.text_area("Enter prompt text:")
    if st.button("Send Prompt") and prompt:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(prompt)  # returns AIMessage
        st.markdown(f"**Response:** {response.content}")  # access text with .content