# DocuBrain: RAG-Based Document Q&A System (Gemini + Pinecone + Streamlit)

# Step 1: Import Required Libraries
import os
import streamlit as st
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()

# Step 2: Initialize APIs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Step 3: Load and Parse PDF Document
def load_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages

# Step 4: Split Text into Chunks
def split_text(pages):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    return docs

# Step 5: Convert to Embeddings and Create Pinecone Vector Store
def create_vector_store(docs):
    embeddings = GooglePalmEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    index_name = "docubrain-index"
    if index_name not in pinecone.list_indexes():
        Pinecone.from_documents(docs, embeddings, index_name=index_name)
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    return vectorstore

# Step 6: Create RAG QA Chain
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Step 7: Streamlit App UI
st.title("📄 DocuBrain – Ask Questions from Your PDF")

uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing document...")
    pages = load_document("temp.pdf")
    docs = split_text(pages)
    vectorstore = create_vector_store(docs)
    qa_chain = create_qa_chain(vectorstore)

    st.success("Document ready. Ask your questions below!")

    query = st.text_input("Ask a question about the document:")
    if query:
        result = qa_chain({"query": query})
        st.markdown("### 📌 Answer:")
        st.write(result['result'])

        st.markdown("---")
        st.markdown("### 📚 Sources:")
        for doc in result['source_documents']:
            st.write("-", doc.metadata)
else:
    st.warning("Please upload a PDF to begin.")
