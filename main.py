import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone

# --- Load Environment Variables ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not GOOGLE_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    st.error("🚨 Missing environment variables. Check your .env file.")
    st.stop()

# --- Load PDF Document ---
def load_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# --- Split PDF into Chunks ---
def split_text(pages):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(pages)

# --- Create or Use Pinecone Vector Store ---
def create_vector_store(docs):
    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX, dimension=768, metric="cosine")

    # Use LangChain's Pinecone wrapper
    return LangchainPinecone.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX
    )

# --- Create QA Chain ---
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# --- Streamlit UI ---
st.set_page_config(page_title="DocuBrain", page_icon="📄")
st.title("📄 DocuBrain – Ask Questions from Your PDF")

uploaded_file = st.file_uploader("📎 Upload your PDF document", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    st.info("📑 Processing document...")

    try:
        pages = load_document(temp_path)
        docs = split_text(pages)
        vectorstore = create_vector_store(docs)
        qa_chain = create_qa_chain(vectorstore)

        st.success("✅ Document ready! Ask your question below.")
        query = st.text_input("❓ Ask a question:")

        if query:
            result = qa_chain({"query": query})
            st.markdown("### 📌 Answer:")
            st.write(result["result"])

            st.markdown("---")
            st.markdown("### 📚 Source Metadata:")
            for doc in result["source_documents"]:
                st.json(doc.metadata)

    except Exception as e:
        st.error(f"🚨 Error while processing: {e}")
else:
    st.warning("📥 Please upload a PDF to get started.")
