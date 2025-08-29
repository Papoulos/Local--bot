import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS

# Conditionally import real or mock classes based on environment variable
if os.getenv("USE_MOCK_OLLAMA", "false").lower() == "true":
    from .mock_ollama import MockOllamaEmbeddings as OllamaEmbeddings
else:
    from langchain_ollama import OllamaEmbeddings

def create_vector_store(documents_path: str, vector_store_path: str, model_name: str = "mistral"):
    """
    Creates a vector store from documents and saves it to disk.
    """
    print(f"Loading documents from: {documents_path}")
    loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
    documents = loader.load()

    if not documents:
        print("No documents found. Aborting.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"Creating embeddings with model '{model_name}'... This may take a while.")
    embeddings = OllamaEmbeddings(model=model_name)

    print("Creating FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)

    print(f"Saving vector store to: {vector_store_path}")
    db.save_local(vector_store_path)
    print("Vector store created successfully.")
