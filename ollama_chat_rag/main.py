import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class RAGRequest(BaseModel):
    question: str

def create_vector_store(documents_path="documents"):
    script_dir = os.path.dirname(__file__)
    documents_path = os.path.join(script_dir, documents_path)
    loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL"))
    db = FAISS.from_documents(texts, embeddings)
    return db

db = create_vector_store()
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))

@app.post("/chat")
async def chat(request: ChatRequest):
    response = llm.invoke(request.message)
    return {"response": response.content}

@app.post("/rag")
async def rag(request: RAGRequest):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    response = qa_chain.invoke({"query": request.question})
    return {"response": response["result"]}
