import os
import json
import uuid
import time
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Depends, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pydantic import BaseModel

# Conditionally import real or mock classes based on environment variable
if os.getenv("USE_MOCK_OLLAMA", "false").lower() == "true":
    from .mock_ollama import MockChatOllama as ChatOllama
    from .mock_ollama import MockOllamaEmbeddings as OllamaEmbeddings
else:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

app = FastAPI()

# Load config
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

API_KEY = os.getenv("API_KEY")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Pydantic models for the new API structure
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

class ResponseMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = str(uuid.uuid4())
    object: str = "chat.completion"
    created: int = int(time.time())
    model: str
    choices: List[Choice]

def create_vector_store(documents_path="documents", model_name="mistral"):
    script_dir = os.path.dirname(__file__)
    documents_path = os.path.join(script_dir, documents_path)
    loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=model_name)
    db = FAISS.from_documents(texts, embeddings)
    return db

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    model_key = request.model
    if model_key not in config["models"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")

    model_config = config["models"][model_key]
    model_type = model_config["type"]
    model_name = model_config["model_name"]

    # For simplicity, we take the last user message as the prompt
    # A more advanced implementation would handle the full conversation history
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == 'user':
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    response_content = ""
    llm = ChatOllama(model=model_name)

    if model_type == "llm":
        response = llm.invoke(user_message)
        response_content = response.content
    elif model_type == "rag":
        db = create_vector_store(model_name=model_name)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        response = qa_chain.invoke({"query": user_message})
        response_content = response["result"]
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported model type: {model_type}")

    response_message = ResponseMessage(role="assistant", content=response_content)
    choice = Choice(message=response_message)

    return ChatCompletionResponse(
        model=model_key,
        choices=[choice]
    )
