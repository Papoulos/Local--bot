import os
import json
import uuid
import time
import logging
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Depends, HTTPException, Request
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
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
# File handler
file_handler = logging.FileHandler("chat.log", mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


app = FastAPI()

# Load config
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

API_KEY = os.getenv("API_KEY")

# In-memory cache for expensive objects
llm_cache = {}
db_cache = {}

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

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_api_key)])
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    # Log incoming request
    user_message = ""
    for msg in reversed(request_data.messages):
        if msg.role == 'user':
            user_message = msg.content
            break

    logging.info(f"Request - IP: {request.client.host}, Method: {request.method}, Keyword: {user_message}")

    model_key = request_data.model
    if model_key not in config["models"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")

    model_config = config["models"][model_key]
    model_type = model_config["type"]
    model_name = model_config["model_name"]

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    # Use cache to load/store LLM
    if model_name not in llm_cache:
        llm_cache[model_name] = ChatOllama(model=model_name)
    llm = llm_cache[model_name]

    response_content = ""

    if model_type == "llm":
        response = llm.invoke(user_message)
        response_content = response.content
    elif model_type == "rag":
        vector_store_path = model_config.get("vector_store_path")
        if not vector_store_path:
            raise HTTPException(status_code=500, detail=f"Vector store path not configured for model '{model_key}'.")

        # Construct the full path to the vector store directory
        script_dir = os.path.dirname(__file__)
        vector_store_full_path = os.path.join(script_dir, vector_store_path)

        if not os.path.isdir(vector_store_full_path):
            raise HTTPException(status_code=500, detail=f"Vector store directory not found at '{vector_store_full_path}'. Please build it first.")

        # Use a unique key for the db_cache based on the path
        db_cache_key = vector_store_full_path

        # Use cache to load/store vector database
        if db_cache_key not in db_cache:
            print(f"Loading vector store from: {vector_store_full_path}")
            embeddings = OllamaEmbeddings(model=model_name)
            db_cache[db_cache_key] = FAISS.load_local(vector_store_full_path, embeddings, allow_dangerous_deserialization=True)

        db = db_cache[db_cache_key]

        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        response = qa_chain.invoke({"query": user_message})
        response_content = response["result"]
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported model type: {model_type}")

    response_message = ResponseMessage(role="assistant", content=response_content)
    choice = Choice(message=response_message)

    logging.info(f"Response - Destination IP: {request.client.host}")

    return ChatCompletionResponse(
        model=model_key,
        choices=[choice]
    )
