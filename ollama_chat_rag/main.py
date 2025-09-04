import os
import json
import uuid
import time
import logging
import sys
from typing import List, AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from leann import LeannSearcher


# Conditionally import real or mock classes based on environment variable
if os.getenv("USE_MOCK_OLLAMA", "false").lower() == "true":
    from .mock_ollama import MockChatOllama as ChatOllama
    from .mock_ollama import MockOllamaEmbeddings as OllamaEmbeddings
else:
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])
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
leann_searcher_cache = {}

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
    stream: bool = False

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

# Pydantic models for streaming
class ChoiceDelta(BaseModel):
    content: str | None = None
    role: str | None = None

class ChoiceChunk(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = int(time.time())
    model: str
    choices: List[ChoiceChunk]


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

async def stream_generator(model_key: str, user_message: str, llm, model_config: dict) -> AsyncGenerator[str, None]:
    """Yields server-sent events for streaming responses."""
    request_id = f"chatcmpl-{uuid.uuid4()}"
    model_type = model_config["type"]
    model_name = model_config["model_name"]

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_key,
        choices=[ChoiceChunk(delta=ChoiceDelta(role="assistant"))]
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    if model_type == "llm":
        async for chunk in llm.astream(user_message):
            chunk_delta = ChatCompletionChunk(
                id=request_id,
                model=model_key,
                choices=[ChoiceChunk(delta=ChoiceDelta(content=chunk.content))]
            )
            yield f"data: {chunk_delta.model_dump_json()}\n\n"

    elif model_type == "rag":
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
        rag_type = model_config.get("rag_type", "faiss") # Default to faiss for backward compatibility

        if rag_type == "faiss":
            if model_name not in db_cache:
                db_cache[model_name] = create_vector_store(model_name=model_name)
            db = db_cache[model_name]
            retriever = db.as_retriever()

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            async for chunk in rag_chain.astream(user_message):
                chunk_delta = ChatCompletionChunk(
                    id=request_id,
                    model=model_key,
                    choices=[ChoiceChunk(delta=ChoiceDelta(content=chunk))]
                )
                yield f"data: {chunk_delta.model_dump_json()}\n\n"

        elif rag_type == "leann":
            index_path = model_config["index_path"]
            full_index_path = os.path.join(script_dir, index_path)

            if full_index_path not in leann_searcher_cache:
                if not os.path.exists(full_index_path):
                    raise FileNotFoundError(f"Leann index not found at {full_index_path}. Please build it first using the 'build-leann-index' command.")
                leann_searcher_cache[full_index_path] = LeannSearcher(full_index_path)

            searcher = leann_searcher_cache[full_index_path]
            results = searcher.search(user_message, top_k=3)
            context = "\n\n".join([res["text"] for res in results])

            rag_chain = (
                prompt
                | llm
                | StrOutputParser()
            )

            async for chunk in rag_chain.astream({"context": context, "question": user_message}):
                chunk_delta = ChatCompletionChunk(
                    id=request_id,
                    model=model_key,
                    choices=[ChoiceChunk(delta=ChoiceDelta(content=chunk))]
                )
                yield f"data: {chunk_delta.model_dump_json()}\n\n"

    # Final chunk with finish reason
    final_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_key,
        choices=[ChoiceChunk(delta=ChoiceDelta(), finish_reason="stop")]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    user_message = ""
    for msg in reversed(request_data.messages):
        if msg.role == 'user':
            user_message = msg.content
            break

    logging.info(f"Request - IP: {request.client.host}, Method: {request.method}, Keyword: {user_message}, Stream: {request_data.stream}")

    model_key = request_data.model
    if model_key not in config["models"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")

    model_config = config["models"][model_key]
    model_type = model_config["type"]
    model_name = model_config["model_name"]

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    if model_name not in llm_cache:
        llm_cache[model_name] = ChatOllama(model=model_name)
    llm = llm_cache[model_name]

    if request_data.stream:
        # For streaming, we return a StreamingResponse
        return StreamingResponse(
            stream_generator(model_key, user_message, llm, model_config),
            media_type="text/event-stream"
        )
    else:
        # Original non-streaming logic
        response_content = ""
        if model_type == "llm":
            response = llm.invoke(user_message)
            response_content = response.content
        elif model_type == "rag":
            rag_type = model_config.get("rag_type", "faiss")

            if rag_type == "faiss":
                if model_name not in db_cache:
                    db_cache[model_name] = create_vector_store(model_name=model_name)
                db = db_cache[model_name]

                qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                response = qa_chain.invoke({"query": user_message})
                response_content = response["result"]

            elif rag_type == "leann":
                index_path = model_config["index_path"]
                full_index_path = os.path.join(script_dir, index_path)

                if full_index_path not in leann_searcher_cache:
                    if not os.path.exists(full_index_path):
                        raise FileNotFoundError(f"Leann index not found at {full_index_path}. Please build it first using the 'build-leann-index' command.")
                    leann_searcher_cache[full_index_path] = LeannSearcher(full_index_path)

                searcher = leann_searcher_cache[full_index_path]
                results = searcher.search(user_message, top_k=3)
                context = "\n\n".join([res["text"] for res in results])

                template = """Answer the question based only on the following context:
                {context}

                Question: {question}
                """
                prompt = PromptTemplate.from_template(template)

                chain = prompt | llm | StrOutputParser()
                response_content = chain.invoke({"context": context, "question": user_message})

        else:
            raise HTTPException(status_code=500, detail=f"Unsupported model type: {model_type}")

        response_message = ResponseMessage(role="assistant", content=response_content)
        choice = Choice(message=response_message)
        logging.info(f"Response - Destination IP: {request.client.host}")
        return ChatCompletionResponse(
            model=model_key,
            choices=[choice]
        )
