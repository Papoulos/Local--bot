import os
import json
import uuid
import time
import logging
import sys
import asyncio
from typing import List, AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Conditionally import real or mock classes based on environment variable
if os.getenv("USE_MOCK_OLLAMA", "false").lower() == "true":
    from .mock_ollama import MockChatOllama as ChatOllama
else:
    from langchain_ollama import ChatOllama

load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this should be restricted
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

API_KEY = os.getenv("API_KEY")

# In-memory cache for expensive objects
llm_cache = {}

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


async def stream_generator(model_key: str, user_message: str, llm, model_config: dict) -> AsyncGenerator[str, None]:
    """Yields server-sent events for streaming responses."""
    request_id = f"chatcmpl-{uuid.uuid4()}"
    model_name = model_config["model_name"]

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_key,
        choices=[ChoiceChunk(delta=ChoiceDelta(role="assistant"))]
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    async for chunk in llm.astream(user_message):
        chunk_delta = ChatCompletionChunk(
            id=request_id,
            model=model_key,
            choices=[ChoiceChunk(delta=ChoiceDelta(content=chunk.content))]
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
@limiter.limit("60/minute")
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
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
        response = await llm.ainvoke(user_message)
        response_content = response.content

        response_message = ResponseMessage(role="assistant", content=response_content)
        choice = Choice(message=response_message)
        logging.info(f"Response - Destination IP: {request.client.host}")
        return ChatCompletionResponse(
            model=model_key,
            choices=[choice]
        )
