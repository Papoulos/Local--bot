# Local AI Bot API Gateway

This project provides a secure and configurable API gateway to interact with local language models running via [Ollama](https://ollama.ai/). It acts as a middleware, allowing you to define specific "models" in a configuration file that can point to different Ollama models or even trigger complex chains like Retrieval-Augmented Generation (RAG).

All API calls are secured by a secret API key. The request and response formats are standardized to be compatible with OpenAI's Chat Completions API, making it easy to integrate with existing tools.

## Features

- **Configurable Model Routing:** Define keywords (e.g., "chat", "cypher_expert") in a `config.json` file and map them to different Ollama models or processing chains.
- **API Key Authentication:** Protect your local models with a secret API key.
- **Standardized API:** OpenAI-compatible request and response formats for easy integration.
- **RAG Support:** Built-in support for Retrieval-Augmented Generation to chat with your documents.
- **Streaming Support:** OpenAI-compatible streaming of responses using server-sent events.
- **Simple CLI:** A command-line interface to easily start the server.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ollama-api-gateway
    ```

2.  **Install Dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r ollama_chat_rag/requirements.txt
    ```

3.  **Configure Environment:**
    Copy the example `.env` file and edit it to set your secret API key.
    ```bash
    cp ollama_chat_rag/.env.example ollama_chat_rag/.env
    ```
    Now, open `ollama_chat_rag/.env` and change the `API_KEY`.

4.  **Configure Models:**
    Open `ollama_chat_rag/config.json` to define your model mappings. The `type` can be `llm` for a standard model call or `rag` to use the document retrieval system.
    ```json
    {
      "models": {
        "chat": {
          "type": "llm",
          "model_name": "mistral"
        },
        "cypher": {
          "type": "rag",
          "model_name": "mistral"
        }
      }
    }
    ```

5.  **(For RAG) Add Documents:**
    If you are using a `rag` type model, place your `.txt` files inside the `ollama_chat_rag/documents/` directory.

## How to Run

Start the FastAPI server using the built-in command-line interface:

```bash
python -m ollama_chat_rag.cli start
```

The API will be available at `http://localhost:8000`.

## Secure Deployment

To securely publish this API on the internet, please follow the detailed instructions in the deployment guide:

**[➡️ Secure Deployment Guide](./DEPLOYMENT.md)**

### Running in Mock Mode for Testing

For development or testing purposes, you can run the server in a "mock" mode. In this mode, it does not connect to a real Ollama instance. Instead, it uses mock objects that return predictable, pre-defined responses.

To start the server in mock mode, use the `--mock` flag:

```bash
python -m ollama_chat_rag.cli start --mock
```

You can then make API calls as usual, but you will receive mock responses, which is useful for testing your client application without relying on a running AI model.

## How to Use the API

You can interact with the API using any HTTP client, like `curl`.

-   Replace `YOUR_API_KEY` with the key you set in the `.env` file.
-   The `model` in the JSON body should be one of the keys you defined in `config.json`.

### Example: Standard Chat (`llm` type)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "X-API-Key: YOUR_API_KEY" \
-d '{
    "model": "chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ],
    "stream": false
}'
```

### Example: RAG Chat (`rag` type)

This will use the documents in the `documents` folder to answer the question.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "X-API-Key: YOUR_API_KEY" \
-d '{
    "model": "cypher",
    "messages": [
        {"role": "user", "content": "What is the main topic of the document?"}
    ]
}'
```

### Example: Streaming Response

To receive a streaming response, set `"stream": true` in your request. The server will respond with a `text/event-stream` payload. Use a client that can handle server-sent events. With `curl`, the `-N` flag is useful to disable buffering.

```bash
curl -X POST -N http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "X-API-Key: YOUR_API_KEY" \
-d '{
    "model": "chat",
    "messages": [
        {"role": "user", "content": "Tell me a short story."}
    ],
    "stream": true
}'
```

The output will be a series of JSON objects, each prefixed with `data: `.

```text
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"chat","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"chat","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"chat","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"chat","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"chat","choices":[{"index":0,"delta":{"content":" time"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

The stream is terminated by a final message `data: [DONE]`.
