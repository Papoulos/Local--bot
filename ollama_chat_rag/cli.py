import typer
import requests
import uvicorn
from multiprocessing import Process
import subprocess
import shlex
import os
from . import build_vector_store

app = typer.Typer()

@app.command()
def build_db(
    docs_path: str = typer.Option("documents", "--docs-path", "-d", help="Path to the documents directory."),
    store_path: str = typer.Option("vector_store", "--store-path", "-s", help="Path to save the vector store."),
    model_name: str = typer.Option("mistral", "--model", "-m", help="Name of the Ollama model to use for embeddings.")
):
    """
    Build the vector store from documents.
    """
    script_dir = os.path.dirname(__file__)
    docs_path_full = os.path.join(script_dir, docs_path)
    store_path_full = os.path.join(script_dir, store_path)
    build_vector_store.create_vector_store(docs_path_full, store_path_full, model_name)


def run_app(prod: bool = False, port: int = 8000):
    """Run the FastAPI application."""
    if prod:
        host = "127.0.0.1"
        workers = (os.cpu_count() or 1) * 2 + 1
        config_path = os.path.join(os.path.dirname(__file__), "gunicorn.conf.py")
        command = f"gunicorn -w {workers} -k uvicorn.workers.UvicornWorker --bind {host}:{port} -c {config_path} ollama_chat_rag.main:app"

        print(f"Starting production server with command: {command}")
        try:
            subprocess.run(shlex.split(command))
        except FileNotFoundError:
            print("\nError: 'gunicorn' command not found.")
            print("Please install it with: pip install gunicorn")

    else:
        uvicorn.run("ollama_chat_rag.main:app", host="0.0.0.0", port=port, reload=False)


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    print("Error: API_KEY not found in environment variables.")
    print("Please create a .env file and set the API_KEY.")
    exit(1)

@app.command()
def chat(
    message: str,
    model: str = typer.Option("chat", "--model", "-m", help="The model to use for the chat."),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="The URL of the API server.")
):
    """
    Chat with a model via the API.
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}]
    }

    try:
        response = requests.post(f"{url}/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        response_data = response.json()

        if response_data.get("choices"):
            content = response_data["choices"][0]["message"]["content"]
            print(f"\n🤖 Assistant:\n{content}")
        else:
            print(f"\nReceived an unexpected response:\n{response_data}")

    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to the API: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

@app.command()
def start(
    background: bool = typer.Option(False, "--background", "-b", help="Run the server in the background."),
    mock: bool = typer.Option(False, "--mock", help="Use mock Ollama for testing."),
    prod: bool = typer.Option(False, "--prod", help="Run in production mode with Gunicorn."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the server on.")
):
    """
    Start the FastAPI application.

    In development mode (default), it uses Uvicorn and listens on 0.0.0.0:8000.
    In production mode (--prod), it uses Gunicorn and listens on 127.0.0.1:8000.
    """
    if mock:
        os.environ["USE_MOCK_OLLAMA"] = "true"

    # Use a lambda to pass the arguments to run_app
    target_func = lambda: run_app(prod=prod, port=port)

    if background:
        p = Process(target=target_func)
        p.start()
        mode = "production (Gunicorn)" if prod else "development (Uvicorn)"
        print(f"FastAPI application started in the background in {mode} mode on port {port}.")
    else:
        run_app(prod=prod, port=port)

if __name__ == "__main__":
    app()
