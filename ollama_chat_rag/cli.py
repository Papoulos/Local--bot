import typer
import requests
import uvicorn
from multiprocessing import Process
import subprocess
import shlex
import os

app = typer.Typer()

def run_app(prod: bool = False):
    """Run the FastAPI application."""
    if prod:
        host = "127.0.0.1"
        port = 8000
        workers = (os.cpu_count() or 1) * 2 + 1
        command = f"gunicorn -w {workers} -k uvicorn.workers.UvicornWorker --bind {host}:{port} ollama_chat_rag.main:app"

        print(f"Starting production server with command: {command}")
        try:
            subprocess.run(shlex.split(command))
        except FileNotFoundError:
            print("\nError: 'gunicorn' command not found.")
            print("Please install it with: pip install gunicorn")

    else:
        uvicorn.run("ollama_chat_rag.main:app", host="0.0.0.0", port=8000, reload=False)


@app.command()
def chat(message: str):
    """
    Chat with the Ollama model.
    """
    response = requests.post("http://localhost:8000/chat", json={"message": message})
    print(response.json()["response"])

@app.command()
def rag(question: str):
    """
    Ask a question to the RAG system.
    """
    response = requests.post("http://localhost:8000/rag", json={"question": question})
    print(response.json()["response"])

@app.command()
def start(
    background: bool = typer.Option(False, "--background", "-b", help="Run the server in the background."),
    mock: bool = typer.Option(False, "--mock", help="Use mock Ollama for testing."),
    prod: bool = typer.Option(False, "--prod", help="Run in production mode with Gunicorn.")
):
    """
    Start the FastAPI application.

    In development mode (default), it uses Uvicorn and listens on 0.0.0.0:8000.
    In production mode (--prod), it uses Gunicorn and listens on 127.0.0.1:8000.
    """
    if mock:
        os.environ["USE_MOCK_OLLAMA"] = "true"

    # Use a lambda to pass the 'prod' argument to run_app
    target_func = lambda: run_app(prod=prod)

    if background:
        p = Process(target=target_func)
        p.start()
        mode = "production (Gunicorn)" if prod else "development (Uvicorn)"
        print(f"FastAPI application started in the background in {mode} mode.")
    else:
        run_app(prod=prod)

if __name__ == "__main__":
    app()
