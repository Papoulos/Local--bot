import typer
import requests
import uvicorn
from multiprocessing import Process
import os
from leann import LeannBuilder
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader


app = typer.Typer()


def load_documents(documents_path="documents"):
    """Load documents from the specified directory."""
    script_dir = os.path.dirname(__file__)
    documents_path = os.path.join(script_dir, documents_path)
    loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
    return loader.load()

def run_app(prod: bool = False):
    """Run the FastAPI application using Uvicorn."""
    if prod:
        # In production mode, we bind to localhost only for security when using a tunnel.
        # Uvicorn's multi-worker mode is not supported on Windows, so we run a single worker.
        # This is sufficient for the intended use case.
        host = "127.0.0.1"
        port = 8000
        print(f"Starting server in production mode on {host}:{port}")
        uvicorn.run("ollama_chat_rag.main:app", host=host, port=port, reload=False)
    else:
        # In development mode, we bind to all interfaces for easier local network access.
        host = "0.0.0.0"
        port = 8000
        print(f"Starting server in development mode on {host}:{port}")
        uvicorn.run("ollama_chat_rag.main:app", host=host, port=port, reload=False)


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
    prod: bool = typer.Option(False, "--prod", help="Run in production mode.")
):
    """
    Start the FastAPI application.

    In development mode (default), it listens on 0.0.0.0:8000.
    In production mode (--prod), it listens on 127.0.0.1:8000 for safer tunneling.
    """
    if mock:
        os.environ["USE_MOCK_OLLAMA"] = "true"

    if background:
        # Pass run_app and its arguments directly to Process.
        # This is compatible with multiprocessing on all platforms.
        p = Process(target=run_app, args=(prod,))
        p.start()
        mode = "production" if prod else "development"
        print(f"FastAPI application started in the background in {mode} mode.")
    else:
        run_app(prod=prod)

@app.command()
def build_leann_index(
    index_path: str = typer.Option("leann_index.leann", "--index-path", "-p", help="Path to save the leann index."),
    documents_path: str = typer.Option("documents", "--documents-path", "-d", help="Path to the documents directory.")
):
    """
    Build a leann index from the documents.
    """
    print("Starting leann index build...")
    documents = load_documents(documents_path)

    # Initialize LeannBuilder
    # Using 'hnsw' backend as a default, good for many use cases.
    builder = LeannBuilder(backend_name="hnsw")

    # Add documents to the builder
    for doc in documents:
        builder.add_text(doc.page_content)

    # Build and save the index
    # The index will be saved in the `ollama_chat_rag` directory.
    script_dir = Path(os.path.dirname(__file__))
    full_index_path = script_dir / index_path

    print(f"Building and saving index to {full_index_path}...")
    builder.build_index(str(full_index_path))

    print("Leann index built successfully.")


if __name__ == "__main__":
    app()
