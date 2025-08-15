import typer
import requests
import uvicorn
from multiprocessing import Process

app = typer.Typer()

def run_app():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

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

import os

@app.command()
def start(background: bool = typer.Option(False, "--background", "-b"), mock: bool = typer.Option(False, "--mock")):
    """
    Start the FastAPI application.
    """
    if mock:
        os.environ["USE_MOCK_OLLAMA"] = "true"

    if background:
        p = Process(target=run_app)
        p.start()
        print("FastAPI application started in the background.")
    else:
        run_app()

if __name__ == "__main__":
    app()
