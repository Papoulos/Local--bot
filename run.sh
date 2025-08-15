#!/bin/bash
pip install -r ollama_chat_rag/requirements.txt
python3 ollama_chat_rag/cli.py "$@"
