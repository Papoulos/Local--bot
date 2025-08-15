from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.pydantic_v1 import BaseModel

class MockChatOllama(BaseModel):
    def invoke(self, message: str):
        return ChatGeneration(message=f"Mock response to: {message}")

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

class MockOllamaEmbeddings(BaseModel):
    def embed_documents(self, texts: list[str]):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, text: str):
        return [0.1] * 768
