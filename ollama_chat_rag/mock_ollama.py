from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel

class MockChatOllama(BaseModel):
    def invoke(self, message: str, **kwargs):
        return AIMessage(content=f"Mock response to: {message}")

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

class MockOllamaEmbeddings(BaseModel):
    def embed_documents(self, texts: list[str]):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, text: str):
        return [0.1] * 768
