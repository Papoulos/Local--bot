from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSerializable
import asyncio

class MockChatOllama(RunnableSerializable):
    def invoke(self, message: str, **kwargs):
        return AIMessage(content=f"Mock response to: {message}")

    async def ainvoke(self, message: str, **kwargs):
        await asyncio.sleep(0.1)
        return AIMessage(content=f"Mock response to: {message}")

    async def astream(self, message: str, **kwargs):
        words = f"Mock response to: {message}".split()
        for word in words:
            await asyncio.sleep(0.05)
            yield AIMessage(content=word + " ")

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
