from langchain_ollama import ChatOllama

class Ollama:
    def __init__(self, model) -> None:
        self.model = model


    def connect(self) -> None:
        llm = ChatOllama(
            model=self.model,
            temperature=1
        )

        return llm
