from langchain_ollama import ChatOllama
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory


from langchain.chains import ConversationChain

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

class LLMusecase:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a chatbot having a conversation with a human. your name is Llama-Han"
                ),  # The persistent system prompt
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # Where the memory will be stored.
                HumanMessagePromptTemplate.from_template(
                    "{input}"
                ),  # Where the human input will injected
            ]
        )

    def generate(self, request) -> None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm_chain = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=memory,
        )

        predict = llm_chain.predict(input=request["input"])

        return {
            "response": predict
        } 