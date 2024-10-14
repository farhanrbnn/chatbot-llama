from langchain_ollama import ChatOllama
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

class LLMusecase:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model="llama3.1:8b", temperature=0
        )
        self.store_chat = None 
        self.prompt = """You are a chatbot having a conversation with a human. your name is Llama-Han it stands for Llama model and farhan as a creator. If the AI does not know the answer to a question, it truthfully says it does not know.
                    Current conversation:
                    {history}
                    Human: {input}
                    AI Assistant:"""

    def generate(self, request) -> dict:
        predict = None
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.prompt)

        if self.store_chat is None:
            memory = ConversationBufferMemory(memory_key="history", return_messages=True)

            llm_chain = ConversationChain(
                llm=self.llm,
                prompt=PROMPT,
                verbose=True,
                memory=memory,
            )

            predict = llm_chain.predict(input=request["input"])

            extracted_messages = llm_chain.memory.chat_memory.messages
            to_dict = messages_to_dict(extracted_messages)

            self.store_chat = to_dict
        else:
            retrieved_message = messages_from_dict(self.store_chat)
            retrieved_chat_history = ChatMessageHistory(messages=retrieved_message)

            retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history)

            reload_chain = ConversationChain(
                llm=self.llm,
                prompt=PROMPT,
                verbose=True,
                memory=retrieved_memory,
            ) 
 
            predict = reload_chain.predict(input=request["input"])

            extracted_messages = reload_chain.memory.chat_memory.messages
            to_dict = messages_to_dict(extracted_messages)

            self.store_chat = to_dict

        return {
            "response": predict
        } 