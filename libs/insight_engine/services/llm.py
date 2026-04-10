"""
LLM service for language model interactions
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config.settings import OPENAI_MODEL, TEMPERATURE


class LLMService:
    """Service for interacting with the language model"""

    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE)

    def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with a prompt

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response content
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def get_llm(self):
        """Get the underlying LLM instance"""
        return self.llm
