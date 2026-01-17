"""Base agent class with common LLM interaction logic."""

import json
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import get_config


T = TypeVar("T", bound=BaseModel)


def create_llm():
    """Create LLM instance based on config."""
    config = get_config()
    
    if config.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.gemini_api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )


class BaseAgent(ABC):
    """
    Base class for all screening agents.
    
    Each agent has a clear responsibility and processes a specific part
    of the screening workflow. Agents communicate through structured data
    (Pydantic models) that flows through the shared state.
    """
    
    name: str = "BaseAgent"
    description: str = "Base agent class"
    
    def __init__(self, llm=None):
        """
        Initialize the agent with an LLM.
        
        Args:
            llm: Optional LLM instance. If not provided, creates one from config.
        """
        if llm is None:
            self.llm = create_llm()
        else:
            self.llm = llm
    
    @abstractmethod
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Process the current state and return updates.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of state updates
        """
        raise NotImplementedError
    
    def _call_llm(self, prompt: str) -> str:
        """
        Make a synchronous LLM call.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    async def _call_llm_async(self, prompt: str) -> str:
        """
        Make an asynchronous LLM call.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def _parse_json_response(self, response: str, model_class: type[T]) -> T | None:
        """
        Parse a JSON response from the LLM into a Pydantic model.
        
        Args:
            response: The LLM response text (should contain JSON)
            model_class: The Pydantic model class to parse into
            
        Returns:
            Parsed model instance or None if parsing fails
        """
        try:
            # Try to extract JSON from the response
            # LLMs sometimes wrap JSON in markdown code blocks
            json_str = response
            
            # Remove markdown code blocks if present
            if "```json" in json_str:
                start = json_str.find("```json") + 7
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            elif "```" in json_str:
                start = json_str.find("```") + 3
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            
            # Parse JSON
            data = json.loads(json_str)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[{self.name}] Failed to parse JSON response: {e}")
            print(f"[{self.name}] Raw response: {response[:500]}...")
            return None
    
    def _extract_json_from_response(self, response: str) -> dict | None:
        """
        Extract JSON dictionary from LLM response.
        
        Args:
            response: The LLM response text
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            json_str = response
            
            # Remove markdown code blocks if present
            if "```json" in json_str:
                start = json_str.find("```json") + 7
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            elif "```" in json_str:
                start = json_str.find("```") + 3
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for this agent.
        
        Returns:
            System prompt string
        """
        return f"""You are {self.name}, a specialized AI agent in a resume screening system.
Your role: {self.description}

IMPORTANT GUIDELINES:
1. Always respond with valid JSON as specified in the prompt
2. Be thorough but concise in your analysis
3. When uncertain, indicate low confidence rather than guessing
4. Focus only on your specific task - other agents handle other aspects
5. Provide reasoning for your conclusions

Remember: Your output will be used by other agents in the pipeline, so accuracy is crucial."""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
