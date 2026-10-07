# TODO 2026-10-06T15:50:12+05:30
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    """Abstract interface for LLM providers (OpenAI, Bedrock, Claude, etc.)"""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: float = 0.2,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Invoke chat completion with optional tools."""
        pass
