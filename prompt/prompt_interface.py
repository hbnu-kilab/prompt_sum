from abc import ABC
from typing import Any


class PromptInterface(ABC):
    def do_llm(Any) -> Any:
        """Do LLM Prompting."""