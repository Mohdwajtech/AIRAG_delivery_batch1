from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class Tool:
    name: str = ""
    description: str = ""
    Args: type[BaseModel] = BaseModel
    idempotent: bool = True
    has_side_effects: bool = False

    def run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def call(self, arguments: dict[str, Any]) -> str:
        args = self.Args(**(arguments or {}))
        return self.run(**args.model_dump())

    def parameters_schema(self) -> dict:
        return self.Args.model_json_schema()

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    def anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema(),
        }
