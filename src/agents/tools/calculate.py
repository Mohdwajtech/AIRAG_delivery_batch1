from __future__ import annotations
import ast
import operator
from pydantic import BaseModel, Field
from .base import Tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression — only + - * / % ** ( ) and numbers")


class CalculateArgs(BaseModel):
    expression: str = Field(
        ..., description="An arithmetic expression, e.g. '1350 * 0.18' or '(42 + 8) / 2'.")


class CalculateTool(Tool):
    name = "calculate"
    description = (
        "Evaluate an arithmetic expression and return the numeric result. "
        "Use this whenever a question needs exact math.")
    Args = CalculateArgs
    idempotent = True
    has_side_effects = False

    def run(self, expression: str) -> str:
        try:
            result = _eval(ast.parse(expression, mode="eval"))
        except Exception as e:
            return f"calculate error: {e}"
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"
