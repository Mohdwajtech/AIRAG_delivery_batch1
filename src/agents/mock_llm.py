from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Callable
from .tools.base import Tool


@dataclass
class LLMResponse:
    kind: str
    tool_name: str | None = None
    tool_args: dict | None = None
    content: str | None = None

    @staticmethod
    def tool(name: str, **args) -> "LLMResponse":
        return LLMResponse("tool_use", tool_name=name, tool_args=args)

    @staticmethod
    def final(content: str) -> "LLMResponse":
        return LLMResponse("final", content=content)


class MockLLM:
    def __init__(self, script=None, router=None):
        self.script = list(script) if script else None
        self.router = router
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.script:
            return self.script.pop(0) if self.script else LLMResponse.final("(no more)")
        if self.router:
            return self.router(messages, tools)
        return self._keyword_route(messages, tools)

    def _keyword_route(self, messages, tools):
        obs = _last_observation(messages)
        if obs is not None:
            return LLMResponse.final(f"Based on the tool result: {obs}")
        q = _last_user(messages).lower()
        names = {t.name for t in tools}
        if "calculate" in names and re.search(r"\d\s*[-+*/%]\s*\d|percent|average", q):
            return LLMResponse.tool("calculate", expression=_extract_expr(q))
        sql_kw = ("how many doc", "which department", "what version", "list the",
                  "count", "how many hr", "how many it", "how many finance", "updated", "chunk")
        if "query_sql" in names and any(k in q for k in sql_kw):
            return LLMResponse.tool("query_sql",
                sql="SELECT COUNT(*) AS total_documents FROM document_metadata",
                purpose="metadata question")
        web_kw = ("latest", "current", "today", "news", "price", "recent", "search", "trend")
        if "search_web" in names and any(k in q for k in web_kw):
            return LLMResponse.tool("search_web", query=_last_user(messages), max_results=3)
        email_kw = ("send", "email", "mail", "notify")
        if "send_email" in names and any(k in q for k in email_kw):
            return LLMResponse.tool("send_email", to="user@example.com",
                subject="Agent-generated message", body=_last_user(messages))
        return LLMResponse.final("I can answer that directly: no tool needed.")


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def complete(self, messages, tools):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            tools=[t.openai_schema() for t in tools], tool_choice="auto")
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return LLMResponse.tool(call.function.name,
                                    **json.loads(call.function.arguments or "{}"))
        return LLMResponse.final(msg.content or "")


def _last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""

def _last_observation(messages):
    for m in reversed(messages):
        if m.get("role") == "tool":
            return str(m.get("content", ""))
    return None

def _extract_expr(q):
    m = re.search(r"[0-9][0-9\.\s\+\-\*/%\(\)]*[0-9\)]", q)
    return m.group(0).strip() if m else "0"
