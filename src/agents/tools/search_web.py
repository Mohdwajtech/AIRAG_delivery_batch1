from __future__ import annotations
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
from .base import Tool

FIXTURES_PATH = Path("data/web_fixtures.json")


class SearchWebArgs(BaseModel):
    query: str = Field(..., description="A concise web-search query.")
    max_results: int = Field(3, ge=1, le=5, description="How many results to return.")


class SearchWebTool(Tool):
    name = "search_web"
    description = (
        "Search the public web and return short result snippets. "
        "Use only when the answer is not in the internal knowledge base.")
    Args = SearchWebArgs
    idempotent = True
    has_side_effects = False

    def __init__(self, offline=None):
        self.offline = (not os.environ.get("TAVILY_API_KEY")) if offline is None else offline

    def _fixture(self, query: str, k: int) -> str:
        try:
            data = json.loads(FIXTURES_PATH.read_text())
        except Exception:
            data = {"_default": [{"title": "No fixture", "snippet": "offline stub."}]}
        results = None
        for key, res in data.items():
            if key != "_default" and key.lower() in query.lower():
                results = res
                break
        results = (results or data.get("_default", []))[:k]
        lines = [f"[web:{query}]"]
        for r in results:
            lines.append(f"- {r['title']}: {r['snippet']}")
        return "\n".join(lines)

    def run(self, query: str, max_results: int = 3) -> str:
        if self.offline:
            return self._fixture(query, max_results)
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            resp = client.search(query=query, max_results=max_results)
            lines = [f"[web:{query}]"]
            for r in resp.get("results", [])[:max_results]:
                lines.append(f"- {r.get('title','')}: {r.get('content','')[:160]}")
            return "\n".join(lines)
        except Exception as e:
            return f"search_web error: {e}\n" + self._fixture(query, max_results)
