"""wk06_pipeline.py — the naive RAG pipeline built in Day 1.

Day 2 imports these functions so we don't spend 20 minutes rebuilding.
The code here is IDENTICAL to what Day 1 constructs cell-by-cell —
learners can read this file and see the whole pipeline in one place.

Pure functions. No frameworks. ~80 lines.
"""
from __future__ import annotations

import os
import numpy as np
from openai import OpenAI

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before importing"

_client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"


# ─── Chunking ────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Sliding window over characters. Simplest possible chunker."""
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = end - overlap
    return chunks


def chunk_documents(documents: list[dict], size: int = 200,
                    overlap: int = 40) -> list[dict]:
    """Chunk every document. Returns flat list with source pointers."""
    all_chunks = []
    for doc in documents:
        for chunk_idx, chunk in enumerate(chunk_text(doc["text"], size, overlap)):
            all_chunks.append({
                "chunk_id":  f"{doc['id']}#{chunk_idx}",
                "source_id": doc["id"],
                "text":      chunk,
            })
    return all_chunks


# ─── Embedding ───────────────────────────────────────────────────────

def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """One API call, list of vectors back."""
    resp = _client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def build_index(chunks: list[dict], model: str = EMBED_MODEL) -> list[dict]:
    """Attach 'vector' field to each chunk. Returns the same list mutated."""
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts, model=model)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec
    return chunks


# ─── Similarity + retrieval ──────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def retrieve(query: str, index: list[dict], k: int = 3,
             embed_model: str = EMBED_MODEL) -> list[dict]:
    """Embed the query, rank chunks by cosine, return top-K with scores."""
    q_vec = embed_batch([query], model=embed_model)[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**c, "score": s} for s, c in scored[:k]]


# ─── Prompt + generate ───────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say so "
    "plainly. Cite the source id in square brackets after any fact you use."
)


def build_prompt(question: str, retrieved: list[dict],
                 system: str = DEFAULT_SYSTEM) -> tuple[str, str]:
    """Return (system_message, user_message) so we can inspect them."""
    context = "\n\n".join(
        f"[{hit['chunk_id']}]\n{hit['text']}"
        for hit in retrieved
    )
    user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    return system, user_msg


def ask_rag(question: str, index: list[dict], k: int = 3,
            system: str = DEFAULT_SYSTEM,
            embed_model: str = EMBED_MODEL,
            chat_model: str = CHAT_MODEL) -> dict:
    """Full pipeline: retrieve → prompt → generate. Returns dict with
    answer, sources, cost, latency-relevant token counts."""
    retrieved = retrieve(question, index, k=k, embed_model=embed_model)
    system_msg, user_msg = build_prompt(question, retrieved, system=system)
    resp = _client.chat.completions.create(
        model=chat_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
    )
    return {
        "question":   question,
        "answer":     resp.choices[0].message.content,
        "sources":    [hit["chunk_id"] for hit in retrieved],
        "tokens_in":  resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "retrieved":  retrieved,  # full retrieved chunks for inspection
    }


# ─── Corpus for Day 1 and Day 2 (shared) ─────────────────────────────

CORPUS = [
    # Coffee
    {"id": "coffee_espresso",
     "text": ("Espresso is a concentrated form of coffee made by forcing hot water "
              "under about 9 bars of pressure through finely ground coffee beans. "
              "A single shot is typically 25 to 30 millilitres and takes 25 to 30 "
              "seconds to extract. Espresso forms the base of drinks like the "
              "latte, cappuccino, and americano.")},
    {"id": "coffee_beans",
     "text": ("Coffee beans come primarily from two species: Arabica and Robusta. "
              "Arabica accounts for about 60 percent of world production and is "
              "prized for its smoother, more nuanced flavour. Robusta contains "
              "roughly twice as much caffeine and has a stronger, more bitter taste. "
              "Most commercial espresso blends mix the two.")},
    {"id": "coffee_brewing",
     "text": ("Pour-over coffee uses a filter cone to drip near-boiling water "
              "through medium-ground coffee. It typically brews for three to four "
              "minutes and produces a clean, light-bodied cup. French press coffee, "
              "in contrast, steeps coarse grounds directly in hot water for four "
              "minutes before pressing, producing a heavier, oil-rich cup.")},
    # Tea
    {"id": "tea_green",
     "text": ("Green tea is made from unoxidised leaves of Camellia sinensis. It is "
              "steeped in water at around 70 to 80 degrees Celsius for one to three "
              "minutes. Hotter water or longer steeping produces a bitter, astringent "
              "cup. Green tea is high in an antioxidant called EGCG.")},
    {"id": "tea_black",
     "text": ("Black tea comes from fully oxidised Camellia sinensis leaves. It is "
              "brewed with water at or near boiling — 95 to 100 degrees Celsius — "
              "for three to five minutes. Popular varieties include Assam, Darjeeling, "
              "and Ceylon. Black tea typically contains more caffeine than green tea.")},
    {"id": "tea_oolong",
     "text": ("Oolong tea is partially oxidised, sitting between green and black tea "
              "in strength and colour. It is brewed at 85 to 95 degrees Celsius for "
              "two to four minutes. Oolong leaves are often rolled and can be re-steeped "
              "several times, with each infusion revealing different flavour notes.")},
    # Hot chocolate
    {"id": "chocolate_traditional",
     "text": ("Traditional hot chocolate is made from melted dark chocolate stirred "
              "into hot milk. The ratio is usually 30 to 50 grams of chocolate per "
              "200 millilitres of milk. Whisking prevents the chocolate from settling. "
              "Some recipes add a pinch of chilli or cinnamon for warmth.")},
    {"id": "chocolate_powder",
     "text": ("Instant hot chocolate uses cocoa powder mixed with sugar, milk powder, "
              "and stabilisers. Adding hot water dissolves the mix in seconds. It is "
              "cheaper and faster than the traditional method but has a thinner mouthfeel "
              "and less intense chocolate flavour.")},
    {"id": "chocolate_history",
     "text": ("Hot chocolate originated with the Maya and Aztec civilisations, who "
              "drank it cold and bitter, spiced with chilli. Europeans encountered "
              "cacao in the 16th century and gradually sweetened the drink and "
              "served it hot. It remained a luxury until industrial cocoa processing "
              "made it affordable in the 19th century.")},
    # Milk-based drinks
    {"id": "milk_latte",
     "text": ("A caffè latte is made with one shot of espresso and around 200 "
              "millilitres of steamed milk topped with a thin layer of microfoam. "
              "The ratio is roughly one part espresso to five parts milk. A "
              "cappuccino uses the same espresso base but has equal parts milk and "
              "foam, giving it a lighter, airier texture.")},
]

# Pricing constants (from W4 multi-model week)
PRICE_INPUT_PER_1M  = {"gpt-4o-mini": 0.15}
PRICE_OUTPUT_PER_1M = {"gpt-4o-mini": 0.60}
PRICE_EMBED_PER_1M  = {"text-embedding-3-small": 0.02,
                       "text-embedding-3-large": 0.13}


def cost_usd(result: dict, chat_model: str = CHAT_MODEL) -> float:
    """Compute cost of a single ask_rag call from token counts."""
    return (
        result["tokens_in"]  * PRICE_INPUT_PER_1M[chat_model]  / 1_000_000 +
        result["tokens_out"] * PRICE_OUTPUT_PER_1M[chat_model] / 1_000_000
    )
