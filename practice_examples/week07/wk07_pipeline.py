"""wk07_pipeline.py — shared corpus + helpers for W7 notebooks.

Day 1 builds these functions cell-by-cell. Day 2 imports them from here so
we don't spend 20 minutes rebuilding.

Pure functions. Same OpenAI-only stack as W6 (`text-embedding-3-small` and
`text-embedding-3-large`).
"""
from __future__ import annotations

import os
import time
import numpy as np
from openai import OpenAI

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before importing"

_client = OpenAI()

# Two embedding models we'll compare
EMBED_SMALL = "text-embedding-3-small"   # 1536 dims, $0.02 per 1M tokens
EMBED_LARGE = "text-embedding-3-large"   # 3072 dims, $0.13 per 1M tokens


# ─── Embedding ───────────────────────────────────────────────────────

def embed_batch(texts: list[str], model: str = EMBED_SMALL) -> list[list[float]]:
    """One API call, list of vectors back."""
    resp = _client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def embed_one(text: str, model: str = EMBED_SMALL) -> list[float]:
    """Convenience — embed a single string."""
    return embed_batch([text], model=model)[0]


# ─── Similarity metrics (all three) ──────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity — measures angle. Higher = more similar. Range [-1, 1]."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def dot(a: list[float], b: list[float]) -> float:
    """Dot product — measures angle + magnitude. Higher = more similar."""
    return float(np.dot(np.array(a), np.array(b)))


def l2(a: list[float], b: list[float]) -> float:
    """L2 (Euclidean) distance — measures straight-line distance. LOWER = more similar."""
    va, vb = np.array(a), np.array(b)
    return float(np.linalg.norm(va - vb))


# ─── Qdrant client resolver (matches capstone convention) ─────────────

def get_qdrant_client():
    """Build a QdrantClient using QDRANT_URL + QDRANT_API_KEY env vars.

    Resolution order (same as capstone src/rag/qdrant_store.py):
      1. QDRANT_URL + QDRANT_API_KEY → Qdrant Cloud
      2. QDRANT_URL only → local (no auth)
      3. Default → http://localhost:6333 (local Docker fallback)
    """
    from qdrant_client import QdrantClient
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


# ─── The corpus: 10 animals across 5 categories ──────────────────────
# Chosen so semantic clustering across categories is interesting to explore.

CORPUS_ANIMALS = [
    # Mammals
    {"id": "cat",     "category": "mammal",  "text": (
        "The domestic cat is a small carnivorous mammal kept as a pet by "
        "humans for thousands of years. Cats have keen senses, retractable "
        "claws, and are known for their agility, night vision, and independent "
        "behaviour. They communicate through vocalisations and body language."
    )},
    {"id": "dog",     "category": "mammal",  "text": (
        "The dog is a domesticated carnivorous mammal descended from wolves. "
        "Dogs are highly social animals kept by humans for companionship, "
        "work, and protection. They are known for loyalty, trainability, and "
        "an acute sense of smell that far exceeds human capability."
    )},
    # Birds
    {"id": "eagle",   "category": "bird",    "text": (
        "Eagles are large birds of prey with keen eyesight, powerful hooked "
        "beaks, and strong talons. They soar at high altitudes and hunt "
        "smaller animals including fish, rabbits, and reptiles. Bald eagles "
        "are the national bird of the United States."
    )},
    {"id": "sparrow", "category": "bird",    "text": (
        "Sparrows are small, plump songbirds found across most of the world. "
        "They typically have short beaks adapted for eating seeds and grain, "
        "though they also feed on insects. Their brown and grey plumage "
        "provides camouflage in urban and rural environments alike."
    )},
    # Fish
    {"id": "salmon",  "category": "fish",    "text": (
        "Salmon are ray-finned fish found in the Northern Pacific and "
        "Atlantic oceans. They are anadromous — born in freshwater, migrating "
        "to the sea to mature, then returning upstream to spawn. Salmon are "
        "prized for their pink flesh, high in omega-3 fatty acids."
    )},
    {"id": "shark",   "category": "fish",    "text": (
        "Sharks are cartilaginous fish that have inhabited the oceans for "
        "over 400 million years. They have a keen sense of smell, multiple "
        "rows of teeth that regenerate throughout life, and a hydrodynamic "
        "shape. Most sharks are carnivorous, though the largest species eat plankton."
    )},
    # Reptiles
    {"id": "python",  "category": "reptile", "text": (
        "Pythons are non-venomous constrictor snakes native to Africa, Asia, "
        "and Australia. They kill prey by wrapping their muscular bodies "
        "around it and squeezing until circulation stops. Some species can "
        "grow over six metres long and live for more than 30 years."
    )},
    {"id": "gecko",   "category": "reptile", "text": (
        "Geckos are small lizards found in warm climates worldwide. They are "
        "known for their remarkable ability to climb smooth vertical surfaces "
        "using microscopic hairs on their toe pads. Many gecko species make "
        "chirping vocalisations, unusual among reptiles."
    )},
    # Insects
    {"id": "bee",     "category": "insect",  "text": (
        "Honey bees are flying insects known for producing honey and "
        "beeswax. They live in colonies with a strict social structure — one "
        "queen, thousands of workers, and drones. Bees pollinate a large "
        "fraction of the world's flowering plants, including many food crops."
    )},
    {"id": "ant",     "category": "insect",  "text": (
        "Ants are eusocial insects that live in large organised colonies. "
        "They communicate primarily through pheromones and are found on "
        "every continent except Antarctica. Some species farm fungi, herd "
        "aphids for their honeydew, or maintain slave colonies of other ant species."
    )},
]


# ─── Test questions used across both notebooks ────────────────────────

TEST_QUESTIONS = [
    {"q": "Which animals live in the ocean?",              "expected_categories": ["fish"]},
    {"q": "What insects live in social colonies?",         "expected_categories": ["insect"]},
    {"q": "What animals have keen eyesight for hunting?",  "expected_categories": ["bird"]},
    {"q": "Which animals are kept as pets?",               "expected_categories": ["mammal", "reptile"]},
    {"q": "How do reptiles kill their prey?",              "expected_categories": ["reptile"]},
]


# ─── Pricing constants (for cost display) ─────────────────────────────

PRICE_EMBED_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}
