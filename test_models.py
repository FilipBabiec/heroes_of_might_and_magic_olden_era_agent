import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
import hashlib
from collections import defaultdict


CANDIDATES = {
    "MiniLM (baseline)": ("sentence-transformers/all-MiniLM-L6-v2", {}),
    "bge-m3":            ("BAAI/bge-m3", {}),
    "Qwen3-0.6B":        ("Qwen/Qwen3-Embedding-0.6B", {"prompt_name": "query"}),
}
TEST_QUERIES = [
    # (question, acceptable ids, category)
    ("creature that takes no damage from spells",
     {"base_passive_absolute_resistance", "base_passive_immunity_magic_3",
      "base_passive_immunity_magic_4", "base_passive_immunity_magic_5"}, "meaning"),
    ("attacks two times per turn",
     {"base_passive_double_strike", "base_passive_double_strike_ranged_attack"}, "meaning"),
    ("steal buffs from enemy creatures",
     {"angel_upg_2"},   # + whatever the 'positive effect' query finds
     "slang"),
    ("spells that heal",
     {'day_13_magic_holy_arms','day_1_magic_healing_water','day_1_magic_healing_water_special'}, "meaning"),
    ("What takes or copies positive effects",
     {'angel_1','angel_upg_alt_1','fairy_dragon_1','phoenix_upg_2','inquisitor_upg_alt_2'}, "meaning"),
    ("What takes or copies buffs",
     {'angel_1','angel_upg_alt_1','fairy_dragon_1','phoenix_upg_2','inquisitor_upg_alt_2'}, "slang"),
    ("What gives spell immunity?",
     {'base_passive_immunity_magic_3', 'base_passive_immunity_magic_5', 'base_passive_immunity_magic_4', 'base_passive_immunity_magic_1', 'base_passive_immunity_magic_2'}, "meaning"),
    ("What gives spell immunity against effects killing additional units?",
     {'mech_guard_passive_1'}, "meaning")
]


def document_embeddings(model, model_id: str, docs: list[dict]) -> np.ndarray:
    texts = [d["text"] for d in docs]
    settings = "float32|normalized"
    fingerprint = hashlib.md5((settings + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:8]
    cache = Path("data/eval_cache") / f"{model_id.replace('/', '__')}_{fingerprint}.npy"
    if cache.exists():
        return np.load(cache)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


def rank_of_expected(scores_row, doc_ids, expected):
    """Position (1 = best) of the first correct document, plus the top 3 ids for inspection."""
    order = np.argsort(-scores_row)
    top3 = [doc_ids[k] for k in order[:3]]
    for position, j in enumerate(order, start=1):
        if doc_ids[j] in expected:
            return position, top3
    return None, top3


def evaluate(model_id: str, query_kwargs: dict, docs: list[dict]) -> None:
    model = SentenceTransformer(model_id, model_kwargs={"dtype": torch.float32})
    doc_ids = [d["id"] for d in docs]
    doc_emb = document_embeddings(model, model_id, docs)
    q_emb = model.encode([q for q, _, _ in TEST_QUERIES], normalize_embeddings=True, **query_kwargs)
    scores = q_emb @ doc_emb.T                      # cosine similarity, because vectors are normalized

    hits, totals = defaultdict(int), defaultdict(int)
    for (query, expected, category), row in zip(TEST_QUERIES, scores):
        position, top3 = rank_of_expected(row, doc_ids, expected)
        hit = position is not None and position <= 3
        hits[category] += hit
        totals[category] += 1
        print(f"  {'OK  ' if hit else 'MISS'} [{category}] rank {position}: {query!r}"
              + ("" if hit else f"  top3={top3}"))
    for category in totals:
        print(f"  hit@3 {category}: {hits[category]}/{totals[category]}")
    print()


docs = json.loads(Path("data/processed/v0.81.02/documents.json").read_text(encoding="utf-8"))
for label, (model_id, query_kwargs) in CANDIDATES.items():
    print(label)
    evaluate(model_id, query_kwargs, docs)

for entry in TEST_QUERIES:
    assert isinstance(entry, tuple) and len(entry) == 3, f"malformed test entry: {entry!r}"