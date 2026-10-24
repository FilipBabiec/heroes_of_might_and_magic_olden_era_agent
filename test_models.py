import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


CANDIDATES = {
    "MiniLM (baseline)": ("sentence-transformers/all-MiniLM-L6-v2", {}),
    "bge-m3":            ("BAAI/bge-m3", {}),
    "Qwen3-0.6B":        ("Qwen/Qwen3-Embedding-0.6B", {"prompt_name": "query"}),
}
TEST_QUERIES = [
    ("which units can't be counterattacked",             {"base_passive_noncounter"}),
    ("spell to restore health of my creature",           {"day_1_magic_healing_water"}),
    ("item that helps with diplomacy",                   {"ambassadors_word_ambassadors_sash_artifact"}),
    ("steal buffs from enemy creatures",                 {"angel_upg_2"}),
    ("attacks two times per turn",                       {"base_passive_double_strike"}),
    ("creature that takes no damage from spells",        {"base_passive_absolute_resistance"}),
    ("shooter without distance penalty",                 {"base_passive_sharpshooter"}),
]


def rank_of_expected(scores_row, doc_ids, expected):
    """Position (1 = best) of the first correct document, plus the top 3 ids for inspection."""
    order = np.argsort(-scores_row)
    top3 = [doc_ids[k] for k in order[:3]]
    for position, j in enumerate(order, start=1):
        if doc_ids[j] in expected:
            return position, top3
    return None, top3


def evaluate(model_id: str, query_kwargs: dict, docs: list[dict]) -> None:
    model = SentenceTransformer(model_id)
    doc_ids = [d["id"] for d in docs]
    doc_emb = model.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=True)
    q_emb = model.encode([q for q, _ in TEST_QUERIES], normalize_embeddings=True, **query_kwargs)
    scores = q_emb @ doc_emb.T                      # cosine similarity, because vectors are normalized

    hits = 0
    for (query, expected), row in zip(TEST_QUERIES, scores):
        position, top3 = rank_of_expected(row, doc_ids, expected)
        hit = position is not None and position <= 3
        hits += hit
        print(f"  {'OK  ' if hit else 'MISS'} rank {position}: {query!r}" + ("" if hit else f"  top3={top3}"))
    print(f"  hit@3: {hits}/{len(TEST_QUERIES)}\n")


docs = json.loads(Path("data/processed/v0.81.02/documents.json").read_text(encoding="utf-8"))
for label, (model_id, query_kwargs) in CANDIDATES.items():
    print(label)
    evaluate(model_id, query_kwargs, docs)