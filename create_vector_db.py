'''
Input: data/processed/v<patch>/*.json
Output: data/hoe.vectordb, deleted and rebuilt on every run
Done when:
'''

import json
from collections import Counter
from pathlib import Path
import chromadb
import numpy as np
import shutil
import torch
import hashlib
from sentence_transformers import SentenceTransformer, util
from raw_data_processing import PROCESSED_DIR, load_json
from download_raw_data import version_key, save_json
from create_sql_db import load_data


FACTION_MAPPINGS = {
    "human": "Temple",
    "nature": "Grove",
    "undead": "Necropolis",
    "demon": "Hive",
    "unfrozen": "Schism",
    "dungeon": "Dungeon",
    "neutral": "Neutral"
}
CHROMA_DIR = Path("data/chroma")
COLLECTION = "hoe_rules"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
CURRENT_PATCH = "v0.81.02"


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


def build_collection(docs: list[dict], patch:str):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    docs = json.loads(Path("data/processed/v0.81.02/documents.json").read_text(encoding="utf-8"))
    embeddings = document_embeddings(model, MODEL_ID, docs)

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"embedding_model": MODEL_ID, "patch": patch},
    )
    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings.tolist(),
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return collection


def clean_metadata(meta: dict) -> dict:
    """Chroma accepts only str/int/float/bool, so drop keys whose value is None."""
    return {k: v for k, v in meta.items() if v is not None}


def build_documents(data: dict):
    documents = []
    for table, rows in data.items():
        print(table)
        for row in rows:
                if table == "abilities":
                    if row["description"] is not None:
                        if row["shared"]:
                            owner = f"Shared {row['kind']} ability, used by several units."
                        else:
                            faction_name = FACTION_MAPPINGS.get(row["faction"], row["faction"])
                            owner = f"{row['kind'].capitalize()} ability of {row['unit_name']} ({faction_name}, tier {row['tier']})."
                        document = {
                            "id": row["ability_id"],
                            "text": f"{row['name']}. {owner} {row['description']}",
                            "metadata": clean_metadata
                                ({
                                    "unit": row["unit_name"],
                                    "tier": row["tier"],
                                    "faction": row["faction"],
                                    "shared": row["shared"],
                                    "kind": row["kind"],
                                    "record_id": row["ability_id"],
                                    "name": row["name"],
                                    "doc_type": "ability"
                                })
                        }
                        documents.append(document)
                if table == "spells":
                    if row["levels"]:
                        spell_levels = ""
                        for level in row["levels"]:
                            spell_levels += f"Level {level["level"]}: {level["description"]}"
                            if level["bonus"]:
                                spell_levels += f" (Change: {level["bonus"]})"
                            spell_levels += "\n"
                        document = {
                            "id": row["id"],
                            "text": f"{row["name"]}, {row["school"]} magic. Levels: {spell_levels}",
                            "metadata": clean_metadata
                                ({
                                    "school": row["school"],
                                    "rank": row["rank"],
                                    "record_id": row["id"],
                                    "name": row["name"],
                                    "doc_type": "spell"
                                })
                        }
                        documents.append(document)
                if table == "artifacts":
                    if row["description"] is not None:
                        upgrade = f" Upgrade: {row['upgrade']}" if row["upgrade"] else ""
                        document = {
                            "id": row["id"],
                            "text": f"{row['name']}. {row['description']}{upgrade}",
                            "metadata": clean_metadata
                                ({
                                    "rarity": row["rarity"],
                                    "slot": row["slot"],
                                    "record_id": row["id"],
                                    "name": row["name"],
                                    "doc_type": "artifact"
                                })
                        }
                        documents.append(document)
    return documents


def search_db(collection, question: str, n:int = 5, where:dict | None = None) -> list[dict]:
    assert collection.metadata["embedding_model"] == MODEL_ID, "collection was built with another model"
    model = SentenceTransformer(MODEL_ID, local_files_only=True)
    q = model.encode([question], prompt_name="query", normalize_embeddings=True)
    res = collection.query(query_embeddings=q.tolist(), n_results=n, where=where)

    return [{"id": i, "distance": d, "metadata": m, "text": t}
            for i, d, m, t in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0])]


def main() -> None:
    data = load_data()
    docs = build_documents(data)
    save_json(docs, PROCESSED_DIR / CURRENT_PATCH / "documents.json")
    collection = build_collection(docs,CURRENT_PATCH)
    assert collection.count() == len(docs) == 790
    assert all(r["metadata"]["doc_type"] == "spell"
               for r in search_db(collection, "heal", where={"doc_type": "spell"}))
    print(search_db(collection, "heal my creature", where={"doc_type": "spell"}))
    print(search_db(collection, "strong attack", where={"$and": [{"doc_type": "ability"}, {"faction": "human"}]}))

if __name__ == "__main__":
    main()
