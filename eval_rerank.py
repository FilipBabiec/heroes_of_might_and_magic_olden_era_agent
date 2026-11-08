"""Compare retrieval with and without reranking: rank per question, hit@3 per category, time per query."""
import time
from collections import defaultdict

import tools as t
from test_models import TEST_QUERIES

K = 3


def rank_in(ids: list[str], expected: set[str]) -> int | None:
    """1-based rank of the first expected id in ids, or None if none of them is there."""
    for rank, doc_id in enumerate(ids, start=1):
        if doc_id in expected:
            return rank
    return None


def embedding_ids(query: str, n: int) -> list[str]:
    emb = t.get_model().encode(query, prompt_name="query", normalize_embeddings=True)
    return t.get_collection().query(query_embeddings=emb, n_results=n)["ids"][0]


def main() -> None:
    t.retrieve("warm-up", 1)  # load both models before anything is timed

    hits = {"embedding": defaultdict(list), "reranked": defaultdict(list)}
    times = {"embedding": [], "reranked": []}

    print(f"{'emb':>4} {'rr':>4}  question")
    for question, expected, category in TEST_QUERIES:
        start = time.perf_counter()
        emb_rank = rank_in(embedding_ids(question, t.CANDIDATES), expected)  # rank within the 30 candidates
        times["embedding"].append(time.perf_counter() - start)

        start = time.perf_counter()
        rr_rank = rank_in([h["id"] for h in t.retrieve(question, 10)], expected)
        times["reranked"].append(time.perf_counter() - start)

        emb_hit = emb_rank is not None and emb_rank <= K
        rr_hit = rr_rank is not None and rr_rank <= K
        for name, hit in (("embedding", emb_hit), ("reranked", rr_hit)):
            hits[name][category].append(hit)
            hits[name]["ALL"].append(hit)

        note = "  WORSE" if emb_hit and not rr_hit else "  better" if rr_hit and not emb_hit else ""
        print(f"{emb_rank or '-':>4} {rr_rank or '-':>4}  {question}{note}")

    print(f"\nhit@{K:<9} embedding  reranked")
    for category in sorted(hits["embedding"], key=lambda c: c == "ALL"):  # ALL last
        e, r = hits["embedding"][category], hits["reranked"][category]
        print(f"{category:<12} {sum(e):>3}/{len(e):<7} {sum(r):>3}/{len(r)}")
    for name, ts in times.items():
        print(f"{name:<12} {1000 * sum(ts) / len(ts):6.0f} ms per query")


if __name__ == "__main__":
    main()