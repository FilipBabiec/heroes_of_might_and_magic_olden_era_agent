"""Time the reranker on 30 real pairs, on the GPU and on the CPU."""
import time

import torch
from sentence_transformers import CrossEncoder

import tools as t

q = "creature that takes no damage from spells"
emb = t.get_model().encode(q, prompt_name="query", normalize_embeddings=True)
docs = t.get_collection().query(query_embeddings=emb, n_results=30)["documents"][0]
pairs = [(q, d) for d in docs]

print("embedding model is on:", t.get_model().device)
devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
for device in devices:
    rr = CrossEncoder(t.RERANKER_ID, model_kwargs={"dtype": torch.float32}, device=device,
                      prompts={"game": t.RERANK_INSTRUCTION}, default_prompt_name="game")
    rr.predict(pairs[:2])  # warm-up, so one-time setup isn't measured
    start = time.perf_counter()
    rr.predict(pairs)
    print(f"{device}: {time.perf_counter() - start:.1f} s for 30 pairs")
    del rr
    if device == "cuda":
        torch.cuda.empty_cache()