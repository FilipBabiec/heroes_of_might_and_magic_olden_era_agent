import tools as t

q = "steal positive effects from enemy creatures"
emb = t.get_model().encode(q, prompt_name="query", normalize_embeddings=True)
res = t.get_collection().query(query_embeddings=emb, n_results=50)
ids = res["ids"][0]

print("Devouring Chalice rank:", ids.index("angel_upg_2") + 1 if "angel_upg_2" in ids else "not in top 50")
for rank, (doc_id, dist) in enumerate(zip(ids[:12], res["distances"][0][:12]), start=1):
    print(f"{rank:3}  {1 - dist / 2:.3f}  {doc_id}")