from sentence_transformers import CrossEncoder
import tools as t

INSTRUCTION = ("Given a player's question about the game Heroes of Might and Magic: Olden Era, "
               "retrieve the ability, spell or artifact whose effect answers the question")
reranker = CrossEncoder("Qwen/Qwen3-Reranker-0.6B",
                        prompts={"game": INSTRUCTION}, default_prompt_name="game")

q = "Can any creature steal buffs from enemies?"
emb = t.get_model().encode(q, prompt_name="query", normalize_embeddings=True)
res = t.get_collection().query(query_embeddings=emb, n_results=100)
ids, docs = res["ids"][0], res["documents"][0]

scores = reranker.predict([(q, doc) for doc in docs])
order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
reranked = [ids[i] for i in order]

print("Devouring Chalice: embedding rank", ids.index("angel_upg_2") + 1,
      "-> reranked", reranked.index("angel_upg_2") + 1)
for rank, i in enumerate(order[:8], start=1):
    print(f"{rank:3}  {float(scores[i]):7.2f}  {ids[i]}")