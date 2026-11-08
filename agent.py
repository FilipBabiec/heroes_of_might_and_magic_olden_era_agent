import tools as t
import json
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from tools import get_model, get_collection, get_reranker


MODEL = "qwen3:8b"
SYSTEM_PROMPT = """You are an assistant for the game Heroes of Might and Magic: Olden Era.
Rules:
- Base every factual statement on tool results. Do not use your own knowledge of earlier
  Heroes games: Olden Era's units, numbers and rules are different.
- If the tools return nothing relevant, or a description says it is not available, say so plainly.
- If a question is about opinion or strategy that the data cannot answer (e.g. 'best faction'),
  say that the game data doesn't settle it, and offer the relevant facts instead.
- Answer concisely, naming the units, abilities or spells you found.
- If the search results don't contain what was asked, say you couldn't find it in the data.
  Never conclude that something does not exist in the game just because a search didn't find it.
- Do not state any facts about the game that did not come from a tool result in this conversation.
"""


def as_json(result) -> str:
    return json.dumps(result, ensure_ascii=False)


@tool
def query_units(faction: t.Faction | None = None, tier_min: int | None = None, tier_max: int | None = None,
                sort_by: t.SortField | None = None, sort_desc: bool = True, limit: int = 10) -> str:
    '''
    List or rank units by exact criteria: faction, tier range (1-7), and a stat to sort by.
    Factions: temple, necropolis, grove, hive, schism, dungeon, neutral.
    Use this for questions like 'strongest', 'all units of', 'which tier 7 units'.
    Do NOT use it for abilities or rules: use search_rules for those.
    '''

    try:
        return as_json(t.query_units(faction, tier_min, tier_max, sort_by, sort_desc, limit))
    except Exception as e:
        return f"Error: {e}"


@tool
def search_rules(query: str, n: int = 5) -> list[dict]:
    '''
    Find abilities, spells and artifacts that match the search query criteria.
    Use the game's official terms (e.g. buffs -> positive effects, raise dead -> revive fallen units).
    For shared abilities the results include units that have them.
    '''
    try:
        return as_json(t.search_rules(query, n))
    except Exception as e:
        return f"Error: {e}"


agent = create_agent(
    model = ChatOllama(model=MODEL,temperature=0),
    tools = [search_rules, query_units],
    system_prompt=SYSTEM_PROMPT,
)


def ask(question: str) -> None:
    print(f"\n=== {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for m in result["messages"]:
        if m.type == "ai" and m.tool_calls:
            for call in m.tool_calls:
                print(f"  -> {call['name']}({call['args']})")
        elif m.type == "tool":
            print(f"  <- {m.content[:200]}")
    print("ANSWER:", result["messages"][-1].content)


if __name__ == "__main__":
    get_model()
    get_collection()
    get_reranker()
    for q in [
        "What are the 5 Temple units with the most HP?",
        "Which units can't be counterattacked?",
        "What does Healing Water do at level 3?",
        "Which units can raise dead units?",
        "Can any creature steal buffs from enemies?",
        "Which faction is the best?",
    ]:
        ask(q)