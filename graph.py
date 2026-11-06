from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

import tools as t
from agent import agent

REWRITER = ChatOllama(model="qwen3:8b",temperature=0,reasoning=False)

REWRITE_PROMPT = """
You turn a player's question about Heroes of Might and Magic: Olden Era into
search terms in the game's official wording. Output ONLY the search terms, on one line.
Do not answer the question. Keep unit, spell and artifact names unchanged.

Glossary (player word -> game term):
- buff -> positive effect
- debuff -> negative effect
- raise dead, resurrect -> revive fallen units
- retaliation, counter -> counterattack
- shooter -> ranged attack
- flyer -> flying
- AoE, splash -> all adjacent creatures

Examples:
Question: Which units can cleanse debuffs from allies?
Search terms: remove negative effects from friendly creatures
Question: Is there a shooter that ignores the range penalty?
Search terms: ranged attack full damage at any distance

Question: {question}
Search terms:
"""


class State(TypedDict, total=False):
    question: str
    search_terms: str
    answer: str
    messages: list


def rewrite(state: State) -> dict:
    reply = REWRITER.invoke(input=REWRITE_PROMPT.format(question=state["question"]))
    return {"search_terms": reply}


def answer(state: State) -> dict:
    content = f"{state['question']}\n\nGame terms to use when searching: {state['search_terms']}"
    result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    return {"answer": result["messages"][-1].content, "messages": result["messages"]}


workflow = StateGraph(State)
workflow.add_node("rewrite", rewrite)
workflow.add_node("answer", answer)

workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite","answer")
workflow.add_edge("answer",END)

app = workflow.compile()


def ask(question: str) -> None:
    out = app.invoke({"question": question})
    print(f"\n=== {question}\n  terms: {out['search_terms']}")
    for m in out["messages"]:
        if m.type == "ai" and m.tool_calls:
            for call in m.tool_calls:
                print(f"  -> {call['name']}({call['args']})")
        elif m.type == "tool":
            print(f"  <- {m.content[:200]}")
    print("ANSWER:", out["answer"])


if __name__ == "__main__":
    t.get_model()          # warm up once, before any parallel tool calls
    t.get_collection()
    for q in [
        "What are the 5 Temple units with the most HP?",
        "Which units can't be counterattacked?",
        "What does Healing Water do at level 3?",
        "Which units can raise dead units?",
        "Can any creature steal buffs from enemies?",
        "Which faction is the best?",
    ]:        # the same six questions
        ask(q)