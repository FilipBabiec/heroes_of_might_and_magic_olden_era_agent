from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

import tools as t
from agent import agent


class GroundingVerdict(BaseModel):
    """Whether every game fact in the answer is supported by the tool results."""
    supported: bool = Field(description="True only if every game fact in the answer is stated in the tool results")
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Claims from the answer that the tool results do not support")


REWRITER = ChatOllama(model="qwen3:8b",temperature=0,reasoning=False)
CHECKER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192).with_structured_output(GroundingVerdict)
WRITER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192)


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


CHECK_PROMPT = """You check an answer about the game Heroes of Might and Magic: Olden Era against the tool results it was based on.

Tool results:
{evidence}

Answer:
{answer}

A claim is supported only if the tool results state it. Unit stats, faction membership and the effects of abilities, spells and artifacts are all claims.
A claim that something does not exist ("no creature can...", "there is no...") is unsupported: a search that found nothing only shows that nothing was found.
An answer that makes no game claims (for example, declining an opinion question) is supported.
Judge only whether the claims are supported, not whether the answer is complete or helpful."""

REVISE_PROMPT = """Rewrite the answer so that it states only what the tool results support.

Question: {question}

Tool results:
{evidence}

Answer:
{answer}

Unsupported claims: {claims}

Rules:
- Remove each unsupported claim, or correct it if the tool results contain the right fact.
- Replace claims that something does not exist with "I couldn't find ... in the game data".
- Keep everything else as it is. Do not add facts that are not in the tool results."""


class State(TypedDict, total=False):
    question: str
    search_terms: str
    answer: str
    messages: list
    verdict: list


def rewrite(state: State) -> dict:
    reply = REWRITER.invoke(input=REWRITE_PROMPT.format(question=state["question"]))
    return {"search_terms": reply.content.strip()}


def answer(state: State) -> dict:
    content = f"{state['question']}\n\nGame terms to use when searching: {state['search_terms']}"
    result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    return {"answer": result["messages"][-1].content, "messages": result["messages"]}


def collect_evidence(messages: list) -> str:
    """Everything the tools returned during the agent run: the only allowed source of facts."""
    results = [m.content for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(results) if results else "(no tool results)"


def check(state: State) -> dict:
    evidence = collect_evidence(state["messages"])
    verdict = CHECKER.invoke(CHECK_PROMPT.format(evidence=evidence, answer=state["answer"]))
    return {"verdict": verdict}


def revise(state: State) ->dict:
    reply=WRITER.invoke(REVISE_PROMPT.format(
        question = state["question"],
        evidence = collect_evidence(state["messages"]),
        answer=state["answer"],
        claims="; ".join(state["verdict"].unsupported_claims),
    ))
    return {"answer": reply.content}


def route_after_check(state: State) -> str:
    return END if state["verdict"].supported else "revise"


workflow = StateGraph(State)
workflow.add_node("rewrite", rewrite)
workflow.add_node("check", check)
workflow.add_node("revise", revise)
workflow.add_node("answer", answer)


workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite","answer")
workflow.add_edge("answer","check")
workflow.add_conditional_edges(
    "check",
    route_after_check,
    {
        END: END,
        "revise": "revise"
    }
)
workflow.add_edge("revise",END)

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
    verdict = out["verdict"]
    print(f"  verdict: supported={verdict.supported}")
    for claim in verdict.unsupported_claims:
        print(f"    unsupported: {claim}")
    if "draft" in out:
        print("DRAFT:", out["draft"])
    print("ANSWER:", out["answer"])


if __name__ == "__main__":
    t.get_model()          # warm up once, before any parallel tool calls
    t.get_collection()
    t.get_reranker()
    print("warm-up done", flush=True)
    for q in [
        "What are the 5 Temple units with the most HP?",
        "Which units can't be counterattacked?",
        "What does Healing Water do at level 3?",
        "Which units can raise dead units?",
        "Can any creature steal buffs from enemies?",
        "Which faction is the best?",
    ]:
        ask(q)

