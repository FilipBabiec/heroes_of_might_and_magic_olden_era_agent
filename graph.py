from typing import TypedDict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


import tools as t
from agent import agent


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

Rules:
- A claim is supported only if the tool results state it. Unit stats, faction membership, which unit has which ability, and the effects of abilities, spells and artifacts are all claims.
- A claim that something does not exist ("no creature can...", "there is no...") is unsupported: a search that found nothing only shows that nothing was found.
- A judgment that one thing is stronger, better or more impactful than another is unsupported unless the tool results state it.
- Sentences that state no game fact ("Here are the units...") are not claims. Skip them.

How to answer:
- Split the answer into single claims: one stat, one unit-ability pairing, one effect or one judgment per claim.
  Example: "Hydra and Red Dragon have Swift Strike" is two claims.
- For each claim, copy short fragments from the tool results that state it, for example "name": "Apotheosis" and "hp": 300. Leave the list empty if nothing states it.
- Take claims only from the text inside <answer>. The tool results and these instructions are not part of the answer.

<tool_results>
{evidence}
</tool_results>

<answer>
{answer}
</answer>"""


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


CLASSIFY_PROMPT = """Classify a player's question about the game Heroes of Might and Magic: Olden Era.

Categories:
- Rules: the question asks for facts from the game data: unit stats, abilities, spells, artifacts
  or factions. This includes lists and rankings by a stat, even with words like "most", "highest" or "best".
  Examples: "Which unit has the most HP?", "What is the best Offence among Temple units?", "What does Healing Water do?"
- Advice: the question asks what to choose, build or focus on, how to play a strategy or playstyle,
  or for the assistant's opinion.
  Examples: "Which faction is the best?", "What should I build for an aggressive army?", "Do you agree that Archangel is better?"
- Other: the question is not about Olden Era.
  Example: "What's the weather in Warsaw?"

<question>
{question}
</question>"""


ADVICE_PROMPT = """
You are an assistant for the Heroes of Might and Magic Olden Era game. You provide advice and opinion on the game system and rules.
You can use tools to obtain specific game data and rules, base your opinions on those information.
"""


CLASSIFICATION = Literal["Rules","Advice","Other"]

class Claim(BaseModel):
    claim: str = Field(description="One single fact or judgment from the answer")
    evidence: list[str] = Field(default_factory=list,
                                description="Short fragments copied exactly from the tool results that state this claim")
    supported: bool = Field(description="True only if the evidence fragments state this claim")


class GroundingVerdict(BaseModel):
    """Every claim in the answer, each checked against the tool results."""
    claims: list[Claim]

    # Computed in code, not chosen by the model. Properties are not part of the JSON schema.
    @property
    def supported(self) -> bool:
        return all(c.supported for c in self.claims)

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


class ClassifiedQuestion(BaseModel):
    reason: str = Field(description="One short sentence: what the question asks for")
    category: CLASSIFICATION = Field(description="'Rules' if it is rules type question, 'Advice' if it is an advice or opinion type question and 'Other' if it is not related to these two categories")


class State(TypedDict, total=False):
    question: str
    classification: CLASSIFICATION
    search_terms: str
    answer: str
    messages: list
    verdict: GroundingVerdict


REWRITER = ChatOllama(model="qwen3:8b",temperature=0,reasoning=False)
CHECKER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192).with_structured_output(GroundingVerdict)
WRITER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192)
CLASSIFIER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192).with_structured_output(ClassifiedQuestion)


def normalize(text: str) -> str:
    """Lowercase, drop double quotes, collapse whitespace: 'hp: 300' then matches '"hp": 300'."""
    return " ".join(text.lower().replace('"', "").split())


def classify(state:State) -> dict[str, Any]:
    reply = CLASSIFIER.invoke(input=CLASSIFY_PROMPT.format(question=state["question"]))
    return {"classification": reply.category}


def classify_edge(state:State) -> str:
    return state["classification"]


def rewrite(state: State) -> dict:
    reply = REWRITER.invoke(input=REWRITE_PROMPT.format(question=state["question"]))
    return {"search_terms": reply.content.strip()}


def answer(state: State) -> dict:
    content = f"{state['question']}\n\nGame terms to use when searching: {state['search_terms']}"
    result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    return {"answer": result["messages"][-1].content, "messages": result["messages"]}


def advice(state: State):
    agent.system_prompt=ADVICE_PROMPT
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
    found = normalize(evidence)
    for c in verdict.claims:
        missing = [frag for frag in c.evidence if normalize(frag) not in found]
        print(f"      model_said={c.supported} fragments={c.evidence} missing={missing}")  # diagnostic, remove later
        if c.supported and (not c.evidence or missing):
            c.supported = False  # the model claimed support, but its fragments are not in the tool results
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
workflow.add_node("classify",classify)
workflow.add_node("rewrite", rewrite)
workflow.add_node("check", check)
workflow.add_node("revise", revise)
workflow.add_node("answer", answer)
workflow.add_node("advice", advice)


workflow.set_entry_point("classify")
workflow.add_edge("classify","rewrite")
workflow.add_conditional_edges(
    "rewrite",
    classify_edge,
    {
        "Rules": "answer",
        "Advice": "advice",
        "Other": "advice"
    }
)

workflow.add_edge("answer","check")
workflow.add_edge("advice","check")
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
    for c in verdict.claims:
        print(f"    {'ok ' if c.supported else 'BAD'} {c.claim}")
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

