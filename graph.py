from typing import TypedDict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


import tools as t
from agent import agent


REWRITE_PROMPT = """You turn a player's question about Heroes of Might and Magic: Olden Era into
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
Search terms:"""


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
- rules: the question asks for facts from the game data: unit stats, abilities, spells, artifacts
  or factions. This includes lists and rankings by a stat, even with words like "most", "highest" or "best".
  Examples: "Which unit has the most HP?", "What is the best Offence among Temple units?", "What does Healing Water do?"
- advice: the question asks what to choose, build or focus on, how to play a strategy or playstyle,
  or for the assistant's opinion.
  Examples: "Which faction is the best?", "What should I build for an aggressive army?", "Do you agree that Archangel is better?"
- other: the question is not about Olden Era.
  Example: "What's the weather in Warsaw?"

<question>
{question}
</question>"""


ADVICE_PROMPT = """You are an assistant for the Heroes of Might and Magic Olden Era game. You provide advice and opinion on the game system and rules.
You have access to game information searched in the official game database below. Clearly state if you give an opinion or a fact. 
Use only the facts below, no knowledge from earlier Heroes games.
The question you have been asked to give an opinion on:
{prompt}

Additional information form the game's database:
{additional_info}"""


PLAN_PROMPT = """You are an assistant for the Heroes of Might and Magic Olden Era game. You have been tasked with establishing what facts do 
you need from the game database to give a meaningful advice and opinion:

{prompt}

Your job is to plan what game information you will need to give the advise and write a question to the search engine tool.
Your two tools that you have available are: 
 - 'search_rules' that searches the game system rules - spell descriptions, abilities descriptions, units descriptions and artifacts descriptions. 
 - 'query_units' tool that returns specific information about the units. This tool searches the SQL database with regular queries.
Write natural language prompt that covers everything you will need."""


CLASSIFICATION = Literal["rules","advice","other"]

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
    prompt: str
    question: str
    classification: CLASSIFICATION
    search_terms: str
    answer: str
    messages: list
    verdict: GroundingVerdict
    additional_info: str


REWRITER = ChatOllama(model="qwen3:8b",temperature=0,reasoning=False)
CHECKER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192).with_structured_output(GroundingVerdict)
WRITER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192)
CLASSIFIER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192).with_structured_output(ClassifiedQuestion)
PLANNER = ChatOllama(model="qwen3:8b", temperature=0, reasoning=False, num_ctx=8192)
ADVISOR = ChatOllama(model="qwen3:14b", temperature=0, reasoning=False, num_ctx=8192)


def normalize(text: str) -> str:
    """Lowercase, drop double quotes, collapse whitespace: 'hp: 300' then matches '"hp": 300'."""
    return " ".join(text.lower().replace('"', "").split())


def classify(state:State) -> dict[str, Any]:
    reply = CLASSIFIER.invoke(input=CLASSIFY_PROMPT.format(question=state["prompt"]))
    if reply.category == "rules":
        return {"classification": reply.category,
                "question": state["prompt"]}
    return {"classification": reply.category}


def route_after_classify(state:State) -> str:
    return state["classification"]


def rewrite(state: State) -> dict:
    reply = REWRITER.invoke(input=REWRITE_PROMPT.format(question=state["question"]))
    return {"search_terms": reply.content.strip()}


def answer(state: State) -> dict:
    content = f"{state['question']}\n\nGame terms to use when searching: {state['search_terms']}"
    result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    if state["classification"] == "rules":
        return {"answer": result["messages"][-1].content, "messages": result["messages"]}
    elif state["classification"] == "advice":
        return {"additional_info": result["messages"][-1].content}


def advice(state: State):
    # content = f"{state['prompt']}\n\nAdditional information from searching the game database: {state['additional_info']}"
    # result = advice_agent.invoke({"messages": [{"role": "user", "content": content}]})
    # return {"answer": result["messages"][-1].content, "messages": result["messages"]}
    reply = ADVISOR.invoke(input=ADVICE_PROMPT.format(prompt=state["prompt"], additional_info=state["additional_info"]))
    return {"answer": reply.content.strip()}


def plan(state: State) -> dict:
    reply = PLANNER.invoke(input=PLAN_PROMPT.format(prompt=state["prompt"]))
    return {"question": reply.content.strip()}


def route_after_revise(state: State) -> str:
    return "advice" if state["classification"] == "advice" else END


def collect_evidence(messages: list) -> str:
    """Everything the tools returned during the agent run: the only allowed source of facts."""
    results = [m.content for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(results) if results else "(no tool results)"


def check(state: State) -> dict:
    evidence = collect_evidence(state["messages"])
    verdict = CHECKER.invoke(CHECK_PROMPT.format(evidence=evidence, answer=state["answer"] if state["classification"] == "rules" else state["additional_info"]))
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
        answer=state["answer"] if state["classification"] == "rules" else state["additional_info"],
        claims="; ".join(state["verdict"].unsupported_claims),
    ))
    return {"answer": reply.content}


def route_after_check(state: State) -> str:
    if not state["verdict"].supported:
        return "revise"
    return "advice" if state["classification"] == "advice" else END


def other(state:State) -> dict:
    return{"answer": "This is an unrelated question that I cannot answer. I am a chatbot used for topics about Heroes of Might and Magic Olden Era. Please ask me related question!"}


workflow = StateGraph(State)
workflow.add_node("classify",classify)
workflow.add_node("rewrite", rewrite)
workflow.add_node("check", check)
workflow.add_node("revise", revise)
workflow.add_node("answer", answer)
workflow.add_node("advice", advice)
workflow.add_node("plan", plan)
workflow.add_node("other", other)


workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "rules": "rewrite",
        "advice": "plan",
        "other": "other"
    }
)
workflow.add_edge("plan","rewrite")
workflow.add_edge("rewrite","answer")
workflow.add_edge("answer","check")
workflow.add_conditional_edges(
    "check",
    route_after_check,
    {
        END: END,
        "revise": "revise",
        "advice": "advice"
    }
)
workflow.add_conditional_edges(
    "revise",
    route_after_revise,
    {
        END: END,
        "advice": "advice"
    }
)
workflow.add_edge("advice",END)
workflow.add_edge("other",END)

app = workflow.compile()


def ask(prompt: str) -> None:
    out = app.invoke({"prompt": prompt})
    print(f"\n=== {prompt}\n  terms: {out['search_terms']}")
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

