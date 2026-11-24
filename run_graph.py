"""
Run questions through the graph one node at a time, showing:
- the path each question takes (which nodes ran, in order),
- how long each node took,
- what each node wrote into the state.

Usage:
    python run_graph.py                      # runs the QUESTIONS below
    python run_graph.py "Your own question"  # runs one question
"""

import sys
import time

import tools as t
from graph import app

# (expected route, question). The route is checked against what classify decided.
QUESTIONS = [
    ("rules", "What are the 5 Temple units with the most HP?"),
    ("rules", "Can any creature steal buffs from enemies?"),
    ("advice", "I want an aggressive Temple army with flyers. Which units should I focus on?"),
    ("advice", "Which faction is the best?"),
    ("other", "What's the weather in Warsaw?"),
]


def short(value, limit: int = 300) -> str:
    """One line, cut to `limit` characters, for printing intermediate values."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + " ..."


def print_update(key: str, value) -> None:
    """Print one key a node wrote, in a readable form for the special ones."""
    if key == "messages":
        for m in value:
            if m.type == "ai" and m.tool_calls:
                for call in m.tool_calls:
                    print(f"    -> {call['name']}({call['args']})")
            elif m.type == "tool":
                print(f"    <- {short(m.content, 150)}")
    elif key == "verdict":
        for c in value.claims:
            print(f"    {'ok ' if c.supported else 'BAD'} {c.claim}")
    else:
        print(f"    {key}: {short(value)}")


def run(expected: str | None, prompt: str) -> None:
    print(f"\n{'=' * 80}\n{prompt}\n  expected route: {expected or '(not given)'}")
    state: dict = {}  # we rebuild the final state ourselves from the node updates
    start = last = time.perf_counter()

    # stream_mode="updates" yields {node_name: what_that_node_returned} after each node finishes
    for step in app.stream({"prompt": prompt}, stream_mode="updates"):
        for node, update in step.items():
            now = time.perf_counter()
            print(f"\n  [{node}]  {now - last:.1f} s")
            last = now
            update = update or {}  # a node that returns nothing yields None
            state.update(update)
            for key, value in update.items():
                print_update(key, value)

    route = state.get("classification")
    route_note = "" if expected is None else ("  OK" if route == expected else "  WRONG ROUTE")
    print(f"\n  route: {route}{route_note}   total: {time.perf_counter() - start:.1f} s")
    print(f"\nANSWER:\n{state.get('answer', '(no answer in the state)')}")


if __name__ == "__main__":
    t.get_model()          # warm up once, before any parallel tool calls
    t.get_collection()
    t.get_reranker()
    print("warm-up done", flush=True)

    questions = [(None, " ".join(sys.argv[1:]))] if len(sys.argv) > 1 else QUESTIONS
    for expected, prompt in questions:
        run(expected, prompt)