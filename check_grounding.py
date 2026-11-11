"""
Test the grounding checker on its own, on cases where the right verdict is known.

Runs only the check node (no agent), so one run takes seconds instead of minutes.
The evidence is produced by calling the real tools, so it looks exactly like what
the agent sees.

Done when: every case prints PASS.
"""

import json

from langchain_core.messages import ToolMessage

import tools as t
from graph import check


def as_json(data) -> str:
    # Must match how the @tool wrappers in agent.py serialise results.
    # If yours uses indent or other options, copy them here.
    return json.dumps(data, ensure_ascii=False)


CHALICE_EVIDENCE = as_json(t.search_rules("steal positive effects from enemy creatures"))
TEMPLE_HP_EVIDENCE = as_json(t.query_units(faction="temple", sort_by="hp", sort_desc=True, limit=5))
FACTION_EVIDENCE = as_json(t.search_rules("best faction"))

# (name, evidence, answer, expected overall verdict, what to look for in the claims)
CASES = [
    (
        "false claims",
        CHALICE_EVIDENCE,
        "No creature in Olden Era can steal buffs. Archangels have 300 HP.",
        False,
        "both sentences BAD: the first contradicts the evidence, the second is not in it",
    ),
    (
        "correct Chalice",
        CHALICE_EVIDENCE,
        "The Archangel's Devouring Chalice dispels all positive effects from enemies within "
        "a 1-hex radius of the chosen hex and applies them to the Archangel.",
        True,
        "all ok",
    ),
    (
        "Q1 Temple HP (earlier false alarm)",
        TEMPLE_HP_EVIDENCE,
        "1. **Apotheosis** - 300 HP\n"
        "2. **Angel** - 225 HP\n"
        "3. **Archangel** - 225 HP\n"
        "4. **Mother Superior** - 125 HP\n"
        "5. **Excommunicator** - 90 HP",
        True,
        "all ok, one claim per unit",
    ),
    (
        "Q6 best faction (earlier missed judgment)",
        FACTION_EVIDENCE,
        "The game data does not explicitly rank factions as \"best.\" However, the search results "
        "highlight **Hive** faction units (e.g., *Hive Mother*, *Hive Queen*) having powerful passive "
        "abilities like **Soul Assimilator**, which grant significant bonuses (e.g., +25% Attack vs. "
        "Temple, +25% Defense vs. Necropolis). The **Temple** faction has *Inspiring* abilities that "
        "boost Morale, but these are less impactful than Hive's direct stat enhancements.",
        False,
        "'less impactful than Hive's' BAD; check the Hive Mother / Hive Queen claims against the evidence",
    ),
]


def main() -> None:
    results = []
    for name, evidence, answer, expected, look_for in CASES:
        print(f"\n=== {name}  (expected supported={expected}; {look_for})")
        state = {"messages": [ToolMessage(content=evidence, tool_call_id="test")], "answer": answer}
        verdict = check(state)["verdict"]
        for c in verdict.claims:
            print(f"    {'ok ' if c.supported else 'BAD'} {c.claim}")
        passed = verdict.supported == expected
        results.append(passed)
        print(f"  -> supported={verdict.supported}  {'PASS' if passed else 'FAIL'}")

    print(f"\n{sum(results)}/{len(results)} cases passed")


if __name__ == "__main__":
    main()