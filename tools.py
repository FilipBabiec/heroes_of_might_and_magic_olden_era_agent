import json
import sqlite3
from pathlib import Path


def tool_setup():
    pass


# input: search text + optional filters | output: a list of compact results, best first, shaped by type
def search_rules(query: str, doc_type: str | None = None, faction: str | None = None, n: int = 5):
    pass


# input: structured filters (faction, tier, sort...) | output:
def query_units(faction: str | None, tier_min: int | None, tier_max: int | None, attack_category: str | None, move_type: str | None, max_gold_cost: int | None, ):
    valid_faction = ("temple","necropolis","grove","hive","schism","dungeon", None)
    if faction not in valid_faction:
        raise ValueError("Wrong faction. Allowed values: temple, necropolis, grove, hive, schism, dungeon, neutral, None")


# input: name | output:
def get_details():
    pass

