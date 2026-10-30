import json
import sqlite3
from pathlib import Path
from typing import Literal, get_args
from collections import defaultdict

from create_sql_db import DB_PATH

Faction = Literal["temple", "necropolis", "grove", "hive", "schism", "dungeon", "neutral"]
AttackCategory = Literal["melee", "ranged", "reach"]
MoveType = Literal["ground", "fly", "teleport", "jump"]
SortField = Literal["hp", "tier", "offence", "defence", "initiative", "speed",
                    "gold_cost", "damage_min", "damage_max"]
FACTION_IDS = {"temple": "human", "necropolis": "undead", "grove": "nature", "hive": "demon",
               "schism": "unfrozen", "dungeon": "dungeon", "neutral": "neutral"}
COLUMNS = ("id, name, faction, tier, hp, offence, defence, damage_min, damage_max, initiative, speed, "
           "luck, morale, num_counters, creature_type, movement_name, attack_type, "
           "gold_cost, wood_cost, ore_cost, mercury_cost, dust_cost, crystal_cost, gemstone_cost")
FACTION_DISPLAY = {unit_id: name.capitalize() for name, unit_id in FACTION_IDS.items()}
DB_PATH = "data/hoe.sqlite"


assert set(FACTION_IDS) == set(get_args(Faction)), "FACTION_IDS and Faction disagree"


def check_choice(name: str, value, allowed: tuple) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"Invalid {name} {value!r}. Allowed: {', '.join(allowed)}.")


def check_tier_values(tier_min: int | None = None, tier_max: int | None = None):
    if tier_min is not None and (tier_min < 1 or tier_min > 7):
        raise ValueError(f"Invalid tier_min value. Allowed: 1-7.")
    if tier_max is not None and (tier_max > 7 or tier_max < 1):
        raise ValueError(f"Invalid tier_max value. Allowed: 1-7.")
    if tier_min is not None and tier_max is not None and tier_min > tier_max:
        raise ValueError(f"Invalid tier range. Tier_min must be lower than tier_max.")

def tool_setup():
    pass


# input: search text + optional filters | output: a list of compact results, best first, shaped by type
def search_rules(query: str, doc_type: str | None = None, faction: str | None = None, n: int = 5):
    pass


# input: structured filters (faction, tier, sort...) | output:
def query_units(
    faction: Faction | None = None,
    tier_min: int | None = None,
    tier_max: int | None = None,
    sort_by: SortField | None = None,
    sort_desc: bool = True,
    limit: int = 10,
    attack_category: AttackCategory | None = None,
    move_type: MoveType | None = None,
    upgraded_only: bool = False,
    max_gold_cost: int | None = None
) -> list[dict]:
    """List or rank units by faction, tier range and a stat. Use get_details for one specific unit."""
    check_choice("faction", faction, get_args(Faction))
    check_choice("sort_by", sort_by, get_args(SortField))
    check_choice("move_type", move_type, get_args(MoveType))
    check_choice("attack_category", attack_category, get_args(AttackCategory))
    check_tier_values(tier_min, tier_max)


    where_sql, where_params = build_where(faction, tier_min, tier_max)
    order_sql, order_params = build_order(sort_by, sort_desc, limit)
    sql = f"SELECT {COLUMNS} FROM units {where_sql} {order_sql}"

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, where_params + order_params).fetchall()
        ability_names = fetch_ability_names(con, [row["id"] for row in rows])
    finally:
        con.close()
    return [to_compact(row, ability_names.get(row["id"], [])) for row in rows]




# input: name | output:
def get_details():
    pass

def build_where(faction: str | None = None, tier_min: int | None = None, tier_max: int | None = None) -> tuple[str, list]:
    conditions = []
    params = []

    if faction is not None:
        conditions.append("faction = ?")
        params.append(FACTION_IDS[faction])

    if tier_min is not None:
        conditions.append("tier >= ?")
        params.append(tier_min)

    if tier_max is not None:
        conditions.append("tier <= ?")
        params.append(tier_max)

    if not conditions:
        return "", []
    return "WHERE " + " AND ".join(conditions), params

def build_order(sort_by: str | None = None, sort_desc: bool = True, limit: int = 10) -> tuple[str, list]:
    check_choice("sort_by", sort_by, get_args(SortField))
    if not (isinstance(limit, int) and 1 <= limit <= 25):
        raise ValueError(f"Invalid limit {limit!r}. Allowed: 1-25.")
    conditions = []
    params = []

    if sort_by is not None:
        conditions.append(f"ORDER BY {sort_by}")
        if sort_desc:
            conditions.append("DESC, name")
        else:
            conditions.append("ASC, name")
    else:
        conditions.append("ORDER BY tier DESC, hp DESC, name")

    if limit is not None:
        conditions.append("LIMIT ?")
        params.append(limit)

    if not conditions:
        return "", []
    return " ".join(conditions), params


def to_compact(row: sqlite3.Row, abilities: list[str]) -> dict:
    data = dict(row)
    cost = {key.removesuffix("_cost"): value for key, value in data.items()
            if key.endswith("_cost") and value is not None}
    return {
        "id": data["id"],
        "name": data["name"],
        "faction": FACTION_DISPLAY[data["faction"]],
        "tier": data["tier"],
        "hp": data["hp"],
        "offence": data["offence"],
        "defence": data["defence"],
        "damage": f"{data['damage_min']}-{data['damage_max']}",
        "initiative": data["initiative"],
        "speed": data["speed"],
        "luck": data["luck"],
        "morale": data["morale"],
        "num_counters": data["num_counters"],
        "creature_type": data["creature_type"],
        "movement": data["movement_name"],
        "attack_type": data["attack_type"],
        "cost": cost,
        "abilities": abilities,
    }


def fetch_ability_names(con: sqlite3.Connection, unit_ids: list[str]) -> dict[str, list[str]]:
    """Own + shared ability names for the given units: {unit_id: [names]}."""
    if not unit_ids:
        return {}
    placeholders = ", ".join("?" * len(unit_ids))
    own = con.execute(f"SELECT unit_id, name FROM abilities WHERE unit_id IN ({placeholders})",
                      unit_ids).fetchall()
    shared = con.execute(f"""SELECT l.unit_id, a.name FROM shared_links l
                             JOIN abilities a ON a.ability_id = l.ability_id
                             WHERE l.unit_id IN ({placeholders})""", unit_ids).fetchall()
    names = defaultdict(list)
    for unit_id, name in own + shared:
        names[unit_id].append(name)
    return names


for unit in query_units(faction="temple", tier_min=7, tier_max=7, sort_by="hp", limit=5):
    print(unit)