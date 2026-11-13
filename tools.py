import json
import sqlite3
from pathlib import Path
from typing import Literal, get_args
from collections import defaultdict
from functools import cache
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from create_sql_db import DB_PATH
from sentence_transformers import CrossEncoder


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
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "hoe.sqlite"
NO_TEXT = "No description available in the game data."
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION = "hoe_rules"
RERANKER_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANK_INSTRUCTION = ("Given a player's question about the game Heroes of Might and Magic: Olden Era, "
                      "retrieve the ability, spell or artifact whose effect answers the question")
CANDIDATES = 30  # stage 1 width: how many embedding hits the reranker gets to see
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERANK_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32  # bf16 is fast on GPU, slow on CPU


assert set(FACTION_IDS) == set(get_args(Faction)), "FACTION_IDS and Faction disagree"


@cache
def get_model():
    model = SentenceTransformer(MODEL_ID, model_kwargs={"dtype": torch.float32}, device="cpu")
    return model


@cache
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    assert collection.metadata["embedding_model"] == MODEL_ID, "collection built with another model"
    return collection


# We use the device="cpu" so the reranker qwen model use the CPU RAM instead of the VRAM which is also used by the qwen3 model that agent uses.
# They would compete for VRAM since they would run simultanously and if so they would spill to CPU RAM, and it hinders the speed by much.
# It is better to allocate the smaller model with simpler task (the reranker) already on the CPU so the bigger model can use freely the GPU.
@cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_ID, device=DEVICE, model_kwargs={"dtype": RERANK_DTYPE},
                        prompts={"game": RERANK_INSTRUCTION}, default_prompt_name="game")


def connect_read_only() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def check_choice(name: str, value, allowed: tuple) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"Invalid {name} {value!r}. Allowed: {', '.join(allowed)}.")


def check_int_range(name: str, n: int, min: int, max: int):
    if n not in range(min,max+1):
        raise ValueError(f"Invalid {name} value: {n}. Allowed: {min}-{max}")


def check_tier_values(tier_min: int | None = None, tier_max: int | None = None):
    if tier_min is not None and (tier_min < 1 or tier_min > 7):
        raise ValueError(f"Invalid tier_min value. Allowed: 1-7.")
    if tier_max is not None and (tier_max > 7 or tier_max < 1):
        raise ValueError(f"Invalid tier_max value. Allowed: 1-7.")
    if tier_min is not None and tier_max is not None and tier_min > tier_max:
        raise ValueError(f"Invalid tier range. Tier_min must be lower than tier_max.")

def tool_setup():
    pass


def retrieve(query: str, n: int) -> list[dict]:
    """Stage 1: embeddings narrow 790 docs to CANDIDATES. Stage 2: the reranker picks the top n."""
    emb = get_model().encode(query, prompt_name="query", normalize_embeddings=True)
    res = get_collection().query(query_embeddings=emb, n_results=CANDIDATES)
    ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]

    scores = get_reranker().predict([(query, doc) for doc in docs])
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:n]
    return [{"id": ids[i], "doc_type": metas[i]["doc_type"], "score": round(float(scores[i]), 2)}
            for i in order]


# input: search text + optional filters | output: a list of compact results, best first, shaped by type
def search_rules(query: str, n: int = 5) -> list[dict]:
    """Find abilities, spells and artifacts whose rules text matches the query.
    Use the game's official terms (e.g. 'positive effects', not 'buffs')."""
    check_int_range("n", n, 1, 10)

    # Swapped the raw query and documents embeddings comparison for reranker (2 stage comparison)
    hits = retrieve(query, n)
    # embedding = get_model().encode(query, prompt_name="query", normalize_embeddings=True)
    # results = get_collection().query(query_embeddings=embedding, n_results=n)
    # hits = [{"id": doc_id, "doc_type": meta["doc_type"], "similarity": round(1 - dist / 2, 2)}
    #         for doc_id, dist, meta in zip(results["ids"][0], results["distances"][0], results["metadatas"][0])]

    con = connect_read_only()
    try:
        records = fetch_abilities(con, [h["id"] for h in hits if h["doc_type"] == "ability"])
        records.update(fetch_spells(con, [h["id"] for h in hits if h["doc_type"] == "spell"]))
        records.update(fetch_artifacts(con, [h["id"] for h in hits if h["doc_type"] == "artifact"]))
    finally:
        con.close()

    # walk the hits in the reranker's order; the SQL IN query returned rows in arbitrary order
    return [{**records[h["id"]], "score": h["score"]} for h in hits if h["id"] in records]


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
    """List or rank units by faction, tier range and a stat."""
    check_choice("faction", faction, get_args(Faction))
    check_choice("sort_by", sort_by, get_args(SortField))
    check_choice("move_type", move_type, get_args(MoveType))
    check_choice("attack_category", attack_category, get_args(AttackCategory))
    check_tier_values(tier_min, tier_max)


    where_sql, where_params = build_where(faction, tier_min, tier_max)
    order_sql, order_params = build_order(sort_by, sort_desc, limit)
    sql = f"SELECT {COLUMNS} FROM units {where_sql} {order_sql}"

    con = connect_read_only()
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


def fetch_abilities(con: sqlite3.Connection, ability_ids: list[str]) -> dict[str, dict]:
    """Compact records for the given abilities: {ability_id: record}."""
    if not ability_ids:
        return {}
    placeholders = ", ".join("?" * len(ability_ids))

    rows = con.execute(f"""SELECT ability_id, name, kind, shared, unit_name, faction, tier, description
                           FROM abilities WHERE ability_id IN ({placeholders})""", ability_ids).fetchall()

    # abilities -> units (the reverse direction of fetch_ability_names)
    unit_rows = con.execute(f"""SELECT l.ability_id, u.name FROM shared_links l
                                JOIN units u ON u.id = l.unit_id
                                WHERE l.ability_id IN ({placeholders})
                                ORDER BY u.name""", ability_ids).fetchall()
    units_by_ability = defaultdict(list)
    for ability_id, unit_name in unit_rows:
        units_by_ability[ability_id].append(unit_name)

    records = {}
    for row in rows:
        record = {
            "type": "ability",
            "name": row["name"],
            "kind": row["kind"],
            "description": row["description"] or "No description available in the game data.",
        }
        if row["shared"]:
            record["units"] = units_by_ability[row["ability_id"]]
        else:
            record["unit"] = row["unit_name"]
            record["faction"] = FACTION_DISPLAY.get(row["faction"], row["faction"])
            record["tier"] = row["tier"]
        records[row["ability_id"]] = record
    return records


def fetch_spells(con: sqlite3.Connection, spell_ids: list[str]) -> dict[str, dict]:
    """Compact spell records with their levels, in level order: {spell_id: record}."""
    if not spell_ids:
        return {}
    placeholders = ", ".join("?" * len(spell_ids))
    rows = con.execute(f"SELECT id, name, school, rank FROM spells WHERE id IN ({placeholders})",
                       spell_ids).fetchall()
    level_rows = con.execute(f"""SELECT spell_id, level, description, bonus FROM spell_levels
                                 WHERE spell_id IN ({placeholders})
                                 ORDER BY spell_id, level""", spell_ids).fetchall()
    levels_by_spell = defaultdict(list)
    for lvl in level_rows:
        level = {"level": lvl["level"], "description": lvl["description"] or NO_TEXT}
        if lvl["bonus"]:
            level["change"] = lvl["bonus"]
        levels_by_spell[lvl["spell_id"]].append(level)
    return {row["id"]: {"type": "spell",
                        "name": row["name"],
                        "school": (row["school"] or "").capitalize() or None,
                        "tier": row["rank"],
                        "levels": levels_by_spell[row["id"]]}
            for row in rows}


def fetch_artifacts(con: sqlite3.Connection, artifact_ids: list[str]) -> dict[str, dict]:
    """Compact artifact records: {artifact_id: record}."""
    if not artifact_ids:
        return {}
    placeholders = ", ".join("?" * len(artifact_ids))
    rows = con.execute(f"""SELECT id, name, rarity, slot, description, upgrade
                           FROM artifacts WHERE id IN ({placeholders})""", artifact_ids).fetchall()
    records = {}
    for row in rows:
        record = {"type": "artifact", "name": row["name"], "rarity": row["rarity"],
                  "slot": row["slot"], "description": row["description"] or NO_TEXT}
        if row["upgrade"]:
            record["upgrade"] = row["upgrade"]
        records[row["id"]] = record
    return records
