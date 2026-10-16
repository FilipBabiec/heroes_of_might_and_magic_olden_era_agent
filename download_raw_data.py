"""
Stage 1a: download raw Olden Era data from the official wiki.

Input:   Cargo tables on wiki.hoodedhorse.com (current patch)
Output:  data/raw/v<patch>/<Table>.json  +  data/raw/v<patch>/manifest.json
Done when: every table in TABLES is saved and the row counts in manifest.json
           match the counts shown on Special:CargoTables.
"""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# The wiki's API endpoint for programs (Special:CargoExport is meant for browsers
# and was refused with 403 from a script).
API_URL = "https://wiki.hoodedhorse.com/Heroes_of_Might_and_Magic_Olden_Era/api.php"
HEADERS = {"User-Agent": "HOE-agent (personal use)"}
PAGE_SIZE = 500  # the API's maximum per request
RAW_DIR = Path("data/raw")

# Which tables to download, which columns, and an optional row filter.
# Field names come from each table's schema page:
#   https://wiki.hoodedhorse.com/Heroes_of_Might_and_Magic_Olden_Era/Special:CargoTables/<Table>
TABLES: dict[str, dict] = {
    "Unit": {
        "fields": [
            "id", "unused", "faction", "tier", "source_path",
            "name_sid", "desc_sid", "base_sid", "upgrade_sid",
            "hp", "offence", "defence", "damage_min", "damage_max",
            "initiative", "speed", "luck", "morale",
            "energy_per_cast", "energy_per_round", "energy_per_take_damage",
            "action_points", "num_counters",
            "morale_min", "morale_max", "luck_min", "luck_max",
            "move_type", "creature_type", "immunities", "disablers",
            "shared_abilities", "native_biome", "ai_archetype", "tags", "leave_corpse",
            "gold_cost", "wood_cost", "ore_cost", "mercury_cost", "dust_cost",
            "crystal_cost", "gemstone_cost", "squad_value", "exp_bonus",
        ],
    },
    "UnitAbilityActive": {
        "fields": [
            "ability_id", "unit_id", "attack_type", "ordinal", "name_sid", "desc_sid",
            "rank", "cd", "energy_level", "active_type", "info_description",
            "excaption_in_tooltip",  # sic, spelled this way in the wiki schema
        ],
    },
    "UnitAbilityPassive": {
        "fields": [
            "ability_id", "unit_id", "ability_type", "ordinal",
            "name_sid", "desc_sid", "excaption_in_tooltip",
        ],
    },
    "Spell": {
        "fields": [
            "id", "name_sid", "school", "rank", "used_on_map", "icon",
            "magic_type_description", "is_special_magic", "is_unique_magic",
            "normal_magic_sid", "learn_cost_gemstones", "learn_cost_crystals",
            "learn_cost_mercury", "learn_cost_star_dust", "excaption_in_tooltip_sid",
            "up_effect_description_sid", "use_expand_tooltip", "energy_cost",
            "energy_type", "source_path",
        ],
    },
    "Artifact": {
        "fields": [
            "id", "name_sid", "description_sid", "upgrade_description_sid",
            "narrative_description_sid", "icon", "slot", "rarity", "artifact_set_id",
            "goods_value", "max_level", "cost_base", "cost_per_level",
            "reward_for_destroy", "is_special_item", "use_expand_tooltip",
            "can_destroy", "can_apply_bonus_always", "source_path",
        ],
    },
    "Faction": {
        "fields": [
            "id", "icon", "icon_faction_laws", "biome", "resource",
            "name_sid", "desc_sid", "source_path",
        ],
    },
    "Translation": {
        "fields": [
            "target_id", "type", "subtype", "variant", "language",
            "name", "description", "bonus_description",
        ],
        # 78,438 rows in 16 languages; English only is ~5,000.
        "where": 'language="en"',
    },
}


def fetch_table(table: str, fields: list[str], where: str | None = None) -> list[dict]:
    """Download all rows of one Cargo table, page by page."""
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "action": "cargoquery",
            "tables": table,
            "fields": ",".join(fields),
            "order_by": "_ID",
            "limit": PAGE_SIZE,
            "offset": offset,
            "format": "json",
        }
        if where:
            params["where"] = where
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:  # the API reports bad queries as JSON, with HTTP 200
            raise RuntimeError(f"{table}: {data['error'].get('info', data['error'])}")
        # Each row comes wrapped as {"title": {...fields...}}; unwrap it.
        page = [item["title"] for item in data["cargoquery"]]
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE
        time.sleep(1)


def fetch_schema(table: str) -> dict:
    params = {"action": "cargofields", "table": table, "format": "json"}
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"{table} schema: {data['error'].get('info', data['error'])}")
    return data["cargofields"]


def save_json(data: object, path: Path) -> None:
    """Write any JSON-serialisable object to a UTF-8 file, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def version_key(version: str) -> tuple[int, ...]:
    """'0.81.02' -> (0, 81, 2), so versions compare as numbers, not as text."""
    return tuple(int(part) for part in version.split("."))


def get_current_patch() -> str:
    """Return the newest patch version listed in the wiki's Patch table, e.g. '0.81.02'."""
    patches = fetch_table("Patch", ["id", "date"])
    if not patches:
        raise RuntimeError("The wiki's Patch table is empty, cannot tell which patch is current.")
    latest = max(patches, key=lambda p: version_key(p["id"]))
    return latest["id"]


def download_patch(patch: str) -> Path:
    """Download all TABLES for one patch into data/raw/v<patch>/. Skip if already there."""
    final_dir = RAW_DIR / f"v{patch}"
    if final_dir.exists():
        print(f"Raw data for v{patch} already exists in {final_dir}, skipping download.")
        return final_dir

    tmp_dir = RAW_DIR / f"v{patch}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)  # leftover from an interrupted earlier run

    row_counts: dict[str, int] = {}
    for table, spec in TABLES.items():
        rows = fetch_table(table, spec["fields"], spec.get("where"))
        save_json(rows, tmp_dir / f"{table}.json")
        save_json(fetch_schema(table), tmp_dir / f"{table}.schema.json")
        row_counts[table] = len(rows)
        print(f"  {table:<20} {len(rows):>6} rows")

    manifest = {
        "patch": patch,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": API_URL,
        "row_counts": row_counts,
        "queries": TABLES,
    }
    save_json(manifest, tmp_dir / "manifest.json")

    tmp_dir.rename(final_dir)  # only now does the patch folder "exist"
    return final_dir


def main() -> None:
    patch = get_current_patch()
    print(f"Current patch on the wiki: v{patch}")
    raw_dir = download_patch(patch)
    print(f"Raw data is in {raw_dir.resolve()}")


if __name__ == "__main__":
    main()