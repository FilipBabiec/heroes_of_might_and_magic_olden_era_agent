import json
from pathlib import Path
import re
import html
from collections import Counter

from download_raw_data import RAW_DIR, version_key

TEXT_FIELDS = ("name", "description", "bonus_description")
MARKUP = r"&#?\w+;|<[^>]+>|\[\[|\]\]|'{2,}"
NEEDED_TYPES = {"unit", "unit_ability", "spell", "spell_rank",
                "artifact", "artifact_narrative", "artifact_upgrade",
                "attack_archetype", "movement"}
MOVEMENT_REFS = {"base_passive_flyer", "base_passive_blink"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def latest_patch_dir() -> Path:
    """Newest complete patch folder in data/raw (ignores unfinished .tmp downloads)."""
    dirs = [p for p in RAW_DIR.glob("v*") if p.is_dir() and not p.name.endswith(".tmp")]
    if not dirs:
        raise FileNotFoundError(f"No raw data in {RAW_DIR}, run download_raw.py first.")
    return max(dirs, key=lambda p: version_key(p.name[1:]))  # "v0.81.02" -> "0.81.02"


def load_patch(patch_dir: Path) -> dict[str, dict]:
    """Load rows and schema for every table listed in the manifest."""
    manifest = load_json(patch_dir / "manifest.json")
    tables = {}
    for table, expected_rows in manifest["row_counts"].items():
        rows = load_json(patch_dir / f"{table}.json")
        schema = load_json(patch_dir / f"{table}.schema.json")
        if len(rows) != expected_rows:
            raise ValueError(f"{table}: manifest says {expected_rows} rows, file has {len(rows)}")
        tables[table] = {"rows": rows, "schema": schema}
    return tables


def normalize_keys(row: dict) -> dict:
    return {key.replace(" ", "_"): value for key, value in row.items()}


def is_used(row: dict) -> bool:
    return row["unused"] is not True


def convert_types(table: dict, schema_table: dict) -> dict:
    return {key: convert_value(value, schema_table[key]) for key, value in table.items()}


def convert_value(value: str, schema_dict: dict):
    if "isList" in schema_dict:
        if value is None or value == "":
            return []
        return [convert_scalar(item.strip(), schema_dict["type"]) for item in value.split(schema_dict["delimiter"])]
    return convert_scalar(value, schema_dict["type"])


def convert_scalar(value: str, schema: str):
    if value is None or value == "":
        return None
    if schema == "Integer":
        return int(value)
    if schema == "Boolean":
        if value == "1":
            return True
        else:
            return False
    if schema == "String":
        return str(value)
    return value


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = html.unescape(text)  # &lt;br/&gt; -> <br/>, &amp; -> &
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)  # <br>, <br/>, <br /> -> line break
    text = re.sub(r"<[^>]+>", "", text)  # drop other tags, keep the text inside
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # [[target|label]] -> label, [[x]] -> x
    text = re.sub(r"'{2,}", "", text)  # ''italic'' and '''bold''' markers
    text = re.sub(r" *\n *", "\n", text)  # no spaces around line breaks
    return text.strip()


def clean_row(row: dict) -> dict:
    return {key: clean_text(value) if key in TEXT_FIELDS else value for key, value in row.items()}


def build_text_lookup(translation_rows: list[dict]) -> dict[tuple, dict]:
    """Index English text rows by (target_id, type, variant)."""
    lookup = {}
    for row in translation_rows:
        if row["type"] not in NEEDED_TYPES:
            continue
        key = (row["target_id"], row["type"], row["variant"])
        if key in lookup:
            raise ValueError(f"Duplicate translation key: {key}")
        lookup[key] = row
    return lookup


def build_units(unit_rows: list[dict], text: dict) -> tuple[list[dict], list[str]]:
    """Left join: every used unit is kept; units without English text get name=None."""
    units, missing = [], []
    for unit in unit_rows:
        t = text.get((unit["id"], "unit", None))
        if t is None:
            missing.append(unit["id"])
        attacks = attack_archetypes_of(unit, text)
        attack_text = text.get((attacks[0], "attack_archetype", None)) if len(attacks) == 1 else None
        move_text = text.get((unit["move_type"], "movement", None))
        units.append({
            **unit,
            "attack_type": attacks[0] if len(attacks) == 1 else None,
            "attack_description": attack_text["description"] if attack_text else None,
            "movement_name": move_text["name"] if move_text else None,
            "movement_description": move_text["description"] if move_text else None,
            "name": t["name"] if t else None,
            "description": t["description"] if t else None
        })
    return units, missing


def attack_archetype_id(ref: str, text: dict) -> str | None:
    """'base_passive_ranged_attack' -> 'ranged_attack' if that attack type exists, else None."""
    archetype = ref.removeprefix("base_passive_")
    return archetype if (archetype, "attack_archetype", None) in text else None


def attack_archetypes_of(unit: dict, text: dict) -> list[str]:
    found = []
    for name_id in unit["shared_abilities"]:
        ref = name_id.removesuffix("_name")
        if ref not in MOVEMENT_REFS and (archetype := attack_archetype_id(ref, text)):
            found.append(archetype)
    return found


def build_abilities(active_ability_rows: list[dict], passive_ability_rows: list[dict], units_by_id: dict[str, dict], text: dict) -> tuple[list[dict], list[str], list[str]]:
    abilities, no_unit, no_text = [], [], []
    for kind, rows in (("active", active_ability_rows), ("passive", passive_ability_rows)):
        for ability in rows:
            if ability["unit_id"] is None:  # shared ability, linked via Unit.shared_abilities
                unit = None
            else:
                unit = units_by_id.get(ability["unit_id"])
                if unit is None:  # belongs to an unused unit
                    no_unit.append(ability["ability_id"])
                    continue
            t = text.get((ability["ability_id"], "unit_ability", None))
            if t is None:
                no_text.append(ability["ability_id"])
            abilities.append({
                **ability,
                "kind": kind,
                "shared": unit is None,
                "unit_name": unit["name"] if unit else None,
                "faction": unit["faction"] if unit else None,
                "tier": unit["tier"] if unit else None,
                "name": t["name"] if t else None,
                "description": t["description"] if t else None,
            })
    return abilities, no_unit, no_text


def build_shared_links(units: list[dict], abilities_by_id: dict[str, dict], text: dict) -> tuple[list[dict], list[str]]:
    """Unit <-> shared-ability pairs. Movement and attack types are skipped (they live on the unit)."""
    links, unresolved = [], set()
    for unit in units:
        for name_id in unit["shared_abilities"]:
            ref = name_id.removesuffix("_name")
            if ref in abilities_by_id:
                links.append({"unit_id": unit["id"], "ability_id": ref})
            elif ref in MOVEMENT_REFS or attack_archetype_id(ref, text):
                pass                    # known kind, already stored on the unit record
            else:
                unresolved.add(ref)
    return links, sorted(unresolved)


def process_data(tables: dict) -> dict:
    for name, table in tables.items():
        table["rows"] = [normalize_keys(row) for row in table["rows"]]
        table["rows"] = [convert_types(row, table["schema"]) for row in table["rows"]]
        if name == "Unit":
            table["rows"] = [row for row in table["rows"] if is_used(row)]
        if name == "Translation":
            table["rows"] = [clean_row(row) for row in table["rows"]]
        # rows = table["rows"]
        # schema = table["schema"]
    return tables


def build_final_tables(tables: dict) -> dict:
    pass


### TESTING ###
def check(tables: dict):
    units = {u["id"]: u for u in tables["Unit"]["rows"]}   # look up units by id
    angel = units["angel"]
    # Checking unused units
    assert len(units) == 147
    # Checking types
    assert angel["hp"] == 225 and isinstance(angel["hp"], int)
    # Checking lists
    assert angel["immunities"] == ["overTime_effect_blood", "overTime_effect_poison", "buff_effect_poison"]
    # Checking missing values
    assert angel["wood_cost"] is None
    # Checking lists
    assert all(isinstance(u["immunities"], list) for u in units.values())   # never None
    # Converting html
    leftover = [t for row in tables["Translation"]["rows"] for f in TEXT_FIELDS
                if (t := row[f]) and re.search(MARKUP, t)]
    assert not leftover, f"{len(leftover)} texts still contain markup, e.g. {leftover[0][:120]!r}"
    print("step 1-5 checks passed")


def build_all(tables: dict) -> dict:
    """Step 6: join stats with text. Returns the records plus a report of everything left out."""
    text = build_text_lookup(tables["Translation"]["rows"])

    units, units_without_text = build_units(tables["Unit"]["rows"], text)
    units_by_id = {u["id"]: u for u in units}

    abilities, of_unused_units, abilities_without_text = build_abilities(
        tables["UnitAbilityActive"]["rows"], tables["UnitAbilityPassive"]["rows"], units_by_id, text)
    abilities_by_id = {a["ability_id"]: a for a in abilities}

    links, unresolved = build_shared_links(units, abilities_by_id, text)

    return {
        "units": units,
        "abilities": abilities,
        "shared_links": links,
        "report": {
            "units_without_text": units_without_text,
            "abilities_of_unused_units": of_unused_units,
            "abilities_without_text": abilities_without_text,
            "unresolved_references": unresolved,
        },
    }


def check_invariants(tables: dict, result: dict) -> None:
    """Rules that must hold for every patch. A failure means a bug in this code."""
    units, abilities, report = result["units"], result["abilities"], result["report"]
    ability_input = len(tables["UnitAbilityActive"]["rows"]) + len(tables["UnitAbilityPassive"]["rows"])

    assert len({u["id"] for u in units}) == len(units), "duplicate unit ids"
    assert len({a["ability_id"] for a in abilities}) == len(abilities), "duplicate ability ids"
    assert len(abilities) + len(report["abilities_of_unused_units"]) == ability_input, "abilities lost"
    assert all(u["attack_type"] is not None for u in units), "unit without attack type"
    assert report["unresolved_references"] == [], f"unknown references: {report['unresolved_references']}"


def check_known_values(result: dict) -> None:
    """Facts about patch 0.81.02. After a game patch, a failure may mean the game changed: look, then update."""
    units = {u["id"]: u for u in result["units"]}
    abilities = {a["ability_id"]: a for a in result["abilities"]}

    assert len(units) == 147
    assert units["angel"]["hp"] == 225
    assert units["angel_upg"]["name"] == "Archangel"
    assert abilities["angel_upg_2"]["name"] == "Devouring Chalice"
    assert abilities["angel_upg_2"]["unit_name"] == "Archangel"
    assert abilities["base_passive_noncounter"]["name"] == "Swift Strike"
    angel_upg_links = {l["ability_id"] for l in result["shared_links"] if l["unit_id"] == "angel_upg"}
    assert angel_upg_links == {"common_attack_1", "base_passive_sharpshooter", "base_passive_duelist"}


def print_summary(result: dict) -> None:
    shared = sum(a["shared"] for a in result["abilities"])
    print(f"units: {len(result['units'])}")
    print(f"abilities: {len(result['abilities'])} ({shared} shared)")
    print(f"shared-ability links: {len(result['shared_links'])}")
    for name, items in result["report"].items():
        print(f"{name}: {len(items)} {items if 0 < len(items) <= 10 else ''}")


def main() -> None:
    tables = process_data(load_patch(latest_patch_dir()))
    result = build_all(tables)
    check_invariants(tables, result)
    check_known_values(result)
    print_summary(result)


if __name__ == "__main__":
    main()
