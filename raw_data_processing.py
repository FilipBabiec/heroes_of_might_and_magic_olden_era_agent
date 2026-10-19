import json
from pathlib import Path

from download_raw_data import RAW_DIR, version_key  # reuse, don't copy


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


def check_for_empty(row: dict) -> dict:
    return {key: value for key, value in row.items() if value != ""}

def check_type(value: str, schema: str):
    print(f"{value}, {schema}")
    if schema == "Integer":
        return int(value)
    elif schema == "Boolean":
        return bool(value)
    elif schema == "String":
        return str(value)
    else:
        return value


def convert_types(table: dict, schema_table: dict) -> dict:
    schema = schema_table["id"]["type"]
    return {key: check_type(value, schema_table[key]["type"]) for key, value in table.items()}

tables = load_patch(latest_patch_dir())

# unit_rows = tables["Unit"]["rows"]        # list of 152 dicts
# unit_schema = tables["Unit"]["schema"]    # dict of 45 field specs
# print(unit_rows[43]["id"], unit_schema["id"])

for name, table in tables.items():
    table["rows"] = [normalize_keys(row) for row in table["rows"]]
    table["rows"] = [check_for_empty(row) for row in table["rows"]]
    table["rows"] = [convert_types(row, table["schema"]) for row in table["rows"]]
    rows = table["rows"]
    schema = table["schema"]
    for row in rows:
        for key, value in row.items():
            print(f"{key}: {value}")

print("=== End ===")