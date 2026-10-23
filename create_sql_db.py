'''
Input: data/processed/v<patch>/*.json
Output: data/hoe.sqlite, deleted and rebuilt on every run
Done when: the row count of each table equals manifest.json, and a few hand-written SQL queries return what the wiki shows
'''

import json
import sqlite3
from pathlib import Path

from raw_data_processing import PROCESSED_DIR, load_json
from download_raw_data import version_key

SQL_TYPES = {bool: "INTEGER", int: "INTEGER", float: "REAL", str: "TEXT", list: "TEXT", dict: "TEXT"}
DB_PATH = Path("data/hoe.sqlite")


def load_data() -> dict:
    dirs = [p for p in PROCESSED_DIR.glob("v*") if p.is_dir() and not p.name.endswith(".tmp")]
    if not dirs:
        raise FileNotFoundError(f"No raw data in {PROCESSED_DIR}, run download_raw.py first.")
    patch_dir = max(dirs, key=lambda p: version_key(p.name[1:]))
    manifest = load_json(patch_dir / "manifest.json")
    tables = {}
    for table, expected_rows in manifest["row_counts"].items():
        rows = load_json(patch_dir / f"{table}.json")
        if len(rows) != expected_rows:
            raise ValueError(f"{table}: manifest says {expected_rows} rows, file has {len(rows)}")
        tables[table] = rows
    return tables


def infer_columns(rows: list[dict]) -> dict[str, str]:
    """Column name -> SQL type, from the Python types actually present in the rows."""
    types = {}
    for row in rows:
        for key, value in row.items():
            if value is None:
                types.setdefault(key, None)          # known column, type still open
                continue
            sql_type = SQL_TYPES[type(value)]
            known = types.get(key)
            if known is None:
                types[key] = sql_type
            elif known != sql_type:
                if {known, sql_type} == {"INTEGER", "REAL"}:
                    types[key] = "REAL"              # 5 and 2.5 in one column -> REAL
                else:
                    raise TypeError(f"column {key!r} mixes {known} and {sql_type}")
    return {key: sql_type or "TEXT" for key, sql_type in types.items()}


def to_sql_value(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def split_spell_levels(spells: list[dict]) -> tuple[list[dict], list[dict]]:
    flat, levels = [], []
    for spell in spells:
        flat.append({k: v for k, v in spell.items() if k != "levels"})
        for lvl in spell["levels"]:
            levels.append({"spell_id": spell["id"], **lvl})
    return flat, levels


def create_and_fill(con: sqlite3.Connection, table: str, rows: list[dict], primary_key: str | None = None) -> None:
    columns = infer_columns(rows)
    col_defs = ", ".join(f"{c} {t}" + (" PRIMARY KEY" if c == primary_key else "") for c, t in columns.items())
    con.execute(f"CREATE TABLE {table} ({col_defs})")

    placeholders = ", ".join(f":{c}" for c in columns)       # :id, :tier, :hp, ...
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    con.executemany(sql, [{c: to_sql_value(row.get(c)) for c in columns} for row in rows])


def build_database(data: dict[str, list[dict]]) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()                              # rebuild from scratch every run
    spells, spell_levels = split_spell_levels(data["spells"])
    with sqlite3.connect(DB_PATH) as con:             # commits at the end if nothing failed
        create_and_fill(con, "units", data["units"], primary_key="id")
        create_and_fill(con, "abilities", data["abilities"], primary_key="ability_id")
        create_and_fill(con, "shared_links", data["shared_links"])
        create_and_fill(con, "spells", spells, primary_key="id")
        create_and_fill(con, "spell_levels", spell_levels)
        create_and_fill(con, "artifacts", data["artifacts"], primary_key="id")
    con.close()


def run_sql(con: sqlite3.Connection):
    for row in con.execute("SELECT faction, COUNT(*) AS units, ROUND(AVG(hp), 1) AS avg_hp FROM units GROUP BY faction ORDER BY avg_hp DESC;"):
        print(row)


def main() -> None:
    data = load_data()
    build_database(data)
    # run_sql(sqlite3.connect(DB_PATH))

if __name__ == "__main__":
    main()



