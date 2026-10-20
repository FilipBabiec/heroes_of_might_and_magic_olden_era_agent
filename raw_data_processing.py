import json
from pathlib import Path
import re
import html
from collections import Counter

from download_raw_data import RAW_DIR, version_key

TEXT_FIELDS = ("name", "description", "bonus_description")
MARKUP = r"&#?\w+;|<[^>]+>|\[\[|\]\]|'{2,}"


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


def main() -> None:
    tables = process_data(load_patch(latest_patch_dir()))
    check(tables)


if __name__ == "__main__":
    main()
