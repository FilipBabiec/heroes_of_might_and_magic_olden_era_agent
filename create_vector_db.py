'''
Input: data/processed/v<patch>/*.json
Output: data/hoe.vectordb, deleted and rebuilt on every run
Done when:
'''

import json
from collections import Counter
from pathlib import Path

from raw_data_processing import PROCESSED_DIR, load_json
from download_raw_data import version_key, save_json
from create_sql_db import load_data



FACTION_MAPPINGS = {
    "human": "Temple",
    "nature": "Grove",
    "undead": "Necropolis",
    "demon": "Hive",
    "unfrozen": "Schism",
    "dungeon": "Dungeon",
    "neutral": "Neutral"
}

def clean_metadata(meta: dict) -> dict:
    """Chroma accepts only str/int/float/bool, so drop keys whose value is None."""
    return {k: v for k, v in meta.items() if v is not None}


def build_documents(data: dict):
    documents = []
    for table, rows in data.items():
        print(table)
        for row in rows:
                # if table == "units":
                #     if row["description"] is not None:
                #         document = {
                #             "id": row["id"],
                #             "text": f"{row["name"]}. {row["description"]} {row["attack_description"]} {row["movement_description"]}",
                #             "metadata": clean_metadata
                #                 ({
                #                     "faction": row["faction"],
                #                     "tier": row["tier"],
                #                     "creature_type": row["creature_type"],
                #                     "record_id": row["id"],
                #                     "name": row["name"],
                #                     "doc_type": "unit",
                #                 })
                #         }
                #         documents.append(document)
                if table == "abilities":
                    if row["description"] is not None:
                        if row["shared"]:
                            owner = f"Shared {row['kind']} ability, used by several units."
                        else:
                            faction_name = FACTION_MAPPINGS.get(row["faction"], row["faction"])
                            owner = f"{row['kind'].capitalize()} ability of {row['unit_name']} ({faction_name}, tier {row['tier']})."
                        document = {
                            "id": row["ability_id"],
                            "text": f"{row['name']}. {owner} {row['description']}",
                            "metadata": clean_metadata
                                ({
                                    "unit": row["unit_name"],
                                    "tier": row["tier"],
                                    "faction": row["faction"],
                                    "shared": row["shared"],
                                    "kind": row["kind"],
                                    "record_id": row["ability_id"],
                                    "name": row["name"],
                                    "doc_type": "ability"
                                })
                        }
                        documents.append(document)
                if table == "spells":
                    if row["levels"]:
                        spell_levels = ""
                        for level in row["levels"]:
                            spell_levels += f"Level {level["level"]}: {level["description"]}"
                            if level["bonus"]:
                                spell_levels += f" (Change: {level["bonus"]})"
                            spell_levels += "\n"
                        document = {
                            "id": row["id"],
                            "text": f"{row["name"]}, {row["school"]} magic. Levels: {spell_levels}",
                            "metadata": clean_metadata
                                ({
                                    "school": row["school"],
                                    "rank": row["rank"],
                                    "record_id": row["id"],
                                    "name": row["name"],
                                    "doc_type": "spell"
                                })
                        }
                        documents.append(document)
                if table == "artifacts":
                    if row["description"] is not None:
                        upgrade = f" Upgrade: {row['upgrade']}" if row["upgrade"] else ""
                        document = {
                            "id": row["id"],
                            "text": f"{row['name']}. {row['description']}{upgrade}",
                            "metadata": clean_metadata
                                ({
                                    "rarity": row["rarity"],
                                    "slot": row["slot"],
                                    "record_id": row["id"],
                                    "name": row["name"],
                                    "doc_type": "artifact"
                                })
                        }
                        documents.append(document)
    return documents

def main() -> None:
    data = load_data()
    docs = build_documents(data)
    save_json(docs, PROCESSED_DIR / "V0.81.02/documents.json")

if __name__ == "__main__":
    main()
