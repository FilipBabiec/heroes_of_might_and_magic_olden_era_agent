import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/hoe.sqlite")


con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row
    # for row in con.execute("SELECT faction, COUNT(*) AS units, ROUND(AVG(hp), 1) AS avg_hp FROM units GROUP BY faction ORDER BY avg_hp DESC;"):
    #     print(row)
    # for row in con.execute("SELECT DISTINCT s.id, s.name FROM spells s JOIN spell_levels l ON l.spell_id = s.id WHERE l.description LIKE '%restores%HP%' LIMIT 5;"):
    #     print(row)
    #
    # print("==================")
    # for row in con.execute("SELECT ability_id, unit_name, name, description FROM abilities WHERE description LIKE '%positive effect%' LIMIT 5;"):
    #     print(row)
    #
    # print("==================")
    # for row in con.execute("SELECT ability_id, name, description FROM abilities WHERE description LIKE '%immune%' OR name LIKE '%Immunity%' LIMIT 5;"):
    #     print(row)

print("==================")
for row in con.execute("SELECT ability_id, name, description FROM abilities WHERE ability_id LIKE 'angel_upg_alt_passive_1' OR ability_id LIKE 'angel_upg_passive_1' LIMIT 5;"):
    print(dict(row))

for row in con.execute("""
        SELECT s.name, l.level, l.description, l.bonus
        FROM spells s
        JOIN spell_levels l ON l.spell_id = s.id
        WHERE s.id = 'space_10_magic_mirror_copy'
        ORDER BY l.level"""):
    print(dict(row))