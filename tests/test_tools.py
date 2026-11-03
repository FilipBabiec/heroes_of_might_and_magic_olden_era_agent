import pytest
from tools import query_units
from tools import search_rules


def test_faction_filter():
    units = query_units(faction="temple", limit=25)
    assert units, "expected at least one Temple unit"
    assert all(u["faction"] == "Temple" for u in units)


def test_sort_and_limit():
    units = query_units(faction="temple", tier_min=7, tier_max=7, sort_by="hp", limit=5)
    hps = [u["hp"] for u in units]
    assert len(units) <= 5
    assert hps == sorted(hps, reverse=True), "HP must be in descending order"
    names = {u["name"] for u in units}
    assert {"Angel", "Archangel"} <= names


def test_bad_faction_lists_allowed_values():
    with pytest.raises(ValueError, match="Allowed"):
        query_units(faction="haven")


def find(units: list[dict], name: str) -> dict:
    """The unit with this name; fails the test (with the names found) if it isn't there."""
    matches = [u for u in units if u["name"] == name]
    assert matches, f"{name!r} not in results: {[u['name'] for u in units]}"
    return matches[0]


def test_tier_range():
    units = query_units(faction="temple", tier_min=6, tier_max=7, limit=25)
    assert units, "expected Temple units of tier 6-7"
    assert all(6 <= u["tier"] <= 7 for u in units)


def test_known_values_minotaur_lord():
    lord = find(query_units(faction="dungeon", tier_min=4, tier_max=4), "Minotaur Lord")
    assert lord["hp"] == 40
    assert lord["offence"] == 11
    assert lord["defence"] == 15
    assert lord["cost"] == {"gold": 530}


def test_archangel_abilities():
    archangel = find(query_units(faction="temple", tier_min=7, tier_max=7), "Archangel")
    assert "Devouring Chalice" in archangel["abilities"]
    assert "Sharpshooter" in archangel["abilities"]


def test_bad_inputs():
    with pytest.raises(ValueError, match="Allowed"):
        units = query_units(faction="temple", tier_min=4, tier_max=6, upgraded_only=True, sort_by="ai_archetype")


@pytest.mark.parametrize("kwargs", [
    {"faction": "haven"},
    {"sort_by": "ai_archetype"},
    {"limit": 0},
    {"limit": 26},
    {"tier_min": 9},
    {"tier_min": 5, "tier_max": 3},
])
def test_bad_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        query_units(**kwargs)


def test_limit_25_is_allowed():
    assert len(query_units(limit=25)) <= 25


def find_name(results: list[dict], name: str, within: int = 3) -> dict:
    top = results[:within]
    matches = [r for r in top if r["name"] == name]
    assert matches, f"{name!r} not in top {within}: {[r['name'] for r in top]}"
    return matches[0]


def test_search_swift_strike():
    first = search_rules("enemies do not counterattack", n=3)[0]
    assert first["name"] == "Swift Strike"
    assert first["type"] == "ability"
    assert first["units"], "shared ability should list its units"


def test_search_healing_water_levels():
    spell = find_name(search_rules("restore HP of a friendly creature", n=5), "Healing Water")
    assert [lvl["level"] for lvl in spell["levels"]] == [1, 2, 3, 4]
    assert spell["levels"][1]["change"] == "HP per Spell Power: 5 > 10."


def test_search_ambassadors_sash():
    artifact = find_name(search_rules("persuasion in diplomacy", n=5), "Ambassador's Sash")
    assert artifact["description"] == "+10% to Persuasion Power in Diplomacy."


@pytest.mark.parametrize("n", [0, 11])
def test_search_bad_n(n):
    with pytest.raises(ValueError):
        search_rules("anything", n=n)

