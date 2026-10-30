import pytest
from tools import query_units


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