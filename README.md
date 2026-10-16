# HOE Agent

A learning project: an AI advisor for **Heroes of Might and Magic: Olden Era** that answers questions about units, spells, artifacts and factions while you play.

The goal is to learn RAG and agent design on a small, real dataset: when to use SQL vs. semantic search, how to give an agent tools, how to evaluate its answers, and how to expose tools through MCP.

> **Status:** early development. Sections marked *(planned)* are not built yet.

## What it does

Ask things like:

- "Which tier IV unit has the most HP?" → answered with SQL over unit stats
- "Which units can't be counterattacked?" → answered with semantic search over ability descriptions
- "Show me everything about the Black Dragon." → exact lookup by name

Every answer cites the table row or description it came from.

## Architecture

```
Wiki Cargo export (JSON)
        │  download.py (cached locally)
        ▼
   data/raw/*.json
        │
        ├──► SQLite   ── numeric stats (HP, attack, initiative, cost, tier)
        └──► Chroma   ── text (ability, spell, artifact, skill descriptions)
                 │
                 ▼
     Agent with three tools
       • query_units   – filter/sort stats with SQL
       • search_rules  – semantic search with metadata filters (type, faction)
       • get_details   – exact lookup of one unit, spell or artifact
                 │
                 ▼
     MCP server ──► Claude Desktop or any other MCP client   (planned)
```

**Why two stores:** numeric questions ("highest HP", "cheapest tier VI") need exact filtering and sorting, which semantic search cannot do reliably. Meaning-based questions ("spells that help against undead") are what embeddings are good at.

## Data

Game data comes from the [Heroes of Might and Magic: Olden Era Official Wiki](https://wiki.hoodedhorse.com/Heroes_of_Might_and_Magic_Olden_Era/Main_Page), which imports it from the game files into [Cargo tables](https://wiki.hoodedhorse.com/Heroes_of_Might_and_Magic_Olden_Era/Special:CargoTables). The download script uses the wiki's `Special:CargoExport` endpoint.

Tables used: `Unit`, `UnitAbilityActive`, `UnitAbilityPassive`, `Spell`, `Artifact`, `Faction`, `Translation` (English only).

- Raw downloads go to `data/raw/` and are **not committed**. Run the download script to recreate them.
- A small pinned sample lives in `tests/fixtures/` so tests do not break when a game patch changes the live data.

**License of the data:** wiki content is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Files in `tests/fixtures/` are derived from it and are distributed under the same license. Heroes of Might and Magic and Olden Era are trademarks of their respective owners. This is an unofficial, non-commercial fan project.

## Project structure

```
hoe-agent/
├── src/hoe_agent/
│   ├── download.py      # fetch Cargo tables → data/raw/
│   ├── build_sql.py     # raw JSON → SQLite        (planned)
│   ├── build_vectors.py # descriptions → Chroma    (planned)
│   ├── tools.py         # query_units, search_rules, get_details (planned)
│   ├── agent.py         # agent loop                (planned)
│   └── mcp_server.py    # MCP server                (planned)
├── tests/
│   ├── fixtures/        # pinned data sample (CC BY-SA 4.0)
│   └── eval_questions.yaml  # questions with expected answers (planned)
├── data/                # generated, git-ignored
└── README.md
```

## Getting started

```bash
git clone https://github.com/FilipBabiec/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m hoe_agent.download     # downloads data to data/raw/
```

## Roadmap

- [ ] Download and cache Cargo tables, write a manifest (timestamp, row counts)
- [ ] Clean descriptions (wiki markup, placeholders) and build SQLite
- [ ] Embed descriptions into Chroma with metadata (type, faction, tier)
- [ ] Implement the three tools and test each one on its own
- [ ] Agent loop with cited answers; log tokens and cost per question
- [ ] Evaluation set with expected answers taken from the data tables
- [ ] MCP server, connected to Claude Desktop

## What I'm learning

| Concept | Where it shows up |
|---|---|
| API calls, tokens, cost | Token and cost logging per question; local vs. cloud model comparison |
| Prompt templates, few-shot | Answer prompt with an example of a cited answer |
| Chunking, embeddings, Chroma | One description = one chunk, with metadata filters |
| Tool calling | `query_units`, `search_rules`, `get_details` |
| Evaluation | Fixed question set checked against the data tables |
| MCP | Same tools usable from the agent and from Claude Desktop |
