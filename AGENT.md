# AGENT.md

## Purpose
This file is for coding agents working on Chairvana.

Goals:
- Preserve data integrity in the people store.
- Keep LLM-driven pipelines robust against malformed/partial outputs.
- Extend features without breaking repository conventions.

## Architecture At A Glance
- Core data access and persistence: [src/people.py](src/people.py)
- LLM client, structured parsing, and call logging: [src/llm_queries.py](src/llm_queries.py)
- Main web app (Flask): [src/web_ui.py](src/web_ui.py)
- DBLP preprocessing + query engine: [src/query_dblp.py](src/query_dblp.py)
- Publication sync and summaries: [src/sync_people_with_publications.py](src/sync_people_with_publications.py)
- Expertise embeddings: [src/add_expertise_embeddings.py](src/add_expertise_embeddings.py)
- Expertise gap analysis: [src/expertise_gap_finder.py](src/expertise_gap_finder.py)
- Batch enrichment/cleanup scripts:
  - [src/auto_complete.py](src/auto_complete.py)
  - [src/clean_people.py](src/clean_people.py)
  - [src/auto_clean_and_dedup.py](src/auto_clean_and_dedup.py)
  - [src/dedup_people.py](src/dedup_people.py)
  - [src/infer_country.py](src/infer_country.py)
  - [src/find_gender.py](src/find_gender.py)
  - [src/find_past_pc_members.py](src/find_past_pc_members.py)
- Prompt templates used by structured LLM calls: [prompts/](prompts/)

## Global Invariants (Do Not Break)
1. All people store reads/writes go through PeopleStore.
- Use [src/people.py](src/people.py).
- Do not directly read/write JSONL store files in [data/.people_repo/](data/.people_repo/).
- Do not bypass PeopleStore with ad-hoc file I/O for people/expertise data.

2. Person identity is name-based.
- People records are keyed by name in PeopleStore.
- Any rename must go through methods that preserve uniqueness checks.

3. Writes are commit-backed in a local git repo.
- The canonical mutable store is under [data/.people_repo/](data/.people_repo/).
- PeopleStore write operations commit automatically.
- Historical edit mode uses base_commit and may reset the people repo work tree to that snapshot before writing.

4. Data normalization rules must stay consistent.
- flags are normalized to lowercase hashtag tokens (for example: #invite).
- country is expected as ISO 3166-1 alpha-3 uppercase code or missing.
- affiliation is single-valued; legacy affiliations list field is removed.
- publication_summary is replaced as a full object when recomputed (not deeply merged).

5. LLM outputs are untrusted until validated.
- Validate optional fields defensively (homepage/email/country may be missing or malformed).
- Prefer skip/continue behavior in batch pipelines over crashing the entire run.
- Keep robust logging and warning paths.

## Data Layout
- People + embeddings store (git-backed):
  - [data/.people_repo/people.jsonl](data/.people_repo/people.jsonl)
  - [data/.people_repo/expertise_embeddings.jsonl](data/.people_repo/expertise_embeddings.jsonl)
  - [data/.people_repo/paper_expertise_embeddings.jsonl](data/.people_repo/paper_expertise_embeddings.jsonl)
- DBLP inputs/cache:
  - [data/dblp_20260318.xml.gz](data/dblp_20260318.xml.gz) (raw dump)
  - [data/dblp.dtd](data/dblp.dtd)
  - [data/dblp_filtered.jsonl](data/dblp_filtered.jsonl) (filtered snapshot for queries)
  - [data/main_track_venues.json](data/main_track_venues.json)
- Expertise gap index cache:
  - [data/expertise_gap_paper_matrix_index.npz](data/expertise_gap_paper_matrix_index.npz)
- LLM logs:
  - [logs/](logs/)

## DBLP And Venue Decisions
- DBLP preprocessing uses SAX streaming with local DTD resolution in [src/query_dblp.py](src/query_dblp.py).
- Target venues are encoded centrally via TARGET_VENUE_PREFIXES (+ PACMPL OOPSLA handling) in [src/query_dblp.py](src/query_dblp.py).
- Author lookup canonicalizes DBLP disambiguation suffixes (for example trailing numeric IDs).
- If venue scope changes, update venue constants in one place and verify dependent scripts/UI behavior.

## LLM Integration Rules
- Prefer structured responses via parse_structured_response in [src/llm_queries.py](src/llm_queries.py).
- Keep prompt templates in [prompts/](prompts/) and avoid hardcoding large prompts inside Python modules.
- Preserve log_llm_call usage for observability and cost tracking.
- Default model choice is centralized (DEFAULT_RESPONSES_MODEL) in [src/llm_queries.py](src/llm_queries.py).

## Web UI Boundaries
- Flask app lives in [src/web_ui.py](src/web_ui.py).
- Keep business/data logic in reusable modules where possible; use routes as orchestration.
- Maintain compatibility with history browsing semantics (history commit selection + base_commit writes).
- Keep filtering semantics stable unless intentionally changed (text filters, tags, topic search, numeric filters).

## Script Design Expectations
- Scripts should:
  - Load data via PeopleStore.
  - Validate and normalize external/LLM values before persisting.
  - Support dry-run when feasible for large updates.
  - Print useful progress for long-running batches.
- Batch workflows should fail soft for individual bad records and continue processing.

## Safe Change Playbook For Agents
1. When adding/changing person fields:
- Update normalization/merge behavior in [src/people.py](src/people.py).
- Update all writers and key UI/forms as needed.
- Keep backward compatibility for existing JSONL entries.

2. When changing venue/expertise logic:
- Start at [src/query_dblp.py](src/query_dblp.py) constants/utilities.
- Verify downstream behavior in [src/sync_people_with_publications.py](src/sync_people_with_publications.py), [src/add_expertise_embeddings.py](src/add_expertise_embeddings.py), [src/expertise_gap_finder.py](src/expertise_gap_finder.py), and [src/web_ui.py](src/web_ui.py).

3. When changing LLM schemas/prompts:
- Update Pydantic response models together with prompt expectations.
- Keep tolerant parsing/validation for optional fields.
- Ensure logs still capture prompt + structured response.

4. When writing from a historical snapshot:
- Use PeopleStore base_commit parameters consistently.
- Assume write operations can reset the people repo state to the selected snapshot before committing new changes.

## Validation Checklist After Edits
- Basic syntax check:
  - python -m py_compile src/*.py
- If routes/templates changed:
  - run [src/web_ui.py](src/web_ui.py) and smoke-test main flows in browser.
- If people store logic changed:
  - verify add/update/delete paths and history behavior via PeopleStore-backed calls.
- If LLM pipeline changed:
  - run a small dry-run batch and verify malformed/empty fields are handled without crashing.

## Out Of Scope For Quick Fixes
- Do not rewrite the storage layer away from PeopleStore in routine feature tasks.
- Do not directly mutate git metadata inside [data/.people_repo/](data/.people_repo/) except through PeopleStore methods.
- Do not silently broaden venue scope without updating all dependent analyses and UI assumptions.
