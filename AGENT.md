# AGENT.md

## Purpose
This file is for coding agents working on Chairvana.

Goals:
- Preserve data integrity in the people store.
- Keep LLM-driven pipelines robust against malformed/partial outputs.
- Extend features without breaking repository conventions.

## Architecture At A Glance
- Core data access and persistence: [src/util/data_store.py](src/util/data_store.py)
- LLM client, structured parsing, and call logging: [src/util/llm_queries.py](src/util/llm_queries.py)
- Main web app (Flask): [src/web/web_ui.py](src/web/web_ui.py)
- DBLP preprocessing + query engine: [src/util/query_dblp.py](src/util/query_dblp.py)
- Web search utilities: [src/util/web_search.py](src/util/web_search.py)
- Publication sync and summaries: [src/cli/sync_people_with_publications.py](src/cli/sync_people_with_publications.py)
- Expertise embeddings: [src/cli/add_expertise_embeddings.py](src/cli/add_expertise_embeddings.py)
- Expertise gap analysis: [src/cli/expertise_gap_finder.py](src/cli/expertise_gap_finder.py)
- Batch enrichment/cleanup scripts (in [src/cli/](src/cli/)):
  - [auto_complete.py](src/cli/auto_complete.py)
  - [clean_people.py](src/cli/clean_people.py)
  - [auto_clean_and_dedup.py](src/cli/auto_clean_and_dedup.py)
  - [dedup_people.py](src/cli/dedup_people.py)
  - [infer_country.py](src/cli/infer_country.py)
  - [find_gender.py](src/cli/find_gender.py)
  - [find_past_pc_members.py](src/cli/find_past_pc_members.py)
  - [enrich_people_with_publications.py](src/cli/enrich_people_with_publications.py)
  - [export_to_csv.py](src/cli/export_to_csv.py)
- Web UI helpers (in [src/web/](src/web/)):
  - [web_ui_expertise_math.py](src/web/web_ui_expertise_math.py)
  - [web_ui_llm_usage.py](src/web/web_ui_llm_usage.py)
  - [web_ui_regions.py](src/web/web_ui_regions.py)
- Static assets and templates (in [src/web/](src/web/)):
  - [static/](src/web/static/)
  - [templates/](src/web/templates/)
- Prompt templates used by structured LLM calls: [prompts/](prompts/)

## Global Invariants (Do Not Break)
1. All people store reads/writes go through DataStore.
- Use [src/util/data_store.py](src/util/data_store.py).
- Do not directly read/write JSONL store files in [data/people_store/](data/people_store/).
- Do not bypass DataStore with ad-hoc file I/O for people/expertise data.

2. Person identity is name-based.
- People records are keyed by name in DataStore.
- Any rename must go through methods that preserve uniqueness checks.

3. Writes are commit-backed in a local git repo.
- The canonical mutable store is under [data/people_store/](data/people_store/).
- DataStore write operations commit automatically.
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
  - [data/people_store/people.jsonl](data/people_store/people.jsonl)
  - [data/people_store/expertise_embeddings.jsonl](data/people_store/expertise_embeddings.jsonl)
  - [data/people_store/paper_expertise_embeddings.jsonl](data/people_store/paper_expertise_embeddings.jsonl)
- DBLP inputs/cache:
  - [data/dblp_20260318.xml.gz](data/dblp_20260318.xml.gz) (raw dump)
  - [data/dblp.dtd](data/dblp.dtd)
  - [data/people_store/dblp_filtered.jsonl](data/people_store/dblp_filtered.jsonl) (filtered snapshot for queries)
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
  - Load data via DataStore.
  - Validate and normalize external/LLM values before persisting.
  - Support dry-run when feasible for large updates.
  - Print useful progress for long-running batches.
- Batch workflows should fail soft for individual bad records and continue processing.

## Safe Change Playbook For Agents
1. When adding/changing person fields:
- Update normalization/merge behavior in [src/data_store.py](src/data_store.py).
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
- Use DataStore base_commit parameters consistently.
- Assume write operations can reset the people repo state to the selected snapshot before committing new changes.

## Validation Checklist After Edits
- Basic syntax check:
  - python -m py_compile src/*.py
- If routes/templates changed:
  - run [src/web_ui.py](src/web_ui.py) and smoke-test main flows in browser.
- If people store logic changed:
  - verify add/update/delete paths and history behavior via DataStore-backed calls.
- If LLM pipeline changed:
  - run a small dry-run batch and verify malformed/empty fields are handled without crashing.

## Out Of Scope For Quick Fixes
- Do not rewrite the storage layer away from DataStore in routine feature tasks.
- Do not directly mutate git metadata inside [data/people_store/](data/people_store/) except through DataStore methods.
- Do not silently broaden venue scope without updating all dependent analyses and UI assumptions.
