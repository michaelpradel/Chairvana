# Chairvana: Where PC Chairs Reach Nirvana

Chairvana supports chairs of program committees (PCs) of academic conferences. It offers support for finding PC members and managing their information and areas of expertise.

## Web UI

Chairvana now includes a web-based main interface for browsing and editing people in `data/people.jsonl`.

## DBLP Preprocessing (Recommended)

Querying the full `data/dblp.xml.gz` can be slow. Create a filtered JSONL snapshot once, then run queries against it:

```bash
python src/query_dblp.py --preprocess
```

This writes `data/dblp_filtered.jsonl` containing only:

- publication types: `inproceedings`, `article`
- venues (by DBLP key prefix): `conf/icse`, `conf/sigsoft`, `conf/kbse`, `conf/issta`, `conf/oopsla`

After preprocessing, normal `src/query_dblp.py` queries automatically use `data/dblp_filtered.jsonl` when present.

### Run

1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Start the UI:
	```bash
	python src/web_ui.py
	```
3. Open http://127.0.0.1:5000 in your browser.

### Edit People

- Use the search box to filter by name or affiliation.
- Select a person in the left panel.
- Edit fields such as `name` and `affiliation` in the right panel and click **Save changes**.
- Changes are persisted back to `data/people.jsonl` via `src/people.py`.
