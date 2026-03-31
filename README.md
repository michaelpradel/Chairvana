# Chairvana: Where PC Chairs Reach Nirvana

Chairvana supports chairs of program committees (PCs) of academic conferences. It offers support for finding PC members and managing their information and areas of expertise.

## Installation

1. Clone this repository and enter the main directory:
	```
	git clone https://github.com/michaelpradel/Chairvana.git
	cd Chairvana
	```

2. Clone a data repository (if you have one), e.g.,:
	```
	git clone https://github.com/michaelpradel/Chairvana-fse2027-data.git data/.people_repo
	```

3. Install dependencies:
	```
	pip install -r requirements.txt
	```

## Web UI

1. Start the UI:
	```bash
	python src/web_ui.py
	```
2. Open http://127.0.0.1:5000 in your browser.

## Command-Line Tools

(The following is highly incomplete. Ignore for now.)

### DBLP Preprocessing

Querying the full `data/dblp.xml.gz` can be slow. Create a filtered JSONL snapshot once, then run queries against it:

```bash
python src/query_dblp.py --preprocess
```

This writes `data/.people_repo/dblp_filtered.jsonl` containing only:

- publication types: `inproceedings`, `article`
- venues (by DBLP key prefix): `conf/icse`, `conf/sigsoft`, `conf/kbse`, `conf/issta`, `conf/oopsla`

After preprocessing, normal `src/query_dblp.py` queries automatically use `data/.people_repo/dblp_filtered.jsonl` when present.

