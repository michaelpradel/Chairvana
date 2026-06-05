# Chairvana: Where PC Chairs Reach Nirvana

Chairvana supports chairs of program committees (PCs) of academic conferences. It offers support for finding PC members and managing their information and areas of expertise.

## Installation

1. Clone this repository and enter the main directory:
	```
	git clone https://github.com/michaelpradel/Chairvana.git
	cd Chairvana
	```
2. Install dependencies:
	```
	pip install -r requirements.txt
	```

3. Optionally, to use LLM-based features, such as auto-completing information (e.g., email, personal website, gender or potential PC members) and features based on semantic embeddings of expertise (e.g., to find reseachers with specific expertise), add an OpenAI API key into file `.openai_token`.

## Setting Up the Data Repository

The heart of Chairvana is a data repository, which contains information about researchers, such as their name, affiliation, email, personal website, and areas of expertise, as well as information about papers. The data store is implemented as a Git repository in `data/people_store`. 

### Starting from an Existing Data Repository


### Initializing the People Store from Scratch



To set up the people store:

2. Clone a data repository (if you have one), e.g.,:
	```
	git clone https://github.com/michaelpradel/Chairvana-fse2027-data.git data/people_store
	```


## Web UI

1. Start the UI:
	```bash
	python src/web/web_ui.py
	```

2. Open http://127.0.0.1:5000 in your browser.

## Command-Line Tools

(The following is highly incomplete. Ignore for now.)

### Tag OOPSLA-Only People

Find people whose `publication_summary` contains only OOPSLA publications and add `#oopslaonly`:

```bash
python src/cli/tag_oopsla_only.py
```

Preview changes without writing:

```bash
python src/cli/tag_oopsla_only.py --dry-run
```

### DBLP Preprocessing

Querying the full `data/dblp.xml.gz` can be slow. Create a filtered JSONL snapshot once, then run queries against it:

```bash
python src/util/query_dblp.py --preprocess
```

This writes `data/people_store/dblp_filtered.jsonl` containing only:

- publication types: `inproceedings`, `article`
- venues (by DBLP key prefix): `conf/icse`, `conf/sigsoft`, `conf/kbse`, `conf/issta`, `conf/oopsla`

After preprocessing, normal `src/query_dblp.py` queries automatically use `data/people_store/dblp_filtered.jsonl` when present.
