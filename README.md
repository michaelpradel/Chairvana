# Chairvana: Where PC Chairs Reach Nirvana

Chairvana supports program committee chairs of academic conferences. It offers support for finding PC members and managing their information and areas of expertise.

![screenshot](figs/screenshot.png)

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

### Option 1: Starting from an Existing Data Repository

The easiest option is to start from an existing data repository, such as [this software engineering-focused data store that was created for gathering the FSE'27 PC](https://github.com/michaelpradel/Chairvana-data-SE).
To use it, simply clone it into `data/people_store`:

```
git clone https://github.com/michaelpradel/Chairvana-data-SE.git data/people_store
```

### Option 2: Initializing the People Store from Scratch

Alternatively, you can create a new data store from scratch. This is recommended for using Chairvana in a different research community or to ensure that all data is freshly collected from the web.
To initialize the data store from scratch, create an empty Git repository and clone it into `data/people_store`.
The repository can be local or remote (e.g., on GitHub). For example, to use a local repository, run:

```
mkdir data/people_store
cd data/people_store
git init
cd ../..
```

To fill the data store with information about researchers, use the web UI or command-line tools, as described below.

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
