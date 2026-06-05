"""Preprocess and query a reduced DBLP JSONL snapshot.

This module supports two operations:
1. One-time preprocessing from ``data/dblp.xml.gz`` to ``data/people_store/dblp_filtered.jsonl``.
2. Querying publications by author from the filtered JSONL only.
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForSysPath

# Add src directory to path to allow imports from util, web, cli folders (when run as script)
_SRC_DIR = _PathForSysPath(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_PROJECT_ROOT = _SRC_DIR.parent

import argparse
import gzip
import json
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Sequence
from xml.sax import handler, make_parser
from xml.sax.handler import feature_external_ges
from xml.sax.xmlreader import AttributesImpl, InputSource

from util.data_store import DataStore
from pydantic import BaseModel

TARGET_PUBLICATION_TAGS = {"article", "inproceedings"}
TARGET_VENUE_PREFIXES = (
    "conf/icse",
    "conf/sigsoft",
    "conf/kbse",
    "conf/issta",
    "conf/oopsla",
)
PACMPL_PREFIX = "journals/pacmpl"

# Years for which we resolve the main research track via LLM.
MAIN_TRACK_YEARS: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
_MAIN_TRACK_CACHE_PATH = _PROJECT_ROOT / "data" / "main_track_venues.json"

# Module-level in-memory cache: venue_prefix -> year_str -> list[venue_string]
_main_track_cache: dict[str, dict[str, list[str]]] | None = None


class MainTrackResponse(BaseModel):
    """Structured LLM response identifying main-track venue strings."""

    main_track_venues: list[str]
    reasoning: str


@dataclass(slots=True)
class Publication:
    pub_type: str
    key: str | None
    title: str | None
    year: int | None
    venue: str | None
    issue: str | None
    authors: list[str]


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


class LocalDtdResolver(handler.EntityResolver):
    def __init__(self, dtd_path: Path) -> None:
        self.dtd_path = dtd_path

    def resolveEntity(self, publicId: str | None, systemId: str) -> str:
        if systemId and systemId.endswith("dblp.dtd"):
            return str(self.dtd_path)
        return systemId


def build_publication(data: dict[str, object]) -> Publication:
    venue = None
    for key in ("journal", "booktitle", "school", "publisher"):
        value = data.get(key)
        if isinstance(value, str) and value:
            venue = value
            break

    key_value = data.get("key")
    title_value = data.get("title")
    year_value = data.get("year")
    issue_value = data.get("number")
    authors_value = data.get("authors")

    return Publication(
        pub_type=str(data["pub_type"]),
        key=key_value if isinstance(key_value, str) else None,
        title=title_value if isinstance(title_value, str) else None,
        year=year_value if isinstance(year_value, int) else None,
        venue=venue,
        issue=issue_value if isinstance(issue_value, str) else None,
        authors=[author for author in authors_value if isinstance(author, str)]
        if isinstance(authors_value, list)
        else [],
    )


def _is_oopsla_pacmpl_publication(publication: Publication) -> bool:
    """Return True if publication is a PACMPL OOPSLA issue paper."""
    if not publication.key or not publication.issue:
        return False
    return publication.key.lower().startswith(f"{PACMPL_PREFIX}/") and publication.issue.upper().startswith("OOPSLA")


def _matches_target_venue(publication: Publication) -> bool:
    """Return True if publication belongs to one of the selected venues."""
    if publication.key:
        normalized_key = publication.key.lower()
        for prefix in TARGET_VENUE_PREFIXES:
            if normalized_key == prefix or normalized_key.startswith(f"{prefix}/"):
                return True

    if _is_oopsla_pacmpl_publication(publication):
        return True

    # Optional fallback to cover preprocessed records without DBLP keys.
    if publication.venue:
        normalized_venue = publication.venue.lower()
        return any(normalized_venue == prefix for prefix in TARGET_VENUE_PREFIXES)

    return False


class DblpFilterHandler(handler.ContentHandler):
    """Stream DBLP XML and emit filtered publications as JSONL."""

    text_fields = {"author", "title", "year", "journal", "booktitle", "school", "publisher", "number"}

    def __init__(self, output_file) -> None:
        super().__init__()
        self.output_file = output_file
        self.current_pub: dict[str, object] | None = None
        self.current_field: str | None = None
        self.current_text: list[str] = []
        self.total_publications = 0
        self.kept_publications = 0

    def startElement(self, name: str, attrs: AttributesImpl) -> None:
        if self.current_pub is None and name in TARGET_PUBLICATION_TAGS:
            key_value = attrs.get("key")
            self.current_pub = {
                "pub_type": name,
                "key": key_value,
                "title": None,
                "year": None,
                "journal": None,
                "booktitle": None,
                "school": None,
                "publisher": None,
                "number": None,
                "authors": [],
            }
            return

        if self.current_pub is not None and name in self.text_fields:
            self.current_field = name
            self.current_text = []

    def characters(self, content: str) -> None:
        if self.current_field is not None:
            self.current_text.append(content)

    def endElement(self, name: str) -> None:
        if self.current_pub is not None and self.current_field == name:
            value = normalize_whitespace("".join(self.current_text))
            if value:
                if name == "author":
                    authors = self.current_pub["authors"]
                    if isinstance(authors, list):
                        authors.append(value)
                elif name == "year":
                    self.current_pub["year"] = int(value) if value.isdigit() else None
                else:
                    self.current_pub[name] = value

            self.current_field = None
            self.current_text = []

        if self.current_pub is not None and name in TARGET_PUBLICATION_TAGS:
            self.total_publications += 1
            publication = build_publication(self.current_pub)
            if _matches_target_venue(publication):
                self.output_file.write(json.dumps(asdict(publication), ensure_ascii=False) + "\n")
                self.kept_publications += 1
            self.current_pub = None


def preprocess_dblp_to_jsonl(
    xml_gz_path: Path,
    dtd_path: Path,
    output_jsonl_path: Path,
    data_store: DataStore | None = None,
) -> tuple[int, int, float]:
    """Create a filtered JSONL snapshot from the DBLP XML dump.

    Keeps only:
    - publication types: article, inproceedings
    - venues keyed under conf/icse, conf/sigsoft, conf/kbse, conf/issta, conf/oopsla
    """
    if not xml_gz_path.exists():
        raise FileNotFoundError(f"Missing DBLP dump: {xml_gz_path}")
    if not dtd_path.exists():
        raise FileNotFoundError(f"Missing DTD file: {dtd_path}")

    started = time.perf_counter()

    sax_parser = make_parser()
    sax_parser.setFeature(feature_external_ges, True)
    sax_parser.setEntityResolver(LocalDtdResolver(dtd_path))

    use_store_commit = (
        data_store is not None
        and output_jsonl_path.resolve() == data_store.dblp_filtered_path.resolve()
    )

    if use_store_commit:
        assert data_store is not None
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp_output:
            tmp_output_path = Path(tmp_output.name)
            content_handler = DblpFilterHandler(tmp_output)
            sax_parser.setContentHandler(content_handler)

            with gzip.open(xml_gz_path, "rb") as xml_file:
                source = InputSource()
                source.setByteStream(xml_file)
                source.setSystemId(str(xml_gz_path))
                sax_parser.parse(source)

        try:
            data_store.replace_dblp_filtered(tmp_output_path)
        finally:
            tmp_output_path.unlink(missing_ok=True)
    else:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open("w", encoding="utf-8") as output_file:
            content_handler = DblpFilterHandler(output_file)
            sax_parser.setContentHandler(content_handler)

            with gzip.open(xml_gz_path, "rb") as xml_file:
                source = InputSource()
                source.setByteStream(xml_file)
                source.setSystemId(str(xml_gz_path))
                sax_parser.parse(source)

    elapsed = time.perf_counter() - started
    return content_handler.total_publications, content_handler.kept_publications, elapsed


class DblpQueryEngine:
    """Query publications by author from filtered DBLP JSONL."""

    def __init__(
        self,
        filtered_jsonl_path: Path | str | None = None,
        preload_index: bool = False,
    ) -> None:
        base_dir = _PROJECT_ROOT
        self.filtered_jsonl_path = (
            Path(filtered_jsonl_path)
            if filtered_jsonl_path is not None
            else base_dir / "data" / "people_store" / "dblp_filtered.jsonl"
        )
        if not self.filtered_jsonl_path.exists():
            raise FileNotFoundError(f"Missing filtered DBLP JSONL: {self.filtered_jsonl_path}")

        self._author_index: dict[str, list[Publication]] | None = None
        self._author_canonical_index: dict[str, list[Publication]] | None = None
        self.index_build_seconds: float | None = None
        if preload_index:
            self.build_author_index()

    @staticmethod
    def _canonical_author_name(author_name: str) -> str:
        normalized = normalize_whitespace(author_name)
        # DBLP may append numeric disambiguation suffixes (e.g., "0001", "0114").
        return re.sub(r"\s\d+$", "", normalized)

    @staticmethod
    def _publication_identity(publication: Publication) -> tuple[Any, ...]:
        if publication.key:
            return ("key", publication.key)
        return (
            "fallback",
            publication.title,
            publication.year,
            publication.venue,
            tuple(publication.authors),
        )

    @classmethod
    def _dedupe_publications(cls, publications: list[Publication]) -> list[Publication]:
        deduped: list[Publication] = []
        seen: set[tuple[Any, ...]] = set()
        for publication in publications:
            identity = cls._publication_identity(publication)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(publication)
        return deduped

    def build_author_index(self, force: bool = False) -> dict[str, list[Publication]]:
        """Build a full in-memory author->publications index from filtered JSONL."""
        if self._author_index is not None and not force:
            return self._author_index

        started = time.perf_counter()
        matches_by_author: DefaultDict[str, list[Publication]] = defaultdict(list)
        canonical_matches_by_author: DefaultDict[str, list[Publication]] = defaultdict(list)

        with self.filtered_jsonl_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                publication = Publication(
                    pub_type=str(row.get("pub_type") or ""),
                    key=row.get("key") if isinstance(row.get("key"), str) else None,
                    title=row.get("title") if isinstance(row.get("title"), str) else None,
                    year=row.get("year") if isinstance(row.get("year"), int) else None,
                    venue=row.get("venue") if isinstance(row.get("venue"), str) else None,
                    issue=row.get("issue") if isinstance(row.get("issue"), str) else None,
                    authors=[a for a in row.get("authors", []) if isinstance(a, str)]
                    if isinstance(row.get("authors"), list)
                    else [],
                )
                for author in publication.authors:
                    normalized_author = normalize_whitespace(author)
                    matches_by_author[normalized_author].append(publication)
                    canonical_matches_by_author[self._canonical_author_name(normalized_author)].append(publication)

        self._author_index = dict(matches_by_author)
        self._author_canonical_index = {
            author: self._dedupe_publications(publications)
            for author, publications in canonical_matches_by_author.items()
        }
        self.index_build_seconds = time.perf_counter() - started
        return self._author_index

    def query_author(self, author_name: str, max_results: int | None = None) -> list[Publication]:
        author_name = normalize_whitespace(author_name)

        if self._author_index is None:
            self.build_author_index()

        publications: list[Publication] = []
        if self._author_index is not None:
            publications.extend(self._author_index.get(author_name, []))

        canonical_name = self._canonical_author_name(author_name)
        if self._author_canonical_index is not None:
            publications.extend(self._author_canonical_index.get(canonical_name, []))

        publications = self._dedupe_publications(publications)
        return publications[:max_results] if max_results is not None else publications

    def query_authors(
        self,
        author_names: Sequence[str],
        max_results_per_author: int | None = None,
    ) -> dict[str, list[Publication]]:
        normalized_authors = [normalize_whitespace(author) for author in author_names]

        if self._author_index is None:
            self.build_author_index()

        result: dict[str, list[Publication]] = {}
        for author in normalized_authors:
            publications: list[Publication] = []
            if self._author_index is not None:
                publications.extend(self._author_index.get(author, []))

            canonical_name = self._canonical_author_name(author)
            if self._author_canonical_index is not None:
                publications.extend(self._author_canonical_index.get(canonical_name, []))

            publications = self._dedupe_publications(publications)
            if max_results_per_author is not None:
                publications = publications[:max_results_per_author]
            result[author] = publications
        return result
    def query_authors_by_venue(
        self,
        venue_prefix: str,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> dict[str, list[Publication]]:
        """Find all authors who published in a specific venue within a year range."""
        if self._author_index is None:
            self.build_author_index()

        result: dict[str, list[Publication]] = {}
        author_index = self._author_index
        if author_index is None:
            return result

        normalized_venue = venue_prefix.lower()

        for author, publications in author_index.items():
            matching = [
                pub
                for pub in publications
                if self._publication_matches_venue(pub, normalized_venue)
                and self._year_in_range(pub.year, min_year, max_year)
            ]
            if matching:
                result[author] = matching

        return result

    def query_prolific_authors_in_target_venues(
        self,
        min_publications: int = 5,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> dict[str, list[Publication]]:
        """Find authors with multiple publications in target venues within a year range.

        An author is included if they have at least min_publications in any of the
        TARGET_VENUE_PREFIXES within the specified year range.
        """
        if self._author_index is None:
            self.build_author_index()

        result: dict[str, list[Publication]] = {}
        canonical_author_index = self._author_canonical_index
        if canonical_author_index is None:
            return result

        for author, publications in canonical_author_index.items():
            matching = [
                pub
                for pub in publications
                if any(self._publication_matches_venue(pub, prefix.lower()) for prefix in TARGET_VENUE_PREFIXES)
                and self._year_in_range(pub.year, min_year, max_year)
            ]
            matching = self._dedupe_publications(matching)
            if len(matching) >= min_publications:
                result[author] = matching

        return result

    def get_distinct_venue_strings(self, venue_prefix: str, year: int) -> list[str]:
        """Return sorted distinct venue (booktitle) strings for a venue prefix and year.

        Deduplicates by publication key so co-authored papers are counted once.
        """
        if self._author_index is None:
            self.build_author_index()

        norm_prefix = venue_prefix.lower()
        seen_keys: set[str] = set()
        venue_strings: set[str] = set()

        for pubs in (self._author_index or {}).values():
            for pub in pubs:
                if pub.key is None or pub.key in seen_keys:
                    continue
                if pub.year == year and self._publication_matches_venue(pub, norm_prefix):
                    seen_keys.add(pub.key)
                    if pub.venue:
                        venue_strings.add(pub.venue)

        return sorted(venue_strings)

    @staticmethod
    def _publication_matches_venue(publication: Publication, normalized_venue_prefix: str) -> bool:
        """Check if a publication matches a venue prefix."""
        if normalized_venue_prefix == "conf/oopsla" and _is_oopsla_pacmpl_publication(publication):
            return True

        if publication.key:
            normalized_key = publication.key.lower()
            return normalized_key == normalized_venue_prefix or normalized_key.startswith(
                f"{normalized_venue_prefix}/"
            )
        return False

    @staticmethod
    def _year_in_range(pub_year: int | None, min_year: int | None, max_year: int | None) -> bool:
        """Check if a publication year falls within the specified range."""
        if pub_year is None:
            return min_year is None and max_year is None
        if min_year is not None and pub_year < min_year:
            return False
        if max_year is not None and pub_year > max_year:
            return False
        return True

def get_publications_by_author(
    author_name: str,
    filtered_jsonl_path: Path | str | None = None,
    max_results: int | None = None,
) -> list[Publication]:
    """Return all publications that include the given author name."""
    engine = DblpQueryEngine(
        filtered_jsonl_path=filtered_jsonl_path,
        preload_index=False,
    )
    return engine.query_author(author_name, max_results=max_results)


def get_authors_by_venue(
    venue_prefix: str,
    min_year: int | None = None,
    max_year: int | None = None,
    filtered_jsonl_path: Path | str | None = None,
) -> dict[str, list[Publication]]:
    """Find all authors who published in a specific venue within a year range."""
    engine = DblpQueryEngine(
        filtered_jsonl_path=filtered_jsonl_path,
        preload_index=False,
    )
    return engine.query_authors_by_venue(venue_prefix, min_year, max_year)


def get_prolific_authors(
    min_publications: int = 5,
    min_year: int | None = None,
    max_year: int | None = None,
    filtered_jsonl_path: Path | str | None = None,
) -> dict[str, list[Publication]]:
    """Find authors with multiple publications in target venues within a year range.

    Searches across all TARGET_VENUE_PREFIXES for authors with at least
    min_publications in the specified year range.
    """
    engine = DblpQueryEngine(
        filtered_jsonl_path=filtered_jsonl_path,
        preload_index=False,
    )
    return engine.query_prolific_authors_in_target_venues(min_publications, min_year, max_year)


def _get_main_track_cache() -> dict[str, dict[str, list[str]]]:
    """Return the in-memory main-track cache, loading from disk on first call."""
    global _main_track_cache
    if _main_track_cache is None:
        if _MAIN_TRACK_CACHE_PATH.exists():
            _main_track_cache = json.loads(_MAIN_TRACK_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _main_track_cache = {}
    cache = _main_track_cache
    if cache is None:
        raise RuntimeError("Main-track cache failed to initialize")
    return cache


def _persist_main_track_cache() -> None:
    """Write the current in-memory cache to disk."""
    cache = _get_main_track_cache()
    _MAIN_TRACK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MAIN_TRACK_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _resolve_main_track_via_llm(
    venue_prefix: str,
    year: int,
    engine: DblpQueryEngine,
) -> list[str] | None:
    """Ask the LLM which venue strings belong to the main research track.

    Returns the accepted venue strings, or None when the filtered JSONL
    contains no publications for this (venue_prefix, year) pair.
    Caches the result on disk before returning.
    """
    from util.llm_queries import get_openai_client, parse_structured_response  # noqa: PLC0415

    venue_strings = engine.get_distinct_venue_strings(venue_prefix, year)
    if not venue_strings:
        return None

    prompt_path = _PROJECT_ROOT / "prompts" / "identify_main_research_track.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    venue_short = venue_prefix.split("/")[-1].upper()
    venue_list_str = "\n".join(f"- {v}" for v in venue_strings)
    input_text = prompt_template.format(
        venue_prefix=venue_prefix,
        venue_short_name=venue_short,
        year=year,
        venue_strings=venue_list_str,
    )

    response = parse_structured_response(
        input_text=input_text,
        response_model=MainTrackResponse,
        description="identify main research track",
        client=get_openai_client(),
    )

    cache = _get_main_track_cache()
    cache.setdefault(venue_prefix, {})[str(year)] = response.main_track_venues
    _persist_main_track_cache()

    return response.main_track_venues


def get_main_track_venues(
    venue_prefix: str,
    year: int,
    engine: DblpQueryEngine,
) -> list[str] | None:
    """Return accepted venue strings for the main research track of a venue/year pair.

    Checks the on-disk cache first.  If no entry exists, queries the LLM,
    stores the result, and returns it.  Returns None when no publications for
    this (venue_prefix, year) are found in the filtered JSONL (caller should
    fall back to prefix-based matching).
    """
    if year not in MAIN_TRACK_YEARS:
        return None

    cache = _get_main_track_cache()
    cached = cache.get(venue_prefix, {}).get(str(year))
    if cached is not None:
        return cached

    return _resolve_main_track_via_llm(venue_prefix, year, engine)


def is_main_track_publication(
    pub: Publication,
    venue_prefix: str,
    engine: DblpQueryEngine,
) -> bool:
    """Return True if pub is in target venue AND in the main research track.

    Falls back to prefix-only matching when cache data is unavailable (no
    publications found for that venue/year, or year outside MAIN_TRACK_YEARS).
    """
    if not _publication_in_target_venue(pub, venue_prefix):
        return False

    # OOPSLA papers in PACMPL (issue OOPSLA*) should count as main-track by definition.
    if venue_prefix.lower() == "conf/oopsla" and _is_oopsla_pacmpl_publication(pub):
        return True

    if pub.year is None:
        return False

    main_venues = get_main_track_venues(venue_prefix, pub.year, engine)

    # No data for this venue+year — conservatively treat pub as in-scope.
    if main_venues is None:
        return True

    # No venue string on the pub — conservatively include it.
    if pub.venue is None:
        return True

    return pub.venue in main_venues


def get_target_publications_for_author(
    person_name: str,
    engine: DblpQueryEngine,
    current_year: int | None = None,
    max_years_back: int = 5,
) -> list[Publication]:
    """Return target-venue main-track publications for a person in the year window."""
    if current_year is None:
        current_year = datetime.now().year

    min_year = current_year - max_years_back
    publications = engine.query_author(person_name)

    return [
        pub
        for pub in publications
        if pub.year is not None
        and pub.year >= min_year
        and pub.year <= current_year
        and any(is_main_track_publication(pub, prefix, engine) for prefix in TARGET_VENUE_PREFIXES)
    ]


def create_publication_summary(
    person_name: str,
    current_year: int | None = None,
    max_years_back: int = 5,
    filtered_jsonl_path: Path | str | None = None,
    engine: DblpQueryEngine | None = None,
) -> dict[str, Any] | None:
    """Create a publication summary for a person from target venues.

    Summarizes papers by this person in target venues from the last N years,
    with counts by venue and year range.

    Args:
        person_name: The person's name to search for in DBLP
        current_year: The year to use as reference (default: current year)
        max_years_back: How many years back to include (default: 5)
        filtered_jsonl_path: Path to filtered DBLP JSONL (default: data/people_store/dblp_filtered.jsonl)
        engine: Existing query engine to reuse instead of rebuilding the index

    Returns:
        A dict with venue and year breakdown, or None if no papers found.
        Structure: {
            "total": <int>,
            "by_venue": {
                "icse": <int>,
                "oopsla": <int>,
                ...
            },
            "year_range": [<min_year>, <max_year>]
        }
    """
    if current_year is None:
        current_year = datetime.now().year

    if engine is None:
        engine = DblpQueryEngine(filtered_jsonl_path=filtered_jsonl_path, preload_index=False)

    target_publications = get_target_publications_for_author(
        person_name,
        engine,
        current_year=current_year,
        max_years_back=max_years_back,
    )

    if not target_publications:
        return None

    # Count by venue
    venue_counts: dict[str, int] = defaultdict(int)
    for pub in target_publications:
        venue_key = _extract_venue_key(pub)
        if venue_key:
            venue_counts[venue_key] += 1

    # Extract year range
    years = [pub.year for pub in target_publications if pub.year is not None]
    year_range = [min(years), max(years)] if years else None

    return {
        "total": len(target_publications),
        "by_venue": dict(venue_counts),
        "year_range": year_range,
    }


def _publication_in_target_venue(pub: Publication, venue_prefix: str) -> bool:
    """Check if a publication is in a target venue."""
    if venue_prefix.lower() == "conf/oopsla" and _is_oopsla_pacmpl_publication(pub):
        return True

    if pub.key:
        normalized_key = pub.key.lower()
        normalized_prefix = venue_prefix.lower()
        return normalized_key == normalized_prefix or normalized_key.startswith(f"{normalized_prefix}/")
    return False


def _extract_venue_key(pub: Publication) -> str | None:
    """Extract a clean venue key from a publication.

    Returns the first path component of the DBLP key, e.g. 'icse' from 'conf/icse/2024'.
    """
    if not pub.key:
        return None

    # DBLP keys like "conf/icse/2024" -> extract "icse"
    parts = pub.key.lower().split("/")
    if len(parts) >= 2 and parts[0] == "conf":
        return parts[1]

    if _is_oopsla_pacmpl_publication(pub):
        return "oopsla"

    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query DBLP publications by author")
    parser.add_argument(
        "author",
        nargs="?",
        help='Author name to search, e.g., "Michael Pradel"',
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Optional path to DBLP XML gzip file (default: data/dblp.xml.gz)",
    )
    parser.add_argument(
        "--dtd",
        type=Path,
        default=None,
        help="Optional path to DTD file (default: data/dblp.dtd)",
    )
    parser.add_argument(
        "--filtered-jsonl",
        type=Path,
        default=None,
        help="Path to filtered DBLP JSONL cache (default: data/people_store/dblp_filtered.jsonl)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run one-time preprocessing from DBLP XML to filtered JSONL and exit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of matching publications to print (default: 10)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum number of matches to return per author",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help="Query authors who published in a specific venue, e.g., 'conf/icse'",
    )
    parser.add_argument(
        "--prolific",
        action="store_true",
        help="Find authors with multiple publications in target venues (use with --year-start and --year-end)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Start year for filtering publications (inclusive)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="End year for filtering publications (inclusive)",
    )
    parser.add_argument(
        "--min-pubs",
        type=int,
        default=5,
        help="Minimum number of publications for prolific authors query (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    base_dir = _PROJECT_ROOT
    xml_path = args.xml if args.xml is not None else base_dir / "data" / "dblp.xml.gz"
    dtd_path = args.dtd if args.dtd is not None else base_dir / "data" / "dblp.dtd"
    filtered_jsonl_path = (
        args.filtered_jsonl
        if args.filtered_jsonl is not None
        else base_dir / "data" / "people_store" / "dblp_filtered.jsonl"
    )

    if args.preprocess:
        data_store = DataStore()
        store_for_preprocess = (
            data_store if filtered_jsonl_path.resolve() == data_store.dblp_filtered_path.resolve() else None
        )
        total, kept, elapsed = preprocess_dblp_to_jsonl(
            xml_path,
            dtd_path,
            filtered_jsonl_path,
            data_store=store_for_preprocess,
        )
        print(f"Preprocessed DBLP in {elapsed:.2f}s")
        print(f"Scanned target publication tags: {total}")
        print(f"Kept filtered publications: {kept}")
        print(f"Wrote filtered JSONL: {filtered_jsonl_path}")
        print(
            "Reminder: DBLP data changed. Recompute the expertise-gap index with "
            "`python src/expertise_gap_finder.py --recompute-paper-embeddings --recompute-similarity-index`."
        )
        return 0

    if args.prolific:
        authors_by_pubs = get_prolific_authors(
            min_publications=args.min_pubs,
            min_year=args.year_start,
            max_year=args.year_end,
            filtered_jsonl_path=filtered_jsonl_path,
        )
        year_range = f"{args.year_start or 'any'}-{args.year_end or 'any'}"
        print(f"Found {len(authors_by_pubs)} authors with {args.min_pubs}+ publications in target venues ({year_range})")
        for author in sorted(authors_by_pubs.keys()):
            pubs = authors_by_pubs[author]
            print(f"{author}: {len(pubs)} publications")
            for pub in pubs[: args.limit]:
                print(f"  - ({pub.year}) {pub.title}")
        return 0

    if args.venue:
        authors_by_venue = get_authors_by_venue(
            venue_prefix=args.venue,
            min_year=args.year_start,
            max_year=args.year_end,
            filtered_jsonl_path=filtered_jsonl_path,
        )
        year_range = f"{args.year_start or 'any'}-{args.year_end or 'any'}"
        print(f"Found {len(authors_by_venue)} authors who published in {args.venue} ({year_range})")
        for author in sorted(authors_by_venue.keys()):
            pubs = authors_by_venue[author]
            print(f"{author}: {len(pubs)} publications")
            for pub in pubs[: args.limit]:
                print(f"  - ({pub.year}) {pub.title}")
        return 0

    if not args.author:
        raise ValueError("author is required unless --preprocess, --venue, or --prolific is used")

    engine = DblpQueryEngine(
        filtered_jsonl_path=filtered_jsonl_path,
        preload_index=True,
    )
    publications = engine.query_author(args.author, max_results=args.max_results)

    print(f"Found {len(publications)} publications for: {args.author}")
    for publication in publications[: args.limit]:
        print(json.dumps(asdict(publication), ensure_ascii=False))

    if engine.index_build_seconds is not None:
        print(f"Built in-memory author index in {engine.index_build_seconds:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
