"""Preprocess and query a reduced DBLP JSONL snapshot.

This module supports two operations:
1. One-time preprocessing from ``data/dblp.xml.gz`` to ``data/dblp_filtered.jsonl``.
2. Querying publications by author from the filtered JSONL only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Sequence
from xml.sax import handler, make_parser
from xml.sax.handler import feature_external_ges
from xml.sax.xmlreader import AttributesImpl, InputSource

TARGET_PUBLICATION_TAGS = {"article", "inproceedings"}
TARGET_VENUE_PREFIXES = (
    "conf/icse",
    "conf/sigsoft",
    "conf/kbse",
    "conf/issta",
    "conf/oopsla",
)


@dataclass(slots=True)
class Publication:
    pub_type: str
    key: str | None
    title: str | None
    year: int | None
    venue: str | None
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
    authors_value = data.get("authors")

    return Publication(
        pub_type=str(data["pub_type"]),
        key=key_value if isinstance(key_value, str) else None,
        title=title_value if isinstance(title_value, str) else None,
        year=year_value if isinstance(year_value, int) else None,
        venue=venue,
        authors=[author for author in authors_value if isinstance(author, str)]
        if isinstance(authors_value, list)
        else [],
    )


def _matches_target_venue(publication: Publication) -> bool:
    """Return True if publication belongs to one of the selected venues."""
    if publication.key:
        normalized_key = publication.key.lower()
        for prefix in TARGET_VENUE_PREFIXES:
            if normalized_key == prefix or normalized_key.startswith(f"{prefix}/"):
                return True

    # Optional fallback to cover preprocessed records without DBLP keys.
    if publication.venue:
        normalized_venue = publication.venue.lower()
        return any(normalized_venue == prefix for prefix in TARGET_VENUE_PREFIXES)

    return False


class DblpFilterHandler(handler.ContentHandler):
    """Stream DBLP XML and emit filtered publications as JSONL."""

    text_fields = {"author", "title", "year", "journal", "booktitle", "school", "publisher"}

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

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    sax_parser = make_parser()
    sax_parser.setFeature(feature_external_ges, True)
    sax_parser.setEntityResolver(LocalDtdResolver(dtd_path))

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
        base_dir = Path(__file__).resolve().parent.parent
        self.filtered_jsonl_path = (
            Path(filtered_jsonl_path)
            if filtered_jsonl_path is not None
            else base_dir / "data" / "dblp_filtered.jsonl"
        )
        if not self.filtered_jsonl_path.exists():
            raise FileNotFoundError(f"Missing filtered DBLP JSONL: {self.filtered_jsonl_path}")

        self._author_index: dict[str, list[Publication]] | None = None
        self.index_build_seconds: float | None = None
        if preload_index:
            self.build_author_index()

    def build_author_index(self, force: bool = False) -> dict[str, list[Publication]]:
        """Build a full in-memory author->publications index from filtered JSONL."""
        if self._author_index is not None and not force:
            return self._author_index

        started = time.perf_counter()
        matches_by_author: DefaultDict[str, list[Publication]] = defaultdict(list)

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
                    authors=[a for a in row.get("authors", []) if isinstance(a, str)]
                    if isinstance(row.get("authors"), list)
                    else [],
                )
                for author in publication.authors:
                    matches_by_author[author].append(publication)

        self._author_index = dict(matches_by_author)
        self.index_build_seconds = time.perf_counter() - started
        return self._author_index

    def query_author(self, author_name: str, max_results: int | None = None) -> list[Publication]:
        author_name = normalize_whitespace(author_name)

        if self._author_index is None:
            self.build_author_index()

        publications = list(self._author_index.get(author_name, [])) if self._author_index else []
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
            publications = list(self._author_index.get(author, [])) if self._author_index else []
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
        author_index = self._author_index
        if author_index is None:
            return result

        for author, publications in author_index.items():
            matching = [
                pub
                for pub in publications
                if any(self._publication_matches_venue(pub, prefix.lower()) for prefix in TARGET_VENUE_PREFIXES)
                and self._year_in_range(pub.year, min_year, max_year)
            ]
            if len(matching) >= min_publications:
                result[author] = matching

        return result

    @staticmethod
    def _publication_matches_venue(publication: Publication, normalized_venue_prefix: str) -> bool:
        """Check if a publication matches a venue prefix."""
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
        help="Path to filtered DBLP JSONL cache (default: data/dblp_filtered.jsonl)",
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

    base_dir = Path(__file__).resolve().parent.parent
    xml_path = args.xml if args.xml is not None else base_dir / "data" / "dblp.xml.gz"
    dtd_path = args.dtd if args.dtd is not None else base_dir / "data" / "dblp.dtd"
    filtered_jsonl_path = (
        args.filtered_jsonl if args.filtered_jsonl is not None else base_dir / "data" / "dblp_filtered.jsonl"
    )

    if args.preprocess:
        total, kept, elapsed = preprocess_dblp_to_jsonl(xml_path, dtd_path, filtered_jsonl_path)
        print(f"Preprocessed DBLP in {elapsed:.2f}s")
        print(f"Scanned target publication tags: {total}")
        print(f"Kept filtered publications: {kept}")
        print(f"Wrote filtered JSONL: {filtered_jsonl_path}")
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
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc