"""Query DBLP XML dump for publications by author.

This script reads ``data/dblp.xml.gz`` and configures XML parsing with the
repository's ``data/dblp.dtd`` file. It provides a reusable function to return
all publications for a specific author and a small CLI for interactive use.
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


class StopParsing(Exception):
    """Raised to stop SAX parsing early when enough matches are found."""


PUBLICATION_TAGS = {
    "article",
    "inproceedings",
    "proceedings",
    "book",
    "incollection",
    "phdthesis",
    "mastersthesis",
    "www",
}


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


class DblpAuthorHandler(handler.ContentHandler):
    text_fields = {"author", "title", "year", "journal", "booktitle", "school", "publisher"}

    def __init__(
        self,
        target_authors: set[str] | None = None,
        max_results_per_author: int | None = None,
        index_all_authors: bool = False,
    ) -> None:
        super().__init__()
        self.target_authors = {normalize_whitespace(author) for author in target_authors or set()}
        self.max_results_per_author = max_results_per_author
        self.index_all_authors = index_all_authors
        self.matches_by_author: DefaultDict[str, list[Publication]] = defaultdict(list)
        self.current_pub: dict[str, object] | None = None
        self.current_field: str | None = None
        self.current_text: list[str] = []
        self.total_matches = 0

    def startElement(self, name: str, attrs: AttributesImpl) -> None:
        if self.current_pub is None and name in PUBLICATION_TAGS:
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

        if self.current_pub is not None and name in PUBLICATION_TAGS:
            publication = self._build_publication(self.current_pub)

            if self.index_all_authors:
                for author in publication.authors:
                    self.matches_by_author[author].append(publication)
                self.total_matches += len(publication.authors)
            else:
                for author in publication.authors:
                    if author not in self.target_authors:
                        continue
                    bucket = self.matches_by_author[author]
                    if self.max_results_per_author is None or len(bucket) < self.max_results_per_author:
                        bucket.append(publication)
                        self.total_matches += 1

                # Optional early-stop when each requested author reached target count.
                if self.target_authors and self.max_results_per_author is not None:
                    done = all(
                        len(self.matches_by_author[author]) >= self.max_results_per_author
                        for author in self.target_authors
                    )
                    if done:
                        raise StopParsing()

            self.current_pub = None

    def _build_publication(self, data: dict[str, object]) -> Publication:
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


class DblpQueryEngine:
    """Query DBLP publications with optional one-time index construction.

    Without preloading, each query scans the XML once.
    With ``preload_index=True``, the constructor performs one full scan and
    subsequent queries are in-memory lookups.
    """

    def __init__(
        self,
        xml_gz_path: Path | str | None = None,
        dtd_path: Path | str | None = None,
        preload_index: bool = False,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.xml_path = (
            Path(xml_gz_path) if xml_gz_path is not None else base_dir / "data" / "dblp.xml.gz"
        )
        self.dtd_file = Path(dtd_path) if dtd_path is not None else base_dir / "data" / "dblp.dtd"

        if not self.xml_path.exists():
            raise FileNotFoundError(f"Missing DBLP dump: {self.xml_path}")
        if not self.dtd_file.exists():
            raise FileNotFoundError(f"Missing DTD file: {self.dtd_file}")

        self._author_index: dict[str, list[Publication]] | None = None
        self.index_build_seconds: float | None = None
        if preload_index:
            self.build_author_index()

    def _parse_with_handler(self, content_handler: DblpAuthorHandler) -> None:
        sax_parser = make_parser()
        sax_parser.setFeature(feature_external_ges, True)
        sax_parser.setEntityResolver(LocalDtdResolver(self.dtd_file))
        sax_parser.setContentHandler(content_handler)

        with gzip.open(self.xml_path, "rb") as xml_file:
            source = InputSource()
            source.setByteStream(xml_file)
            source.setSystemId(str(self.xml_path))
            try:
                sax_parser.parse(source)
            except StopParsing:
                pass

    def build_author_index(self, force: bool = False) -> dict[str, list[Publication]]:
        """Build a full in-memory author->publications index."""
        if self._author_index is not None and not force:
            return self._author_index

        started = time.perf_counter()
        content_handler = DblpAuthorHandler(index_all_authors=True)
        self._parse_with_handler(content_handler)
        self._author_index = dict(content_handler.matches_by_author)
        self.index_build_seconds = time.perf_counter() - started
        return self._author_index

    def query_author(self, author_name: str, max_results: int | None = None) -> list[Publication]:
        author_name = normalize_whitespace(author_name)

        if self._author_index is not None:
            publications = list(self._author_index.get(author_name, []))
            return publications[:max_results] if max_results is not None else publications

        matches = self.query_authors([author_name], max_results_per_author=max_results)
        return matches.get(author_name, [])

    def query_authors(
        self,
        author_names: Sequence[str],
        max_results_per_author: int | None = None,
    ) -> dict[str, list[Publication]]:
        normalized_authors = [normalize_whitespace(author) for author in author_names]

        if self._author_index is not None:
            result: dict[str, list[Publication]] = {}
            for author in normalized_authors:
                publications = list(self._author_index.get(author, []))
                if max_results_per_author is not None:
                    publications = publications[:max_results_per_author]
                result[author] = publications
            return result

        content_handler = DblpAuthorHandler(
            target_authors=set(normalized_authors),
            max_results_per_author=max_results_per_author,
            index_all_authors=False,
        )
        self._parse_with_handler(content_handler)

        result = {}
        for author in normalized_authors:
            result[author] = list(content_handler.matches_by_author.get(author, []))
        return result


def get_publications_by_author(
    author_name: str,
    xml_gz_path: Path | str | None = None,
    dtd_path: Path | str | None = None,
    max_results: int | None = None,
) -> list[Publication]:
    """Return all publications that include the given author name."""
    engine = DblpQueryEngine(xml_gz_path=xml_gz_path, dtd_path=dtd_path, preload_index=False)
    return engine.query_author(author_name, max_results=max_results)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query DBLP publications by author")
    parser.add_argument("author", help='Author name to search, e.g., "Michael Pradel"')
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
        "--limit",
        type=int,
        default=10,
        help="Number of matching publications to print (default: 10)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Optional early stop after collecting this many matches",
    )
    parser.add_argument(
        "--preload-index",
        action="store_true",
        help="Build full author index once (faster for repeated queries in one process)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engine = DblpQueryEngine(args.xml, args.dtd, preload_index=args.preload_index)
    publications = engine.query_author(args.author, max_results=args.max_results)

    print(f"Found {len(publications)} publications for: {args.author}")
    for publication in publications[: args.limit]:
        print(json.dumps(asdict(publication), ensure_ascii=False))

    if args.preload_index and engine.index_build_seconds is not None:
        print(f"Built in-memory author index in {engine.index_build_seconds:.2f}s")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc