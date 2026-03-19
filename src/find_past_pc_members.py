"""Find and extract Program Committee members from past conferences.

Given a conference name and year (for example, "ISSTA" and 2024), this script:
1. Uses `web_search.py` to find the research-track page on researchr.org.
2. Downloads and parses the HTML with BeautifulSoup.
3. Finds the "Program Committee" panel/section.
4. Extracts PC member names and affiliations.
5. Updates person records through `people.py`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag

from people import PeopleStore
from web_search import search_best_web_result

RESEARCHR_DOMAIN = "researchr.org"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and extract Program Committee members for a conference/year."
    )
    parser.add_argument("conference", help="Conference acronym or name, e.g., ISSTA")
    parser.add_argument("year", type=int, help="Conference year, e.g., 2024")
    parser.add_argument(
        "--people-file",
        type=Path,
        default=None,
        help="Optional people JSONL path override. By default, people.py chooses the store location.",
    )
    return parser.parse_args(argv)


def search_research_track_url(conference: str, year: int) -> str:
    primary_query = f"{conference} {year} research track Program Committee site:{RESEARCHR_DOMAIN}"
    primary_result = search_best_web_result(primary_query).top_link
    if is_researchr_url(primary_result):
        return primary_result

    # Retry with a stricter query if the first result is outside researchr.org.
    fallback_query = f"{conference} {year} researchr research track"
    fallback_result = search_best_web_result(fallback_query).top_link
    if is_researchr_url(fallback_result):
        return fallback_result

    raise ValueError(
        "Could not find a researchr.org conference page. "
        f"Primary result: {primary_result}; fallback result: {fallback_result}"
    )


def is_researchr_url(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc == RESEARCHR_DOMAIN or netloc.endswith(f".{RESEARCHR_DOMAIN}")


def download_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Chairvana PC Extractor)"})
    with urlopen(request, timeout=30) as response:  # noqa: S310
        raw = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
    return raw.decode(encoding, errors="replace")


def find_program_committee_container(soup: BeautifulSoup, base_url: str) -> Tag:
    markers: list[Tag] = []
    marker_tags = {"a", "button", "h1", "h2", "h3", "h4", "h5", "h6", "span", "div"}
    for tag in soup.find_all(marker_tags):
        label = normalize_whitespace(tag.get_text(" ", strip=True))
        if not label:
            continue
        if len(label) > 100:
            continue

        lowered = label.lower()
        if (
            lowered == "program committee"
            or lowered.startswith("program committee")
            or ("program committee" in lowered and "technical papers" in lowered)
        ):
            markers.append(tag)

    if not markers:
        raise ValueError("Could not find a Program Committee marker in the page")

    for marker in markers:
        target = resolve_marker_target(soup, marker, base_url)
        if target is not None:
            return target

    # Fallback: return the first marker if no explicit target can be resolved.
    return markers[0]


def resolve_marker_target(soup: BeautifulSoup, marker: Tag, base_url: str) -> Tag | None:
    # Handle links or buttons that target an in-page panel, e.g., href="#program-committee".
    for attr in ("href", "data-bs-target", "data-target"):
        target_ref = marker.get(attr)
        if isinstance(target_ref, str) and target_ref.startswith("#"):
            target = soup.select_one(target_ref)
            if isinstance(target, Tag):
                return target

    # If this is a regular heading, prefer its nearest section-like parent.
    if marker.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        for ancestor in marker.parents:
            if isinstance(ancestor, Tag) and ancestor.name in {"section", "article", "div", "main"}:
                return ancestor

    # If the marker is a link to another page on researchr, follow it.
    href = marker.get("href")
    if isinstance(href, str) and href and not href.startswith("#"):
        next_url = urljoin(base_url, href)
        if is_researchr_url(next_url):
            try:
                next_html = download_html(next_url)
            except Exception:  # noqa: BLE001
                return marker
            next_soup = BeautifulSoup(next_html, "html.parser")
            try:
                return find_program_committee_container(next_soup, next_url)
            except Exception:  # noqa: BLE001
                return marker

    return marker


def looks_like_name(value: str) -> bool:
    cleaned = normalize_whitespace(value)
    if len(cleaned) < 4:
        return False
    if any(char.isdigit() for char in cleaned):
        return False
    words = cleaned.split(" ")
    return len(words) >= 2


def parse_name_affiliation(text: str) -> tuple[str, str] | None:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return None

    # Common formats: "Name (Affiliation)", "Name - Affiliation", "Name, Affiliation".
    paren_match = re.match(r"^(.+?)\s*\((.+)\)$", cleaned)
    if paren_match:
        name = normalize_whitespace(paren_match.group(1))
        affiliation = normalize_whitespace(paren_match.group(2))
        if looks_like_name(name):
            return name, affiliation

    for separator in (" - ", " – ", " — ", ", "):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            name = normalize_whitespace(left)
            affiliation = normalize_whitespace(right)
            if looks_like_name(name) and affiliation:
                return name, affiliation

    return None


def extract_pc_members(container: Tag) -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def is_non_member_label(name: str) -> bool:
        lowered = name.lower()
        disallowed_fragments = (
            "attending venue",
            "venue:",
            "reception:",
            "banquet",
            "toggle navigation",
            "travel support",
            "visa support",
        )
        return any(fragment in lowered for fragment in disallowed_fragments)

    def strip_role_suffix(name: str) -> str:
        cleaned = normalize_whitespace(name)
        role_patterns = (
            r"\s+Area\s+Co-Chair(?:\s+for\s+.+)?$",
            r"\s+Program\s+Co-Chair$",
            r"\s+Review\s+Process\s+Co-Chair$",
            r"\s+General\s+Co-Chair$",
            r"\s+Track\s+Chair$",
            r"\s+Program\s+Chair$",
            r"\s+General\s+Chair$",
            r"\s+PC\s+Chair$",
            r"\s+Committee\s+Member$",
        )
        for pattern in role_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return normalize_whitespace(cleaned)

    def add_member(name: str, affiliation: str) -> None:
        clean_name = strip_role_suffix(name)
        clean_aff = normalize_whitespace(affiliation)
        if ":" in clean_name or is_non_member_label(clean_name):
            return
        if not looks_like_name(clean_name) or not clean_aff:
            return
        key = (clean_name, clean_aff)
        if key not in seen:
            seen.add(key)
            members.append(key)

    # 0) researchr committee pages commonly use "div.media-body" cards.
    media_blocks = container.select("div.media-body")
    for block in media_blocks:
        name_tag = block.find("h3")
        if not isinstance(name_tag, Tag):
            continue
        name = normalize_whitespace(name_tag.get_text(" ", strip=True))

        aff_parts: list[str] = []
        for aff_tag in block.find_all("h4"):
            aff_text = normalize_whitespace(aff_tag.get_text(" ", strip=True))
            if aff_text:
                aff_parts.append(aff_text)

        if aff_parts:
            add_member(name, ", ".join(aff_parts))

    # If researchr member cards are present, trust that structured representation
    # and avoid generic fallbacks that can pull navigation/venue list text.
    if media_blocks and members:
        return members

    # 1) Structured tables are the most reliable source.
    for row in container.select("tr"):
        cells = [normalize_whitespace(cell.get_text(" ", strip=True)) for cell in row.select("th, td")]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 2:
            add_member(cells[0], cells[1])
        elif len(cells) == 1:
            parsed = parse_name_affiliation(cells[0])
            if parsed is not None:
                add_member(parsed[0], parsed[1])

    # 2) List entries often contain one member each.
    for item in container.select("li"):
        text = normalize_whitespace(item.get_text(" ", strip=True))
        parsed = parse_name_affiliation(text)
        if parsed is not None:
            add_member(parsed[0], parsed[1])

    # 3) Extract from person links and nearby text.
    for person_link in container.select("a[href*='/person/']"):
        name = normalize_whitespace(person_link.get_text(" ", strip=True))
        if not looks_like_name(name):
            continue

        parent = person_link.parent if isinstance(person_link.parent, Tag) else None
        if parent is None:
            continue

        parent_text = normalize_whitespace(parent.get_text(" ", strip=True))
        if not parent_text:
            continue

        remainder = normalize_whitespace(parent_text.replace(name, "", 1))
        remainder = remainder.strip("-–—,;:()[] ")
        if remainder:
            add_member(name, remainder)

    return members


def build_people_updates(
    conference: str, year: int, members: list[tuple[str, str]]
) -> list[dict[str, object]]:
    conference_name = normalize_whitespace(conference)
    updates: list[dict[str, object]] = []
    for name, affiliation in members:
        updates.append(
            {
                "name": name,
                "affiliation": affiliation,
                "pc_memberships": [{"conference": conference_name, "year": year}],
            }
        )
    return updates


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    conference_url = search_research_track_url(args.conference, args.year)
    html = download_html(conference_url)
    soup = BeautifulSoup(html, "html.parser")

    pc_container = find_program_committee_container(soup, conference_url)
    members = extract_pc_members(pc_container)
    if not members:
        # Some researchr pages link to committee content but the marker itself is not the parent container.
        members = extract_pc_members(soup)
    if not members:
        raise ValueError("Found Program Committee panel, but no member entries were extracted")

    store = PeopleStore(args.people_file)
    added_count, updated_count = store.update_many(
        build_people_updates(args.conference, args.year, members)
    )

    print(f"Conference page: {conference_url}")
    print(f"Extracted {len(members)} PC members")
    print(f"Added {added_count} people and updated {updated_count} people")
    print(f"Saved JSONL to: {store.path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc
