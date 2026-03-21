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
import logging
import re
from typing import Sequence
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag

from people import PeopleStore
from web_search import search_research_track_pc_page

logger = logging.getLogger(__name__)

RESEARCHR_DOMAIN = "researchr.org"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def is_committee_panel_label(label: str) -> bool:
    lowered = normalize_whitespace(label).lower()
    if not lowered:
        return False

    committee_titles = (
        "program committee",
        "review committee",
    )
    if any(lowered == title or lowered.startswith(f"{title} ") for title in committee_titles):
        return True

    # Some venues prefix the panel title with the conference name, e.g.,
    # "OOPSLA Review Committee".
    if any(lowered.endswith(f" {title}") for title in committee_titles):
        return True

    return "program committee" in lowered and "technical papers" in lowered


def estimate_member_signal(container: Tag) -> int:
    # Prefer containers with person/profile-like structures.
    media_cards = len(container.select("div.media-body h3"))
    person_links = len(container.select("a[href*='/person/']")) + len(
        container.select("a[href*='/profile/']")
    )
    table_rows = len(container.select("tr"))
    list_items = len(container.select("li"))
    return (media_cards * 4) + (person_links * 2) + table_rows + list_items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and extract Program Committee members for a conference/year."
    )
    parser.add_argument("conference", help="Conference acronym or name, e.g., ISSTA")
    parser.add_argument("year", type=int, help="Conference year, e.g., 2024")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted PC members and affiliations without updating the people store.",
    )
    return parser.parse_args(argv)


def search_research_track_url(conference: str, year: int) -> str:
    logger.info(f"Searching for research track page for {conference} {year}...")
    result = search_research_track_pc_page(conference, year)
    logger.debug(f"Web search returned: {result.top_link}")
    if is_researchr_url(result.top_link):
        logger.info(f"Found researchr page: {result.top_link}")
        return result.top_link
    raise ValueError(
        f"Could not find a researchr.org main research track page for {conference} {year}. "
        f"Got: {result.top_link}"
    )


def is_researchr_url(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc == RESEARCHR_DOMAIN or netloc.endswith(f".{RESEARCHR_DOMAIN}")


def is_followable_committee_url(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    base_netloc = urlparse(base_url).netloc.lower()

    if netloc == base_netloc or is_researchr_url(url):
        return True

    path = parsed.path.lower()
    committee_markers = (
        "/committee/",
        "-program-committee",
        "-review-committee",
        "review-committee",
        "program-committee",
    )
    return any(marker in path for marker in committee_markers)


def download_html(url: str) -> str:
    logger.info(f"Downloading HTML from: {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Chairvana PC Extractor)"})
    with urlopen(request, timeout=30) as response:  # noqa: S310
        raw = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
    decoded = raw.decode(encoding, errors="replace")
    logger.debug(f"Downloaded {len(decoded)} characters")
    return decoded


def find_program_committee_container(
    soup: BeautifulSoup, base_url: str, visited_urls: set[str] | None = None
) -> Tag:
    if visited_urls is None:
        visited_urls = set()
    visited_urls.add(base_url)

    logger.info("Searching for Program Committee container...")
    markers: list[Tag] = []
    marker_tags = {"a", "button", "h1", "h2", "h3", "h4", "h5", "h6", "span", "div"}
    for tag in soup.find_all(marker_tags):
        label = normalize_whitespace(tag.get_text(" ", strip=True))
        if not label:
            continue
        if len(label) > 100:
            continue

        if is_committee_panel_label(label):
            markers.append(tag)
            logger.debug(f"Found PC marker: {label}")

    if not markers:
        raise ValueError("Could not find a Program Committee marker in the page")
    logger.info(f"Found {len(markers)} potential Program Committee marker(s)")

    best_target: Tag | None = None
    best_score = -1
    for marker in markers:
        target = resolve_marker_target(soup, marker, base_url, visited_urls)
        if target is None:
            continue
        score = estimate_member_signal(target)
        if score > best_score:
            best_score = score
            best_target = target

    if best_target is not None:
        logger.info(f"Selected committee container with score {best_score}")
        return best_target

    # Fallback: return the first marker if no explicit target can be resolved.
    return markers[0]


def resolve_marker_target(
    soup: BeautifulSoup, marker: Tag, base_url: str, visited_urls: set[str]
) -> Tag | None:
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

    # If the marker is a link to another page, follow it when it still looks
    # like a committee page.
    href = marker.get("href")
    if not isinstance(href, str) or not href:
        nested_link = marker.find("a", href=True)
        if isinstance(nested_link, Tag):
            nested_href = nested_link.get("href")
            if isinstance(nested_href, str) and nested_href:
                href = nested_href

    if isinstance(href, str) and href and not href.startswith("#"):
        next_url = urljoin(base_url, href)
        if is_followable_committee_url(next_url, base_url):
            if next_url in visited_urls:
                logger.debug(f"Skipping already-visited URL to avoid cycle: {next_url}")
                return marker
            logger.debug(f"Following PC marker link to: {next_url}")
            try:
                next_html = download_html(next_url)
            except Exception:  # noqa: BLE001
                logger.debug(f"Failed to download from {next_url}, using marker as fallback")
                return marker
            visited_urls.add(next_url)
            next_soup = BeautifulSoup(next_html, "html.parser")
            main = next_soup.find("main")
            if isinstance(main, Tag):
                return main
            body = next_soup.body
            if isinstance(body, Tag):
                return body
            return marker

    return marker


def looks_like_name(value: str) -> bool:
    cleaned = normalize_whitespace(value)
    if len(cleaned) < 4 or len(cleaned) > 80:
        return False
    if any(char.isdigit() for char in cleaned):
        return False

    lowered = cleaned.lower()
    disallowed_phrases = (
        "technical papers",
        "distinguished paper award",
        "link to publication",
        "media attached",
        "pre-print",
        "not scheduled",
        "doi",
        "talk",
    )
    if any(phrase in lowered for phrase in disallowed_phrases):
        return False

    words = cleaned.split(" ")
    if len(words) < 2 or len(words) > 6:
        return False

    disallowed_name_tokens = {
        "university",
        "institute",
        "college",
        "school",
        "department",
        "laboratory",
        "lab",
        "research",
        "technical",
        "papers",
        "award",
        "doi",
    }
    if any(token.lower().strip(".,;:()[]") in disallowed_name_tokens for token in words):
        return False

    candidate_tokens = 0
    uppercase_like_tokens = 0
    for token in words:
        stripped = token.strip(".,;:()[]")
        if not stripped or not any(char.isalpha() for char in stripped):
            continue
        candidate_tokens += 1

        if stripped[0].isupper():
            uppercase_like_tokens += 1
            continue

        # Handle names like d'Amorim where the first letter is lower-case.
        if "'" in stripped and any(char.isupper() for char in stripped[1:]):
            uppercase_like_tokens += 1

    return candidate_tokens >= 2 and uppercase_like_tokens >= 2


def looks_like_affiliation(value: str) -> bool:
    cleaned = normalize_whitespace(value)
    if len(cleaned) < 3:
        return False

    lowered = cleaned.lower()
    disallowed_affiliation_phrases = (
        "technical papers",
        "distinguished paper award",
        "link to publication",
        "media attached",
        "pre-print",
        "doi",
    )
    return not any(phrase in lowered for phrase in disallowed_affiliation_phrases)


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
    logger.info("Extracting PC members from container...")
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
        if not looks_like_name(clean_name) or not looks_like_affiliation(clean_aff):
            return
        key = (clean_name, clean_aff)
        if key not in seen:
            seen.add(key)
            members.append(key)
            logger.debug(f"Added PC member: {clean_name} ({clean_aff})")

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
        logger.info(f"Extracted {len(members)} members from media blocks")
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

    logger.info(f"Extracted {len(members)} PC members total")
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


def print_members(members: list[tuple[str, str]]) -> None:
    for name, affiliation in members:
        print(f"{name}\t{affiliation}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logger.info(f"Starting PC member extraction for {args.conference} {args.year}")

    conference_url = search_research_track_url(args.conference, args.year)
    html = download_html(conference_url)
    logger.debug("Parsing HTML with BeautifulSoup...")
    soup = BeautifulSoup(html, "html.parser")

    pc_container = find_program_committee_container(soup, conference_url, visited_urls={conference_url})
    members = extract_pc_members(pc_container)
    if not members:
        logger.warning("No members found in container, trying full page...")
        # Some researchr pages link to committee content but the marker itself is not the parent container.
        members = extract_pc_members(soup)
    if not members:
        raise ValueError("Found Program Committee panel, but no member entries were extracted")

    print(f"Conference page: {conference_url}")
    print(f"Extracted {len(members)} PC members")

    if args.dry_run:
        logger.info("Dry run mode - displaying members without updating store")
        print_members(members)
        return 0

    logger.info("Updating people store...")
    store = PeopleStore()
    added_count, updated_count = store.update_many(
        build_people_updates(args.conference, args.year, members)
    )

    logger.info(f"Successfully updated store: {added_count} added, {updated_count} updated")
    print(f"Added {added_count} people and updated {updated_count} people")
    print(f"Saved JSONL to: {store.path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("An error occurred during execution:")
        raise
