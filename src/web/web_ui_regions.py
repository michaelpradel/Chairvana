from __future__ import annotations

import re
from collections import Counter
from typing import Any

GLOBAL_REGIONS = ["North America", "South America", "Europe", "Asia", "Africa"]

# Keep the region set fixed to five buckets for the Distribution panel.
# Oceania and Antarctica country codes are grouped into Asia and Africa, respectively.
_REGION_COUNTRY_CODES: dict[str, set[str]] = {
    "North America": {
        "AIA", "ATG", "ABW", "BHS", "BRB", "BLZ", "BMU", "BES", "VGB", "CAN", "CYM", "CRI", "CUB",
        "CUW", "DMA", "DOM", "SLV", "GRL", "GRD", "GLP", "GTM", "HTI", "HND", "JAM", "MTQ", "MEX",
        "MSR", "NIC", "PAN", "PRI", "BLM", "KNA", "LCA", "MAF", "SPM", "VCT", "SXM", "TTO", "TCA",
        "USA", "VIR", "UMI",
    },
    "South America": {
        "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "FLK", "GUF", "GUY", "PRY", "PER", "SGS", "SUR",
        "URY", "VEN",
    },
    "Europe": {
        "ALB", "AND", "AUT", "BLR", "BEL", "BIH", "BGR", "HRV", "CZE", "DNK", "EST", "FRO", "FIN",
        "FRA", "DEU", "GIB", "GRC", "GGY", "VAT", "HUN", "ISL", "IRL", "IMN", "ITA", "JEY", "LVA",
        "LIE", "LTU", "LUX", "MLT", "MDA", "MCO", "MNE", "NLD", "MKD", "NOR", "POL", "PRT", "ROU",
        "RUS", "SMR", "SRB", "SVK", "SVN", "ESP", "SJM", "SWE", "CHE", "UKR", "GBR", "ALA",
    },
    "Asia": {
        "AFG", "ARM", "AZE", "BHR", "BGD", "BTN", "BRN", "KHM", "CHN", "CXR", "CCK", "CYP", "GEO",
        "HKG", "IND", "IDN", "IRN", "IRQ", "ISR", "JPN", "JOR", "KAZ", "KWT", "KGZ", "LAO", "LBN",
        "MAC", "MYS", "MDV", "MNG", "MMR", "NPL", "PRK", "OMN", "PAK", "PSE", "PHL", "QAT", "SAU",
        "SGP", "KOR", "LKA", "SYR", "TWN", "TJK", "THA", "TLS", "TUR", "TKM", "ARE", "UZB", "VNM",
        "YEM", "IOT",
        # Oceania grouped into Asia to keep the required five-region chart.
        "ASM", "AUS", "COK", "FJI", "PYF", "GUM", "HMD", "KIR", "MHL", "FSM", "NRU", "NCL", "NZL",
        "NIU", "NFK", "MNP", "PLW", "PNG", "PCN", "WSM", "SLB", "TKL", "TON", "TUV", "VUT", "WLF",
    },
    "Africa": {
        "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "COD",
        "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN",
        "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT", "MAR", "MOZ", "NAM", "NER",
        "NGA", "REU", "RWA", "SHN", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA",
        "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE",
        # Antarctica grouped into Africa to keep the required five-region chart.
        "ATA",
    },
}

_COUNTRY_CODE_TO_REGION: dict[str, str] = {
    country_code: region
    for region, country_codes in _REGION_COUNTRY_CODES.items()
    for country_code in country_codes
}

_NORMALIZED_REGION_NAMES: dict[str, str] = {
    re.sub(r"\s+", " ", region).strip().casefold(): region for region in GLOBAL_REGIONS
}


def normalize_tag_query(raw_tags: str) -> list[str]:
    tokens = [token.strip() for token in raw_tags.replace(",", " ").split() if token.strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.casefold()
        if not lowered.startswith("#"):
            lowered = f"#{lowered}"
        if lowered == "#" or lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(lowered)
    return normalized


def normalize_region_name(raw_region: str) -> str | None:
    normalized = re.sub(r"\s+", " ", raw_region).strip().casefold()
    if not normalized:
        return None
    return _NORMALIZED_REGION_NAMES.get(normalized)


def region_for_person(person: dict[str, Any]) -> str | None:
    country = person.get("country")
    if not isinstance(country, str) or not country.strip():
        return None
    return _COUNTRY_CODE_TO_REGION.get(country.strip().upper())


def tagged_people_distribution(people: list[dict[str, Any]], tag_query: str) -> dict[str, Any]:
    selected_tags = normalize_tag_query(tag_query)

    if not selected_tags:
        # No tag given: show distribution of the entire set of people.
        matched_people = people
    else:
        matched_people = []
        for person in people:
            flags = person.get("flags")
            if not isinstance(flags, list):
                continue
            normalized_flags = {
                str(flag).strip().casefold() for flag in flags if isinstance(flag, str) and str(flag).strip()
            }
            if any(tag in normalized_flags for tag in selected_tags):
                matched_people.append(person)

    gender_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    region_counter: Counter[str] = Counter()

    for person in matched_people:
        gender = person.get("gender")
        if isinstance(gender, str) and gender.strip():
            gender_counter[gender.strip().casefold()] += 1
        else:
            gender_counter["unknown"] += 1

        country = person.get("country")
        if isinstance(country, str) and country.strip():
            normalized_country = country.strip().upper()
            country_counter[normalized_country] += 1
            region = region_for_person(person)
            if region is not None:
                region_counter[region] += 1
        else:
            country_counter["UNKNOWN"] += 1

    return {
        "input": " ".join(selected_tags),
        "selected_tags": selected_tags,
        "matched_count": len(matched_people),
        "gender": dict(gender_counter.most_common()),
        "country": dict(country_counter.most_common(12)),
        "region": {
            region: region_counter[region]
            for region in GLOBAL_REGIONS
            if region_counter[region] > 0
        },
    }


def top_common_tags(people: list[dict[str, Any]], limit: int = 5) -> list[str]:
    tag_counter: Counter[str] = Counter()

    for person in people:
        flags = person.get("flags")
        if not isinstance(flags, list):
            continue

        # Count each tag at most once per person to avoid skew from duplicates.
        unique_tags_for_person: set[str] = set()
        for flag in flags:
            if not isinstance(flag, str):
                continue
            normalized = flag.strip().casefold()
            if not normalized:
                continue
            if not normalized.startswith("#"):
                normalized = f"#{normalized}"
            if normalized == "#":
                continue
            unique_tags_for_person.add(normalized)

        tag_counter.update(unique_tags_for_person)

    return [tag for tag, _ in tag_counter.most_common(limit)]
