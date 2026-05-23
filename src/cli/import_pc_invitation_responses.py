"""Import PC invitation responses and update person records with acceptance tags.

Given a CSV file with responses (csv_file) and a tag name (tag_name),
this script updates the people store as follows:

1. For people with the given tag who answered "Yes" → add configurable yes tag (default: "#acceptedr1")
2. For people with the given tag who answered "No" → add "#declinedr1" tag
3. For people in the response list without the given tag → print warning
4. For people with the given tag not in the response list → add "#declinedr1" tag

Person matching strategy:
- First try email address matching (handles semicolon-separated emails)
- If no email match, try direct name match ("Given Family" format)
- If no direct match, offer fuzzy matching with user confirmation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import csv
import logging
from typing import Any, Sequence
from difflib import SequenceMatcher

from util.data_store import DataStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import PC invitation responses and update person records with acceptance tags."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to CSV file with PC invitation responses.",
    )
    parser.add_argument(
        "tag_name",
        help="Tag name to match (e.g., 'inviter1'). Will be treated case-insensitively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without modifying the people store.",
    )
    parser.add_argument(
        "--yes-tag",
        default="acceptedr1",
        help="Tag to add for 'Yes' answers (default: acceptedr1).",
    )
    parser.add_argument(
        "--no-tag",
        default="declinedr1",
        help="Tag to add for 'No' answers (default: declinedr1).",
    )

    return parser.parse_args(argv)


def parse_csv_responses(csv_file: Path) -> list[dict[str, str]]:
    """Parse CSV file with PC invitation responses.
    
    Expected columns: "Given (first) name", "Family (last) name", "Your email",
                     "Do you accept the invitation to serve on the committee?"
    """
    responses = []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no headers")
        
        # Check for required columns
        required_cols = ["Given (first) name", "Family (last) name", "Your email", 
                        "Do you accept the invitation to serve on the committee?"]
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"CSV missing required column: {col}")
        
        for row in reader:
            responses.append(row)
    
    return responses


def extract_emails(email_str: str) -> list[str]:
    """Extract individual email addresses from semicolon-separated string."""
    if not email_str or not email_str.strip():
        return []
    
    emails = []
    for email in email_str.split(";"):
        email_clean = email.strip().lower()
        if email_clean:
            emails.append(email_clean)
    return emails


def find_person_by_email(people: dict[str, dict[str, Any]], emails: list[str]) -> str | None:
    """Find person name by matching any of the provided emails."""
    for person_name, person_data in people.items():
        person_email = person_data.get("email")
        if isinstance(person_email, str):
            person_email_clean = person_email.strip().lower()
            if person_email_clean in emails:
                return person_name
    
    return None


def similarity_score(a: str, b: str) -> float:
    """Calculate string similarity score (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_person_by_name_direct(people: dict[str, dict[str, Any]], full_name: str) -> str | None:
    """Find person by direct name match (case-insensitive)."""
    full_name_lower = full_name.strip().lower()
    
    for person_name in people:
        if person_name.lower() == full_name_lower:
            return person_name
    
    return None


def find_person_by_name_fuzzy(
    people: dict[str, dict[str, Any]], 
    full_name: str,
    threshold: float = 0.8
) -> str | None:
    """Find person by fuzzy name matching with threshold."""
    full_name_lower = full_name.strip().lower()
    best_match = None
    best_score = 0.0
    
    for person_name in people:
        score = similarity_score(full_name_lower, person_name.lower())
        if score > best_score:
            best_score = score
            best_match = person_name
    
    if best_match and best_score >= threshold:
        return best_match
    
    return None


def prompt_fuzzy_match(csv_name: str, candidate: str, score: float) -> bool:
    """Ask user to confirm a fuzzy match."""
    print(f"\nFuzzy match (score: {score:.2f})")
    print(f"  CSV name: {csv_name}")
    print(f"  Candidate: {candidate}")
    while True:
        response = input("Accept this match? (y/n): ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'")


def find_person(
    people: dict[str, dict[str, Any]], 
    response_row: dict[str, str],
    allow_fuzzy: bool = True
) -> str | None:
    """Find a person by email first, then by name (direct then fuzzy)."""
    # Try email matching
    emails = extract_emails(response_row.get("Your email", ""))
    if emails:
        email_match = find_person_by_email(people, emails)
        if email_match:
            return email_match
    
    # Try direct name match
    first_name = response_row.get("Given (first) name", "").strip()
    last_name = response_row.get("Family (last) name", "").strip()
    full_name = f"{first_name} {last_name}" if first_name and last_name else ""
    
    if full_name:
        direct_match = find_person_by_name_direct(people, full_name)
        if direct_match:
            return direct_match
        
        # Try fuzzy match
        if allow_fuzzy:
            fuzzy_match = find_person_by_name_fuzzy(people, full_name)
            if fuzzy_match:
                score = similarity_score(full_name.lower(), fuzzy_match.lower())
                if prompt_fuzzy_match(full_name, fuzzy_match, score):
                    return fuzzy_match
    
    return None


def normalize_tag_name(tag: str) -> str:
    """Normalize tag name to lowercase with # prefix."""
    tag_clean = tag.strip().lstrip("#").lower()
    return f"#{tag_clean}"


def has_tag(person: dict[str, Any], tag: str) -> bool:
    """Check if person has a tag (case-insensitive)."""
    flags = person.get("flags")
    if not isinstance(flags, list):
        return False
    
    tag_lower = tag.lower()
    return any(f.lower() == tag_lower for f in flags)


def add_tag(person: dict[str, Any], tag: str) -> dict[str, Any]:
    """Add a tag to person's flags, returning updated person dict."""
    person = dict(person)  # Copy to avoid mutating original
    flags = person.get("flags")
    if not isinstance(flags, list):
        flags = []
    else:
        flags = list(flags)  # Copy list
    
    # Check if tag already present (case-insensitive)
    tag_lower = tag.lower()
    if not any(f.lower() == tag_lower for f in flags):
        flags.append(tag)
    
    person["flags"] = flags
    return person


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    
    # Validate inputs
    if not args.csv_file.exists():
        logger.error(f"CSV file not found: {args.csv_file}")
        return 1
    
    # Parse CSV responses
    try:
        responses = parse_csv_responses(args.csv_file)
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        return 1
    
    if not responses:
        logger.warning("No responses found in CSV file")
        return 0
    
    logger.info(f"Loaded {len(responses)} responses from {args.csv_file}")
    
    # Load people store
    store = DataStore()
    people = store.load()
    logger.info(f"Loaded {len(people)} people from store")
    
    # Normalize tag name
    input_tag = normalize_tag_name(args.tag_name)
    yes_tag = normalize_tag_name(args.yes_tag)
    no_tag = normalize_tag_name(args.no_tag)
    logger.info(f"Processing responses for tag: {input_tag}")
    
    # Track statistics
    matched_yes = []
    matched_no = []
    unmatched_in_responses = []
    not_in_responses = []
    email_updates: dict[str, str] = {}  # person_name -> new email from response
    
    # Process each response
    print()  # Spacing for interactive prompts
    for idx, response in enumerate(responses, 1):
        first_name = response.get("Given (first) name", "").strip()
        last_name = response.get("Family (last) name", "").strip()
        csv_full_name = f"{first_name} {last_name}" if first_name and last_name else ""
        
        # Find matching person
        matched_person = find_person(people, response, allow_fuzzy=not args.dry_run)
        
        if not matched_person:
            logger.warning(f"[{idx}] No person found for: {csv_full_name}")
            unmatched_in_responses.append(csv_full_name)
            continue
        
        matched_person_data = people[matched_person]

        # Check if response email differs from stored email
        csv_emails = extract_emails(response.get("Your email", ""))
        if csv_emails:
            csv_email_primary = csv_emails[0]
            stored_email = matched_person_data.get("email", "").strip().lower()
            if csv_email_primary != stored_email:
                email_updates[matched_person] = csv_email_primary
                logger.info(
                    f"[{idx}] {matched_person}: email will be updated "
                    f"'{stored_email}' -> '{csv_email_primary}'"
                )

        response_answer = response.get("Do you accept the invitation to serve on the committee?", "").strip()
        
        # Check if person has the input tag
        if has_tag(matched_person_data, input_tag):
            if response_answer.lower() == "yes":
                matched_yes.append(matched_person)
                logger.info(f"[{idx}] {matched_person}: accepted (will add {yes_tag})")
            elif response_answer.lower() == "no":
                matched_no.append(matched_person)
                logger.info(f"[{idx}] {matched_person}: declined (will add {no_tag})")
            else:
                logger.warning(f"[{idx}] {matched_person}: unclear response '{response_answer}'")
        else:
            logger.warning(f"[{idx}] {matched_person}: found in responses but does not have tag {input_tag}")
    
    # Find people with tag not in responses
    responses_emails: set[str] = set()
    responses_names: set[str] = set()
    for response in responses:
        emails = extract_emails(response.get("Your email", ""))
        responses_emails.update(emails)
        
        first_name = response.get("Given (first) name", "").strip()
        last_name = response.get("Family (last) name", "").strip()
        if first_name and last_name:
            responses_names.add(f"{first_name} {last_name}".lower())
    
    for person_name, person_data in people.items():
        if not has_tag(person_data, input_tag):
            continue
        
        # Check if this person is in responses.
        # Also check the updated email (from email_updates) in case the stored
        # email is stale — otherwise a person with an outdated email would be
        # incorrectly treated as absent from the responses.
        person_email = person_data.get("email", "").strip().lower()
        updated_email = email_updates.get(person_name, "")
        in_responses = (
            (person_email in responses_emails)
            or (updated_email in responses_emails)
            or (person_name.lower() in responses_names)
        )

        if not in_responses:
            not_in_responses.append(person_name)
            logger.info(f"{person_name}: has {input_tag} but not in responses (will add {no_tag})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Responses processed: {len(responses)}")
    print(f"People with {input_tag} who accepted: {len(matched_yes)}")
    print(f"People with {input_tag} who declined: {len(matched_no)}")
    print(f"People in responses without {input_tag}: {len(unmatched_in_responses)}")
    print(f"People with {input_tag} not in responses: {len(not_in_responses)}")
    print()
    
    if unmatched_in_responses:
        print("WARNING: The following people in responses could not be matched:")
        for name in unmatched_in_responses:
            print(f"  - {name}")
        print()

    print(f"People with updated email addresses: {len(email_updates)}")
    
    if args.dry_run:
        logger.info("DRY RUN: No changes made to people store")
        return 0
    
    # Apply updates
    updates_to_apply = []
    
    # Collect all person names that need any update
    all_updated_names: set[str] = set(matched_yes) | set(matched_no) | set(not_in_responses) | set(email_updates)

    for person_name in all_updated_names:
        person = people[person_name]
        # Apply email update if present
        if person_name in email_updates:
            person = dict(person)
            person["email"] = email_updates[person_name]
        # Apply tag updates
        if person_name in matched_yes:
            person = add_tag(person, yes_tag)
        elif person_name in set(matched_no) | set(not_in_responses):
            person = add_tag(person, no_tag)
        updates_to_apply.append(person)
    
    if updates_to_apply:
        logger.info(f"Updating {len(updates_to_apply)} people in store...")
        try:
            added, updated = store.update_many(updates_to_apply)
            logger.info(f"Store update complete: {added} added, {updated} updated")
        except Exception as e:
            logger.error(f"Failed to update store: {e}")
            return 1
    else:
        logger.info("No updates to apply")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
