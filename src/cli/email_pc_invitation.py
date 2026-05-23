#!/usr/bin/env python3
"""Send PC invitation emails to people with a specific tag.

This script retrieves people with a given tag from the data store,
generates personalized invitation emails from a template, and sends them
via SMTP. It shows sample emails before sending and logs all activity.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import json
import random
import smtplib
import textwrap
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from getpass import getpass
from typing import Any, Sequence

from util.data_store import DataStore


EMAIL_CONFIG = {
    "server": "mail.your-server.de",
    "port": 465,
    "user": "michael@binaervarianz.de",
    "security": "SSL/TLS",
}

TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "templates" / "pc_invitation.txt"


class EmailPreparer:
    """Prepare and manage email content for batch sending."""

    def __init__(self, template_path: Path, deadline_days: int = 7):
        self.template_path = template_path
        self.deadline_days = deadline_days
        self.deadline_str = self._compute_deadline()
        self.subject = ""
        self.template_body = ""
        self._load_template()

    def _compute_deadline(self) -> str:
        """Compute deadline date string (7 days from today)."""
        deadline = datetime.now() + timedelta(days=self.deadline_days)
        return deadline.strftime("%B %d, %Y")

    def _load_template(self) -> None:
        """Load and parse template into subject and body."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

        with self.template_path.open("r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        if not lines[0].startswith("Subject:"):
            raise ValueError("Template must start with 'Subject: XXX' line")

        # Extract subject (remove "Subject: " prefix)
        self.subject = lines[0].replace("Subject:", "").strip()

        # Body starts after the "Subject:" line and the empty line
        # Skip the subject line and the empty line, then join the rest
        body_lines = lines[2:] if len(lines) > 2 else []
        self.template_body = "\n".join(body_lines).strip()

    @staticmethod
    def _wrap_text(text: str, width: int = 72) -> str:
        """
        Wrap text to specified width without breaking words.

        Preserves existing line breaks from the template.
        """
        wrapped_lines: list[str] = []
        for line in text.splitlines():
            if not line.strip():
                wrapped_lines.append("")
                continue

            wrapped = textwrap.fill(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.append(wrapped)

        return "\n".join(wrapped_lines)

    def prepare_email(self, person: dict[str, Any]) -> dict[str, str] | None:
        """
        Prepare an email for a given person.

        Returns:
            Dict with 'to', 'subject', 'body' keys, or None if person has no email.
        """
        email = person.get("email", "").strip()
        if not email:
            return None

        name = person.get("name", "Unknown").strip()

        # Replace placeholders in subject and body
        subject = self.subject.replace("<NAME>", name).replace("<DEADLINE>", self.deadline_str)
        body = self.template_body.replace("<NAME>", name).replace("<DEADLINE>", self.deadline_str)
        
        # Wrap body to 72 characters for plain text email compatibility
        body = self._wrap_text(body)

        return {
            "to": email,
            "subject": subject,
            "body": body,
        }


class EmailSender:
    """Send emails via SMTP."""

    def __init__(self, server: str, port: int, user: str):
        self.server = server
        self.port = port
        self.user = user

    def send_email(self, to: str, subject: str, body: str, password: str) -> bool:
        """
        Send a single email.

        Returns:
            True if successful, False otherwise.
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.user
            msg["To"] = to
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            # Use SSL/TLS on port 465
            with smtplib.SMTP_SSL(self.server, self.port) as server:
                server.login(self.user, password)
                server.send_message(msg)

            return True
        except Exception as e:
            return False


class EmailLogger:
    """Log email activity to file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path

    def log(self, message: str) -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def log_summary(self, total: int, sent: int, skipped: int, failed: int) -> None:
        """Log summary statistics."""
        self.log("")
        self.log("=" * 60)
        self.log(f"Summary: Total={total}, Sent={sent}, Skipped={skipped}, Failed={failed}")
        self.log("=" * 60)


def normalize_tag(tag: str) -> str:
    """Normalize tag to have # prefix if missing."""
    tag = tag.strip()
    if not tag.startswith("#"):
        tag = f"#{tag}"
    return tag


def get_people_with_tag(data_store: DataStore, tag: str) -> list[dict[str, Any]]:
    """Get all people with a specific tag."""
    normalized_tag = normalize_tag(tag)
    people = data_store.list_people()
    filtered = [p for p in people if normalized_tag in p.get("flags", [])]
    return filtered


def show_email_samples(
    emails: list[dict[str, str]], sample_size: int = 3
) -> None:
    """Show sample emails to the user."""
    print("\n" + "=" * 70)
    print("EMAIL PREVIEW")
    print("=" * 70)

    num_samples = min(sample_size, len(emails))
    sample_indices = random.sample(range(len(emails)), num_samples)

    for idx, sample_idx in enumerate(sample_indices, 1):
        email = emails[sample_idx]
        print(f"\n--- Sample {idx}/{num_samples} ---")
        print(f"To: {email['to']}")
        print(f"Subject: {email['subject']}")
        print()
        print(email["body"])
        print()

    print("=" * 70)
    print(f"Total emails to send: {len(emails)}")
    print("=" * 70)


def ask_confirmation() -> bool:
    """Ask user for confirmation.

    Returns:
        True if user confirmed, False otherwise.
    """
    while True:
        response = input("\nProceed with sending these emails? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        else:
            print("Please answer 'yes' or 'no'.")


def get_password() -> str:
    """Prompt user for email password securely."""
    return getpass(f"Enter password for {EMAIL_CONFIG['user']}: ")


def send_emails(
    emails: list[dict[str, str]], password: str, logger: EmailLogger
) -> tuple[int, int, int]:
    """
    Send all emails.

    Returns:
        Tuple of (sent_count, skipped_count, failed_count)
    """
    sender = EmailSender(EMAIL_CONFIG["server"], EMAIL_CONFIG["port"], EMAIL_CONFIG["user"])

    sent_count = 0
    failed_count = 0

    for idx, email in enumerate(emails, 1):
        try:
            if sender.send_email(email["to"], email["subject"], email["body"], password):
                sent_count += 1
                logger.log(f"✓ Sent to {email['to']} ({idx}/{len(emails)})")
                print(f"✓ Sent {idx}/{len(emails)}: {email['to']}")
            else:
                failed_count += 1
                logger.log(f"✗ Failed to send to {email['to']} ({idx}/{len(emails)})")
                print(f"✗ Failed {idx}/{len(emails)}: {email['to']}")
        except Exception as e:
            failed_count += 1
            logger.log(f"✗ Error sending to {email['to']}: {str(e)}")
            print(f"✗ Error {idx}/{len(emails)}: {email['to']} - {str(e)}")

        # Add random delay between emails (0.5 to 1.5 seconds)
        if idx < len(emails):
            delay = random.uniform(0.5, 1.5)
            time.sleep(delay)

    return sent_count, failed_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Send PC invitation emails to people with a specific tag."
    )
    parser.add_argument(
        "tag",
        help='Tag to filter people (e.g., "#inviter1" or "inviter1")',
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(".") / "logs",
        help="Directory to save log file (default: ./logs)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Create log directory if it doesn't exist
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"email_pc_invitation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = EmailLogger(log_path)

    logger.log(f"Starting email campaign for tag: {args.tag}")
    logger.log(f"Email configuration: {EMAIL_CONFIG['user']} @ {EMAIL_CONFIG['server']}")

    try:
        # Load data
        data_store = DataStore()
        people = get_people_with_tag(data_store, args.tag)

        if not people:
            print(f"No people found with tag: {args.tag}")
            logger.log(f"No people found with tag: {args.tag}")
            return 1

        print(f"\nFound {len(people)} people with tag: {args.tag}")

        # Prepare emails
        preparer = EmailPreparer(TEMPLATE_PATH)
        emails = []
        skipped_people = []

        for person in people:
            email_data = preparer.prepare_email(person)
            if email_data:
                emails.append(email_data)
            else:
                skipped_people.append(person.get("name", "Unknown"))

        if not emails:
            print("No people with email addresses found.")
            logger.log("No people with email addresses found.")
            if skipped_people:
                print(f"\nSkipped {len(skipped_people)} people without email addresses:")
                for name in skipped_people:
                    print(f"  - {name}")
            return 1

        # Show samples and ask for confirmation
        show_email_samples(emails)

        if skipped_people:
            print(f"\nWarning: {len(skipped_people)} people have no email address and will be skipped:")
            for name in skipped_people:
                print(f"  - {name}")
            logger.log(f"Skipped {len(skipped_people)} people without email addresses: {', '.join(skipped_people)}")

        # Ask for confirmation
        if not ask_confirmation():
            print("Cancelled by user.")
            logger.log("Cancelled by user before sending.")
            return 0

        # Get password from user
        print()
        password = get_password()

        # Send emails
        print(f"\nSending {len(emails)} emails...")
        sent_count, failed_count = send_emails(emails, password, logger)

        # Print summary
        print("\n" + "=" * 70)
        print("SENDING COMPLETE")
        print("=" * 70)
        print(f"Total emails to send: {len(emails)}")
        print(f"Successfully sent: {sent_count}")
        print(f"Failed: {failed_count}")
        if skipped_people:
            print(f"Skipped (no email): {len(skipped_people)}")
        print(f"Log file: {log_path}")
        print("=" * 70)

        # Log summary
        logger.log_summary(
            total=len(people),
            sent=sent_count,
            skipped=len(skipped_people),
            failed=failed_count,
        )

        return 0 if failed_count == 0 else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logger.log(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
