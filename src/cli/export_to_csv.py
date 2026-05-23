#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

"""
Helper script to export people flagged with a specific tag from people.jsonl to CSV.
"""

import json
import csv
import sys
import argparse
from pathlib import Path


def split_given_and_family_name(full_name):
    """Split a full name into given and family parts using a simple heuristic."""
    parts = [part for part in full_name.strip().split() if part]
    if not parts:
        return '', ''
    if len(parts) == 1:
        return parts[0], ''
    return ' '.join(parts[:-1]), parts[-1]


def export_tagged_people_to_csv(jsonl_path, csv_path, tag, researchr_role=None):
    """
    Read people.jsonl and export people flagged with a specific tag to CSV.
    
    Args:
        jsonl_path (str): Path to the people.jsonl file
        csv_path (str): Path to output CSV file
        tag (str): Tag to filter by (e.g., 'invite', 'review')
        researchr_role (str | None): Role name for researchr.org export format
    """
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        return False
    
    tagged_people = []
    
    # Read JSONL file and filter for the specified tag
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                person = json.loads(line.strip())
                flags = person.get('flags', [])
                
                if f'#{tag}' in flags:
                    tagged_people.append({
                        'name': person.get('name', ''),
                        'affiliation': person.get('affiliation', ''),
                        'email': person.get('email', ''),
                        'homepage': person.get('homepage', ''),
                        'country': person.get('country', '')
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                continue
    
    # Write to CSV file
    if not tagged_people:
        print(f"Warning: No people found with #{tag} flag", file=sys.stderr)
    
    if researchr_role:
        fieldnames = ['given name', 'family name', 'email address', 'affiliation', 'role name']
        rows = []
        for person in tagged_people:
            given_name, family_name = split_given_and_family_name(person.get('name', ''))
            rows.append({
                'given name': given_name,
                'family name': family_name,
                'email address': person.get('email', ''),
                'affiliation': person.get('affiliation', ''),
                'role name': researchr_role,
            })
    else:
        fieldnames = ['name', 'affiliation', 'email', 'homepage', 'country']
        rows = tagged_people
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                doublequote=True,
            )
            if not researchr_role:
                writer.writeheader()
            writer.writerows(rows)
        
        print(f"Success: Exported {len(tagged_people)} people to {csv_path}")
        return True
    except IOError as e:
        print(f"Error: Could not write to {csv_path}: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export people flagged with a specific tag from people.jsonl to CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Export people with #invite tag (default)
  python export_to_csv.py
  
  # Export people with custom tag
  python export_to_csv.py --tag review
  
  # Specify custom input and output paths
  python export_to_csv.py data/custom.jsonl output.csv --tag invite
  
  # Specify custom tag and output filename
  python export_to_csv.py --tag review --output review_people.csv

    # Export in researchr.org import format with fixed role name
    python export_to_csv.py --tag areachair --researchr "Area Chair"
        '''
    )
    
    parser.add_argument(
        'jsonl_path',
        nargs='?',
        default='data/.people_repo/people.jsonl',
        help='Path to the people.jsonl file (default: data/.people_repo/people.jsonl)'
    )
    
    parser.add_argument(
        'csv_path',
        nargs='?',
        default=None,
        help='Path to output CSV file (default: {tag}_people.csv where tag is the export tag)'
    )
    
    parser.add_argument(
        '--tag',
        default='invite',
        help='Tag to filter people by (default: invite). Will look for #TAG in flags.'
    )
    
    parser.add_argument(
        '--output', '-o',
        dest='output_csv',
        help='Output CSV file path. Overrides csv_path argument.'
    )

    parser.add_argument(
        '--researchr',
        metavar='ROLE',
        help='Export CSV in researchr.org format with this fixed role name.'
    )
    
    args = parser.parse_args()
    
    # Determine output CSV path
    if args.output_csv:
        csv_output = args.output_csv
    elif args.csv_path:
        csv_output = args.csv_path
    else:
        csv_output = f'{args.tag}_people.csv'
    
    success = export_tagged_people_to_csv(
        args.jsonl_path,
        csv_output,
        args.tag,
        researchr_role=args.researchr,
    )
    sys.exit(0 if success else 1)
