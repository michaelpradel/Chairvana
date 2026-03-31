#!/usr/bin/env python3
"""
Helper script to export people flagged as #invite from people.jsonl to CSV.
"""

import json
import csv
import sys
from pathlib import Path


def export_invite_people_to_csv(jsonl_path, csv_path):
    """
    Read people.jsonl and export people flagged with #invite to CSV.
    
    Args:
        jsonl_path (str): Path to the people.jsonl file
        csv_path (str): Path to output CSV file
    """
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        return False
    
    invite_people = []
    
    # Read JSONL file and filter for #invite flag
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                person = json.loads(line.strip())
                flags = person.get('flags', [])
                
                if '#invite' in flags:
                    invite_people.append({
                        'name': person.get('name', ''),
                        'affiliation': person.get('affiliation', ''),
                        'homepage': person.get('homepage', ''),
                        'country': person.get('country', '')
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                continue
    
    # Write to CSV file
    if not invite_people:
        print("Warning: No people found with #invite flag", file=sys.stderr)
    
    fieldnames = ['name', 'affiliation', 'homepage', 'country']
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(invite_people)
        
        print(f"Success: Exported {len(invite_people)} people to {csv_path}")
        return True
    except IOError as e:
        print(f"Error: Could not write to {csv_path}: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    # Default paths - can be overridden with command line arguments
    default_jsonl = 'data/.people_repo/people.jsonl'
    default_csv = 'invite_people.csv'
    
    jsonl_input = sys.argv[1] if len(sys.argv) > 1 else default_jsonl
    csv_output = sys.argv[2] if len(sys.argv) > 2 else default_csv
    
    success = export_invite_people_to_csv(jsonl_input, csv_output)
    sys.exit(0 if success else 1)
