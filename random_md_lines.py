#!/usr/bin/env python3

import sys
import random
import re

def collect_lines(filepath):
    items = []
    current_section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Detect section headers like **2025:**
            match = re.match(r"\*\*(.+?)\*\*", line)
            if match:
                current_section = match.group(1)
                continue

            if current_section:
                items.append((line, current_section))

    return items

def main():
    if len(sys.argv) != 2:
        print("Usage: random_md_lines.py <file.md>")
        sys.exit(1)

    items = collect_lines(sys.argv[1])

    if not items:
        print("No lines found.")
        sys.exit(1)

    random.shuffle(items)

    for line, section in items:
        print(line)
        print(f"({section})\n")

if __name__ == "__main__":
    main()
