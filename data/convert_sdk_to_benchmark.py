"""
Convert raw SDK captures + paraphrases into benchmark-format bug reports.

Reads:
  data/sdk-captures/raw/bug_*.json   — 25 raw SDK captures
  data/sdk_paraphrases.json          — 25 × 9 paraphrases (variations 1-9)

Outputs:
  data/sdk_captures.json             — 250 benchmark-format reports

Each bug gets 10 variations:
  - Variation 0: original title/description from the SDK trigger
  - Variations 1-9: from sdk_paraphrases.json

Machine-captured fields (console_logs, network_logs, metadata) are identical
across all 10 variations of the same bug — only title/description change.
"""

import json
import os
import re
import random


def vary_logs(logs, variation_idx, rng):
    """Add slight variation to console/network logs per variation.

    In real life, the exact set of console entries can differ slightly
    between identical bugs — a timing-dependent warning may or may not fire,
    an extra debug log may appear, etc.

    - Variation 0: exact original (no changes)
    - Variations 1-3: might drop 1 non-error entry (30% chance)
    - Variations 4-6: might drop 1-2 non-error entries (40% chance)
    - Variations 7-9: might clear non-error entries entirely (20% chance)
    """
    if variation_idx == 0 or len(logs) <= 1:
        return list(logs)

    result = list(logs)

    if variation_idx <= 3:
        # Occasionally drop one non-critical log entry
        if rng.random() < 0.3 and len(result) > 1:
            # Find a non-error entry to drop
            droppable = [i for i, l in enumerate(result)
                         if not (isinstance(l, str) and "[error]" in l.lower())]
            if droppable:
                result.pop(rng.choice(droppable))
    elif variation_idx <= 6:
        # Drop 1-2 non-critical entries
        if rng.random() < 0.4:
            droppable = [i for i, l in enumerate(result)
                         if not (isinstance(l, str) and "[error]" in l.lower())]
            to_drop = min(rng.randint(1, 2), len(droppable))
            for idx in sorted(rng.sample(droppable, to_drop), reverse=True):
                result.pop(idx)
    else:
        # Occasionally keep only error-level entries (simulate minimal capture)
        if rng.random() < 0.2:
            errors_only = [l for l in result
                           if isinstance(l, str) and "[error]" in l.lower()]
            if errors_only:
                result = errors_only

    return result


def convert_console(sdk_console):
    """SDK console entries -> benchmark string array."""
    result = []
    for entry in sdk_console:
        if isinstance(entry, dict):
            msg = entry.get("message", "")
            level = entry.get("level", "")
            stack = entry.get("stack", "")
            # Format: "[level] message" + first 3 lines of stack
            prefix = f"[{level}] " if level else ""
            text = prefix + msg
            if stack:
                stack_lines = stack.strip().split("\n")[:3]
                text += "\n" + "\n".join(stack_lines)
            result.append(text)
        elif isinstance(entry, str):
            result.append(entry)
    return result


def convert_network(sdk_network):
    """SDK network entries -> benchmark format."""
    result = []
    for entry in sdk_network:
        if isinstance(entry, dict):
            result.append({
                "method": entry.get("method", "GET"),
                "url": entry.get("url", ""),
                "status": entry.get("status", 0),
                "duration": entry.get("duration", 0),
            })
    return result


def extract_stack_trace(sdk_console):
    """Find the first console entry with a stack trace."""
    for entry in sdk_console:
        if isinstance(entry, dict) and entry.get("stack"):
            return entry["stack"]
    return None


def extract_browser(metadata):
    """Extract browser string from SDK metadata."""
    if not metadata:
        return "unknown"
    browser = metadata.get("browser", "")
    ua = metadata.get("userAgent", "")
    if browser:
        # Try to extract version from userAgent
        if "Chrome" in browser and ua:
            match = re.search(r"Chrome/(\d+)", ua)
            if match:
                return f"Chrome {match.group(1)}"
        return browser
    return "unknown"


def extract_url(metadata):
    """Extract page URL from SDK metadata."""
    if not metadata:
        return ""
    url = metadata.get("url", "")
    # Convert full URL to path
    if url.startswith("http"):
        from urllib.parse import urlparse
        return urlparse(url).path or "/"
    return url or "/"


def main():
    raw_dir = os.path.join(os.path.dirname(__file__), "sdk-captures", "raw")
    paraphrases_path = os.path.join(os.path.dirname(__file__), "sdk_paraphrases.json")
    output_path = os.path.join(os.path.dirname(__file__), "sdk_captures.json")

    # Load raw captures
    captures = {}
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, filename), encoding="utf-8") as f:
            data = json.load(f)
        captures[data["bug_slug"]] = data

    print(f"Loaded {len(captures)} raw captures")

    # Load paraphrases
    with open(paraphrases_path, encoding="utf-8") as f:
        paraphrases = json.load(f)

    print(f"Loaded paraphrases for {len(paraphrases)} bugs")

    # Generate benchmark reports
    rng = random.Random(42)  # deterministic noise
    all_reports = []

    for slug, capture in captures.items():
        report_data = capture.get("report", {})
        console_entries = report_data.get("console", [])
        network_entries = report_data.get("network", [])
        metadata = report_data.get("metadata", {})

        # Convert to benchmark format (base — variation 0 gets exact copy)
        console_logs = convert_console(console_entries)
        network_logs = convert_network(network_entries)
        stack_trace = extract_stack_trace(console_entries)
        browser = extract_browser(metadata)
        page_url = extract_url(metadata)
        component = capture["component"]
        error_type = capture["error_type"]
        group = f"sdk_group_{slug}"

        # Variation 0: original title/description, exact logs
        all_reports.append({
            "id": f"sdk_{slug}_00",
            "title": capture["original_title"],
            "description": capture["original_description"],
            "console_logs": console_logs,
            "network_logs": network_logs,
            "stack_trace": stack_trace,
            "url": page_url,
            "browser": browser,
            "error_type": error_type,
            "component": component,
            "group": group,
        })

        # Variations 1-9: from paraphrases, with slight log variation
        bug_paraphrases = paraphrases.get(slug, [])
        for i, para in enumerate(bug_paraphrases[:9], start=1):
            all_reports.append({
                "id": f"sdk_{slug}_{i:02d}",
                "title": para["title"],
                "description": para["description"],
                "console_logs": vary_logs(console_logs, i, rng),
                "network_logs": network_logs,
                "stack_trace": stack_trace,
                "url": page_url,
                "browser": browser,
                "error_type": error_type,
                "component": component,
                "group": group,
            })

    # Verify
    groups = set(r["group"] for r in all_reports)
    print(f"Generated {len(all_reports)} reports in {len(groups)} groups")

    # Check all groups have 10 members
    from collections import Counter
    group_sizes = Counter(r["group"] for r in all_reports)
    for g, count in group_sizes.items():
        if count != 10:
            print(f"  WARNING: {g} has {count} reports (expected 10)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
