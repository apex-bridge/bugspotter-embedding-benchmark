"""
Generate ground truth pairs (D1-D4) from the combined bug_reports.json dataset.

Pair types:
  D1 — Exact duplicates (same group, very high word overlap)
  D2 — Semantic duplicates / paraphrases (same group, different wording)
  D3 — Hard negatives (different bug in same COMPONENT — overlapping vocabulary)
  D4 — Easy negatives (completely different bugs, different components)

The key to a good benchmark: D3 pairs must be HARD to distinguish from D2.
We achieve this by pairing bugs from the same component (e.g., two different
checkout bugs, two different auth bugs) — they share terminology but describe
different issues.

Output: pairs_ground_truth.csv
"""

import json
import csv
import random
import argparse
import os
from itertools import combinations
from collections import defaultdict


def load_reports(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_groups(reports: list[dict]) -> dict[str, list[dict]]:
    """Group reports by their 'group' field."""
    groups = defaultdict(list)
    for r in reports:
        g = r.get("group")
        if g:
            groups[g].append(r)
    return dict(groups)


def text_similarity_crude(a: dict, b: dict) -> float:
    """Quick word-overlap ratio to distinguish D1 (exact) from D2 (paraphrase)."""
    words_a = set(a.get("title", "").lower().split() + a.get("description", "").lower().split())
    words_b = set(b.get("title", "").lower().split() + b.get("description", "").lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


def generate_duplicate_pairs(groups: dict) -> tuple[list, list]:
    """Generate D1 and D2 pairs from within-group combinations."""
    d1_pairs = []
    d2_pairs = []

    for group_id, members in groups.items():
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            overlap = text_similarity_crude(a, b)
            pair = {
                "report_a_id": a["id"],
                "report_b_id": b["id"],
                "label": "duplicate",
            }
            # D1: very high word overlap (near-identical text)
            # D2: same bug, different words (the main test)
            if overlap > 0.65:
                pair["pair_type"] = "D1"
                d1_pairs.append(pair)
            else:
                pair["pair_type"] = "D2"
                d2_pairs.append(pair)

    return d1_pairs, d2_pairs


def get_component(report: dict) -> str:
    """Extract a meaningful component label for grouping hard negatives."""
    # Synthetic/BS captures have explicit component
    if report.get("component"):
        return report["component"]
    # GitHub issues — use the repo name from the URL
    url = report.get("url", "")
    if "github.com" in url:
        # e.g. https://github.com/facebook/react/issues/1234 -> "facebook/react"
        parts = url.replace("https://github.com/", "").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    # Fallback to error_type
    return report.get("error_type", "unknown")


def generate_hard_negatives(reports: list[dict], groups: dict, target: int = 600) -> list:
    """
    D3 — Different bugs from the SAME COMPONENT or same error_type.

    This is the critical category. These pairs share vocabulary (e.g., both
    mention "checkout", "modal", "login") but describe different bugs.
    They should be hard for embedding models to distinguish from real duplicates.
    """
    by_component = defaultdict(list)
    for r in reports:
        by_component[get_component(r)].append(r)

    d3_pairs = []
    seen = set()

    # First pass: same component, different group (the hardest negatives)
    for component, members in by_component.items():
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            if a.get("group") == b.get("group"):
                continue  # Same group = duplicate, skip
            key = tuple(sorted([a["id"], b["id"]]))
            if key in seen:
                continue
            seen.add(key)
            d3_pairs.append({
                "report_a_id": a["id"],
                "report_b_id": b["id"],
                "label": "not_duplicate",
                "pair_type": "D3",
            })

    # Second pass: same error_type, different group
    by_error = defaultdict(list)
    for r in reports:
        by_error[r.get("error_type", "unknown")].append(r)

    for error_type, members in by_error.items():
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            if a.get("group") == b.get("group"):
                continue
            key = tuple(sorted([a["id"], b["id"]]))
            if key in seen:
                continue
            seen.add(key)
            d3_pairs.append({
                "report_a_id": a["id"],
                "report_b_id": b["id"],
                "label": "not_duplicate",
                "pair_type": "D3",
            })
            if len(d3_pairs) >= target * 3:
                break
        if len(d3_pairs) >= target * 3:
            break

    random.shuffle(d3_pairs)
    return d3_pairs[:target]


def generate_easy_negatives(reports: list[dict], target: int = 1400) -> list:
    """
    D4 — Completely different bugs (different group AND different component/error_type).
    """
    d4_pairs = []
    seen = set()
    attempts = 0
    max_attempts = target * 20

    while len(d4_pairs) < target and attempts < max_attempts:
        a, b = random.sample(reports, 2)
        if a.get("group") == b.get("group"):
            attempts += 1
            continue
        # Must be different component AND different error_type
        comp_a = a.get("component") or a.get("url", "")
        comp_b = b.get("component") or b.get("url", "")
        if comp_a == comp_b or a.get("error_type") == b.get("error_type"):
            attempts += 1
            continue
        key = tuple(sorted([a["id"], b["id"]]))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        d4_pairs.append({
            "report_a_id": a["id"],
            "report_b_id": b["id"],
            "label": "not_duplicate",
            "pair_type": "D4",
        })
        attempts += 1

    return d4_pairs[:target]


def main():
    parser = argparse.ArgumentParser(description="Generate ground truth pairs")
    parser.add_argument("--input", default="data/bug_reports.json")
    parser.add_argument("--output", default="data/pairs_ground_truth.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    reports = load_reports(args.input)
    print(f"Loaded {len(reports)} bug reports")

    groups = build_groups(reports)
    print(f"Found {len(groups)} duplicate groups")

    # Generate all pair types
    d1_pairs, d2_pairs = generate_duplicate_pairs(groups)
    print(f"D1 (exact duplicates): {len(d1_pairs)}")
    print(f"D2 (semantic duplicates): {len(d2_pairs)}")

    d3_pairs = generate_hard_negatives(reports, groups, target=600)
    print(f"D3 (hard negatives — same component): {len(d3_pairs)}")

    d4_pairs = generate_easy_negatives(reports, target=1400)
    print(f"D4 (easy negatives): {len(d4_pairs)}")

    # Combine and assign pair IDs
    all_pairs = d1_pairs + d2_pairs + d3_pairs + d4_pairs
    for i, pair in enumerate(all_pairs):
        pair["pair_id"] = f"pair_{i:04d}"

    random.shuffle(all_pairs)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "report_a_id", "report_b_id", "label", "pair_type"])
        writer.writeheader()
        writer.writerows(all_pairs)

    total_dup = len(d1_pairs) + len(d2_pairs)
    total_neg = len(d3_pairs) + len(d4_pairs)
    print(f"\nTotal: {len(all_pairs)} pairs saved to {args.output}")
    print(f"  Duplicates (D1+D2): {total_dup}")
    print(f"  Not-duplicates (D3+D4): {total_neg}")


if __name__ == "__main__":
    main()
