"""
Scrape duplicate bug report pairs from Mozilla and Eclipse Bugzilla.

These are the standard academic datasets for bug deduplication research
(Sun et al., ICSE 2011). We use them as a validation set to compare
our results with published literature.

Approach:
  1. Fetch bugs with resolution=DUPLICATE from Bugzilla REST API
  2. Follow the "duplicate of" link to get the canonical bug
  3. Build pairs: (duplicate_bug, canonical_bug) = ground truth duplicate
  4. Sample non-duplicate pairs from the same product/component (hard negatives)

Output:
  data/bugzilla_bugs.json         — raw bug reports
  data/bugzilla_pairs.csv         — labeled pairs
"""

import requests
import json
import csv
import time
import os
import random
import argparse
from collections import defaultdict

# Bugzilla REST API endpoints
SOURCES = {
    "mozilla": {
        "base_url": "https://bugzilla.mozilla.org/rest",
        "products": ["Firefox", "Core"],
        "label": "mozilla",
    },
    "eclipse": {
        "base_url": "https://bugs.eclipse.org/bugs/rest",
        "products": ["Platform", "JDT"],
        "label": "eclipse",
    },
}

TARGET_DUPLICATES = 250  # duplicate pairs per source
TARGET_HARD_NEG = 100    # hard negative pairs per source
MAX_BUGS_PER_QUERY = 100


def fetch_duplicate_bugs(base_url: str, product: str, limit: int = 500) -> list[dict]:
    """Fetch bugs with resolution=DUPLICATE from Bugzilla."""
    bugs = []
    offset = 0

    while len(bugs) < limit:
        params = {
            "product": product,
            "resolution": "DUPLICATE",
            "limit": MAX_BUGS_PER_QUERY,
            "offset": offset,
            "include_fields": "id,summary,description,product,component,creation_time,dupe_of",
        }

        try:
            resp = requests.get(f"{base_url}/bug", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching from {base_url}: {e}")
            break

        batch = data.get("bugs", [])
        if not batch:
            break

        bugs.extend(batch)
        offset += len(batch)
        print(f"  Fetched {len(bugs)} duplicate bugs from {product}...")
        time.sleep(1)  # Rate limiting

    return bugs


def fetch_bug_by_id(base_url: str, bug_id: int) -> dict | None:
    """Fetch a single bug by ID."""
    try:
        resp = requests.get(
            f"{base_url}/bug/{bug_id}",
            params={"include_fields": "id,summary,description,product,component,creation_time"},
            timeout=15,
        )
        resp.raise_for_status()
        bugs = resp.json().get("bugs", [])
        return bugs[0] if bugs else None
    except Exception as e:
        print(f"  Error fetching bug {bug_id}: {e}")
        return None


def fetch_non_duplicate_bugs(base_url: str, product: str, limit: int = 200) -> list[dict]:
    """Fetch non-duplicate bugs from the same product for hard negatives."""
    params = {
        "product": product,
        "resolution": "FIXED",
        "limit": limit,
        "include_fields": "id,summary,description,product,component,creation_time",
    }

    try:
        resp = requests.get(f"{base_url}/bug", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("bugs", [])
    except Exception as e:
        print(f"  Error fetching non-duplicates: {e}")
        return []


def bug_to_report(bug: dict, source_label: str) -> dict:
    """Convert Bugzilla bug to our standard report format."""
    desc = bug.get("description", "") or ""
    # Bugzilla descriptions can be very long — truncate
    if len(desc) > 1000:
        desc = desc[:1000] + "..."

    return {
        "id": f"bz_{source_label}_{bug['id']}",
        "title": bug.get("summary", ""),
        "description": desc,
        "console_logs": [],  # Bugzilla bugs don't have structured console logs
        "network_logs": [],
        "url": "",
        "browser": "",
        "error_type": "bugzilla",
        "component": bug.get("component", ""),
        "group": None,  # Will be set based on duplicate relationships
        "source": source_label,
        "bugzilla_id": bug["id"],
        "product": bug.get("product", ""),
    }


def scrape_source(source_name: str, config: dict, target_pairs: int) -> tuple[list, list]:
    """Scrape one Bugzilla source and build pairs."""
    base_url = config["base_url"]
    label = config["label"]

    print(f"\n{'='*50}")
    print(f"Scraping: {source_name} ({base_url})")
    print(f"{'='*50}")

    all_reports = {}
    duplicate_pairs = []

    for product in config["products"]:
        print(f"\nProduct: {product}")

        # 1. Fetch duplicate bugs
        dup_bugs = fetch_duplicate_bugs(base_url, product, limit=target_pairs * 2)
        print(f"  Found {len(dup_bugs)} duplicate bugs")

        # 2. For each duplicate, fetch its canonical bug and build pair
        for bug in dup_bugs:
            dupe_of = bug.get("dupe_of")
            if not dupe_of:
                continue

            # Convert duplicate bug
            dup_report = bug_to_report(bug, label)
            dup_report["group"] = f"bz_{label}_group_{dupe_of}"
            all_reports[dup_report["id"]] = dup_report

            # Fetch and convert canonical bug
            canonical_id = f"bz_{label}_{dupe_of}"
            if canonical_id not in all_reports:
                canonical_bug = fetch_bug_by_id(base_url, dupe_of)
                if canonical_bug:
                    canonical_report = bug_to_report(canonical_bug, label)
                    canonical_report["group"] = f"bz_{label}_group_{dupe_of}"
                    all_reports[canonical_id] = canonical_report
                    time.sleep(0.5)  # Rate limiting
                else:
                    continue

            duplicate_pairs.append({
                "report_a_id": dup_report["id"],
                "report_b_id": canonical_id,
                "label": "duplicate",
                "pair_type": "D2",  # Bugzilla duplicates are semantic (different reporters)
            })

            if len(duplicate_pairs) >= target_pairs:
                break

        if len(duplicate_pairs) >= target_pairs:
            break

    # 3. Build hard negative pairs (same component, not duplicates)
    hard_neg_pairs = []
    by_component = defaultdict(list)
    for rid, report in all_reports.items():
        by_component[report["component"]].append(rid)

    seen = set()
    for component, members in by_component.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if all_reports[a].get("group") == all_reports[b].get("group"):
                    continue  # Same group = duplicate
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                seen.add(key)
                hard_neg_pairs.append({
                    "report_a_id": a,
                    "report_b_id": b,
                    "label": "not_duplicate",
                    "pair_type": "D3",
                })
                if len(hard_neg_pairs) >= TARGET_HARD_NEG:
                    break
            if len(hard_neg_pairs) >= TARGET_HARD_NEG:
                break
        if len(hard_neg_pairs) >= TARGET_HARD_NEG:
            break

    all_pairs = duplicate_pairs + hard_neg_pairs
    print(f"\nSource {source_name}: {len(all_reports)} reports, "
          f"{len(duplicate_pairs)} dup pairs, {len(hard_neg_pairs)} hard neg pairs")

    return list(all_reports.values()), all_pairs


def main():
    parser = argparse.ArgumentParser(description="Scrape Bugzilla for duplicate bug pairs")
    parser.add_argument("--output-bugs", default="data/bugzilla_bugs.json")
    parser.add_argument("--output-pairs", default="data/bugzilla_pairs.csv")
    parser.add_argument("--sources", nargs="*", default=["mozilla"],
                        choices=["mozilla", "eclipse"],
                        help="Which Bugzilla instances to scrape (default: mozilla)")
    parser.add_argument("--target-pairs", type=int, default=TARGET_DUPLICATES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_reports = []
    all_pairs = []

    for source_name in args.sources:
        config = SOURCES[source_name]
        reports, pairs = scrape_source(source_name, config, args.target_pairs)
        all_reports.extend(reports)
        all_pairs.extend(pairs)

    # Assign pair IDs
    random.shuffle(all_pairs)
    for i, pair in enumerate(all_pairs):
        pair["pair_id"] = f"bz_pair_{i:04d}"

    # Save
    os.makedirs(os.path.dirname(args.output_bugs) or ".", exist_ok=True)

    with open(args.output_bugs, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_reports)} reports to {args.output_bugs}")

    with open(args.output_pairs, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "report_a_id", "report_b_id", "label", "pair_type"])
        writer.writeheader()
        writer.writerows(all_pairs)

    dup_count = sum(1 for p in all_pairs if p["label"] == "duplicate")
    neg_count = sum(1 for p in all_pairs if p["label"] == "not_duplicate")
    print(f"Saved {len(all_pairs)} pairs to {args.output_pairs}")
    print(f"  Duplicates: {dup_count}")
    print(f"  Hard negatives: {neg_count}")


if __name__ == "__main__":
    main()
