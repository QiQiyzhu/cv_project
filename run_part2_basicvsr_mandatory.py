#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vsr_benchmarks import get_gt_ready_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List GT-ready benchmark datasets registered for full-metric evaluation.")
    parser.add_argument(
        "--workdir",
        default=str(PROJECT_ROOT),
        help="Project root used to resolve registered benchmark dataset paths.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the dataset registry as JSON instead of a readable text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = get_gt_ready_benchmarks(args.workdir)
    items = [benchmarks[name].to_json_dict() for name in sorted(benchmarks)]

    if args.json:
        print(json.dumps(items, indent=2))
        return

    for item in items:
        print(f"[{item['name']}]")
        print(f"  lr_root: {item['lr_root']}")
        print(f"  gt_root: {item['gt_root']}")
        print(f"  sequence_source: {item['sequence_source']}")
        print(f"  frame_match: {item['frame_match']}")
        print(f"  lr_exists: {item['lr_exists']}")
        print(f"  gt_exists: {item['gt_exists']}")
        if item["description"]:
            print(f"  description: {item['description']}")
        if item["notes"]:
            print(f"  notes: {item['notes']}")
        print()


if __name__ == "__main__":
    main()
