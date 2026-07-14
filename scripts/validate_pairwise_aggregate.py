#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controllers.AMR.data.pairwise_aggregate import validate_pairwise_aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a pairwise aggregate Shadow Antibiogram parquet artifact."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Parquet file or directory containing part-*.parquet files.",
    )
    parser.add_argument(
        "--expected-genus",
        action="append",
        default=[],
        help="Expected PathogenGenus value. Can be passed multiple times.",
    )
    parser.add_argument(
        "--expected-material",
        action="append",
        default=[],
        help="Expected TextMaterialgroupRkiL0 value. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allow-missing-expected",
        action="store_true",
        help="Report missing configured genera/materials without failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data_path)
    summary = validate_pairwise_aggregate(
        df,
        expected_genera=args.expected_genus or None,
        expected_materials=args.expected_material or None,
        strict_expected=not args.allow_missing_expected,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
