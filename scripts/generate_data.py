"""Write the toy reconciliation case to disk.

Usage:  uv run python scripts/generate_data.py [--out data/toy]
"""

import argparse
from pathlib import Path

from harness.reconciliation import toy_case, write_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/toy"))
    args = parser.parse_args()

    write_case(toy_case(), args.out)
    print(f"Wrote bank.csv, ledger.csv and answer_key.json to {args.out}")


if __name__ == "__main__":
    main()
