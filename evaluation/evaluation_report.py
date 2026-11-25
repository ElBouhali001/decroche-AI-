"""Write a CSV summary from benchmark result rows."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiments/results/benchmark.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/evaluation_report.csv"))
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys() if rows else ["model", "language", "wer", "latency"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
