"""Generate dataset_statistics.csv from combined metadata."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    stats: Counter[str] = Counter()
    stats["total_recordings"] = len(rows)
    stats["french_recordings"] = sum(row["language"] == "fr" for row in rows)
    stats["english_recordings"] = sum(row["language"] == "en" for row in rows)
    stats["synthetic_recordings"] = sum(row["recording_type"] == "synthetic" for row in rows)
    stats["speakers"] = len({row["speaker_id"] for row in rows if row["speaker_id"]})
    for split in ("train", "validation", "test"):
        stats[f"{split}_count"] = sum(row["split"] == split for row in rows)
    output = args.metadata.parent / "dataset_statistics.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(stats.items())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
