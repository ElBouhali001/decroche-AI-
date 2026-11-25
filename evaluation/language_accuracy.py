"""Measure language detection accuracy from a metadata CSV."""

import argparse
import csv
from pathlib import Path

from backend.stt import SpeechToText


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--model", default="base")
    args = parser.parse_args()
    root = args.metadata.parent.parent
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    correct = sum(SpeechToText(args.model).transcribe(root / row["audio_file"]).language.code == row["language"] for row in rows)
    print({"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else 0.0})


if __name__ == "__main__":
    main()
