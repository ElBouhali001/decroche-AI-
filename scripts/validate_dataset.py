"""Validate dataset metadata and audio paths without fabricating records."""

import argparse
import csv
from pathlib import Path


def validate(metadata_path: Path) -> list[str]:
    errors: list[str] = []
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen: set[str] = set()
    root = metadata_path.parent.parent
    for row in rows:
        path = root / row["audio_file"]
        if not path.is_file():
            errors.append(f"missing audio: {path}")
        if not row.get("transcription", "").strip():
            errors.append(f"missing transcription: {row.get('audio_id')}")
        if row.get("language") not in {"fr", "en"}:
            errors.append(f"unsupported language: {row.get('audio_id')}")
        if row.get("audio_file") in seen:
            errors.append(f"duplicate metadata path: {row['audio_file']}")
        seen.add(row.get("audio_file", ""))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    args = parser.parse_args()
    errors = validate(args.metadata)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Dataset metadata and paths are valid.")


if __name__ == "__main__":
    main()
