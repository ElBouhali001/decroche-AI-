"""Create reproducible speaker-disjoint train/validation/test CSV files.

Speakers are grouped, shuffled with a fixed seed, then assigned greedily so
that each split gets roughly its target share of clips (80/10/10 by default).
Each language is split on its own so French and English keep the same ratios.
A speaker never appears in two splits.
"""

import argparse
import csv
import random
from pathlib import Path

try:
    from .prepare_metadata import SCHEMA
except ImportError:
    from prepare_metadata import SCHEMA

SPLIT_NAMES = ("train", "validation", "test")


def _split_one_language(records, ratios, rng):
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        key = record.get("speaker_id") or f"record:{record['audio_id']}"
        groups.setdefault(key, []).append(record)
    keys = sorted(groups)
    rng.shuffle(keys)

    result: list[list[dict[str, str]]] = [[], [], []]
    total = len(records)
    targets = [ratio * total for ratio in ratios]

    # With few speakers, make sure validation and test each get one before train
    # takes the rest. Otherwise give each speaker to the split furthest below target.
    if 2 <= len(keys) <= 10:
        # Smallest speakers go to test and validation so train keeps most clips.
        keys.sort(key=lambda k: len(groups[k]))
        result[2].extend(groups[keys[0]])
        if len(keys) >= 3:
            result[1].extend(groups[keys[1]])
            remaining = keys[2:]
        else:
            remaining = keys[1:]
        for key in remaining:
            result[0].extend(groups[key])
        return result

    for key in keys:
        deficits = [targets[i] - len(result[i]) for i in range(3)]
        index = max(range(3), key=lambda i: deficits[i])
        result[index].extend(groups[key])
    return result


def split(records, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    """Return (train, validation, test) lists, speaker-disjoint and per language."""
    rng = random.Random(seed)
    by_language: dict[str, list[dict[str, str]]] = {}
    for record in records:
        by_language.setdefault(record.get("language", ""), []).append(record)
    output: list[list[dict[str, str]]] = [[], [], []]
    for language in sorted(by_language):
        parts = _split_one_language(by_language[language], ratios, rng)
        for index, part in enumerate(parts):
            for row in part:
                row["split"] = SPLIT_NAMES[index]
            output[index].extend(part)
    return output[0], output[1], output[2]


def _write(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    parts = split(records, args.seed)
    for name, rows in zip(SPLIT_NAMES, parts):
        _write(args.metadata.parent / f"{name}.csv", rows)
        print(f"{name}: {len(rows)}")
    # Keep the split column in the combined metadata so statistics are correct.
    _write(args.metadata, [row for part in parts for row in part])


if __name__ == "__main__":
    main()
