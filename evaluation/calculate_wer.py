"""Calculate WER from a CSV with reference and hypothesis columns.

Both sides are normalised first (case, punctuation, fillers, number words),
and the result is corpus-level: total errors divided by total reference words.
An optional "language" column picks French or English normalisation.
"""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.text_normalizer import align, normalize  # noqa: E402


def word_error_rate(reference: str, hypothesis: str, language: str = "fr") -> float:
    return align(normalize(reference, language), normalize(hypothesis, language)).wer


def corpus_wer(rows: list[dict[str, str]]) -> float:
    errors = words = 0
    for row in rows:
        language = row.get("language", "fr") or "fr"
        result = align(normalize(row["reference"], language), normalize(row["hypothesis"], language))
        errors += result.errors
        words += result.reference_words
    return errors / words if words else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    with args.csv_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    print(round(corpus_wer(rows), 4))


if __name__ == "__main__":
    main()
