"""Re-score saved hypotheses without running Whisper again.

Useful after a change to the normaliser or the number extractor, or to leave
some clips out (for example synthetic voices that turned out to be unusable).

    python -m evaluation.rescore experiments/results/hypotheses_synthetic_small_int8.csv \
        --metadata dataset/synthetic/metadata.csv --exclude-speakers "say:Bells" "say:Zarvox"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.run_wer import _entity_hits, summarize  # noqa: E402
from evaluation.text_normalizer import align, normalize  # noqa: E402


def rescore(hyp_rows, meta_rows, exclude_speakers=()):
    meta = {row["audio_id"]: row for row in meta_rows}
    results = []
    for row in hyp_rows:
        source = meta.get(row["audio_id"], {})
        if source.get("speaker_id") in exclude_speakers:
            continue
        language = row["language"]
        a = align(normalize(row["reference"], language), normalize(row["hypothesis"], language))
        results.append({
            "audio_id": row["audio_id"], "language": language,
            "reference": row["reference"], "hypothesis": row["hypothesis"],
            "substitutions": a.substitutions, "deletions": a.deletions, "insertions": a.insertions,
            "reference_words": a.reference_words, "wer": round(a.wer, 4),
            "processing_seconds": row.get("processing_seconds") or 0,
            "duration_seconds": row.get("duration_seconds", ""),
            **_entity_hits(source or row, row["hypothesis"], language),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hypotheses", type=Path)
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with entity columns and speaker_id")
    parser.add_argument("--exclude-speakers", nargs="*", default=[])
    args = parser.parse_args()
    hyp = list(csv.DictReader(args.hypotheses.open(encoding="utf-8")))
    meta = list(csv.DictReader(args.metadata.open(encoding="utf-8")))
    results = rescore(hyp, meta, set(args.exclude_speakers))
    for row in summarize(results, sorted({r["language"] for r in results})):
        print(f"{row['language']:>3}  clips {row['clips']:>3}  WER {row['wer']}  phone {row['phone_accuracy']} "
              f"({row['phone_clips']})  name {row['name_accuracy']}  city {row['city_accuracy']}")


if __name__ == "__main__":
    main()
