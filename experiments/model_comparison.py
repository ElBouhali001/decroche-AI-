"""Run the benchmark and save machine-readable experiment results."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.stt import SpeechToText


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/model_comparison.csv"))
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    root = args.metadata.parent.parent
    results = []
    for model in args.models:
        stt = SpeechToText(model)
        for row in rows:
            result = stt.transcribe(root / row["audio_file"])
            results.append({"model": model, "language": row["language"], "audio_id": row["audio_id"],
                            "detected_language": result.language.code, "processing_seconds": result.processing_seconds})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys() if results else ["model"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
