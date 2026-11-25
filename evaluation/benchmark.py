"""Benchmark Whisper model sizes on a metadata CSV."""

import argparse
import csv
import time
from pathlib import Path

from backend.stt import SpeechToText


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    args = parser.parse_args()
    root = args.metadata.parent.parent
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for model in args.models:
        recognizer = SpeechToText(model)
        for row in rows:
            started = time.perf_counter()
            result = recognizer.transcribe(root / row["audio_file"])
            elapsed = time.perf_counter() - started
            duration = result.duration_seconds or 0
            print({"model": model, "audio_id": row["audio_id"], "latency": elapsed,
                   "real_time_factor": elapsed / duration if duration else None,
                   "language": result.language.code})


if __name__ == "__main__":
    main()
