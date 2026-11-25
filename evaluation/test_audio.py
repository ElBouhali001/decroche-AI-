"""Run STT against labelled test audio."""

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
    stt = SpeechToText(args.model)
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        result = stt.transcribe(root / row["audio_file"])
        print(row["audio_id"], result.language.code, result.text)


if __name__ == "__main__":
    main()
