"""Build provenance-preserving metadata from audio and source TSV files."""

import argparse
import csv
import hashlib
from pathlib import Path

SCHEMA = [
    "audio_id", "audio_file", "transcription", "language", "language_name",
    "dataset_name", "dataset_source", "dataset_license", "speaker_id", "accent",
    "gender", "duration_seconds", "sample_rate", "channels", "audio_format",
    "recording_type", "scenario", "split", "quality_status",
    # Optional ground truth for entity accuracy (phone number, name, city).
    "entity_phone", "entity_name", "entity_city",
]


def _duration(path: Path) -> float | None:
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except (ImportError, RuntimeError, OSError):
        return None


def _source_rows(source_tsv: Path) -> dict[str, dict[str, str]]:
    with source_tsv.open(encoding="utf-8", newline="") as stream:
        sample = stream.readline()
        stream.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        return {row["audio_file"]: row for row in csv.DictReader(stream, delimiter=delimiter)}


def build_metadata(root: Path, directory_name: str, language: str, source_tsv: Path | None = None) -> list[dict[str, str]]:
    directory = root / directory_name / "audio"
    source = _source_rows(source_tsv) if source_tsv else {}
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        row = source.get(path.name, source.get(f"{path.stem}.mp3", {}))
        suffix = path.suffix.lower().lstrip(".")
        records.append({
            "audio_id": f"{language}_{len(records) + 1:05d}",
            "audio_file": str(path.relative_to(root.parent)),
            "transcription": row.get("transcription", ""),
            "language": language,
            "language_name": "French" if language == "fr" else "English",
            "dataset_name": row.get("dataset_name", "Unknown"),
            "dataset_source": row.get("dataset_source", ""),
            "dataset_license": row.get("dataset_license", ""),
            "speaker_id": row.get("client_id", row.get("speaker_id", "")),
            "accent": row.get("accents", row.get("accent", "")),
            "gender": row.get("gender", ""),
            "duration_seconds": row.get("duration_seconds", ""),
            "sample_rate": row.get("sample_rate", ""),
            "channels": row.get("channels", ""),
            "audio_format": suffix,
            "recording_type": row.get("recording_type", "human"),
            "scenario": row.get("scenario", ""),
            "split": row.get("split", ""),
            "quality_status": row.get("quality_status", "unverified"),
            "entity_phone": row.get("entity_phone", ""),
            "entity_name": row.get("entity_name", ""),
            "entity_city": row.get("entity_city", ""),
        })
    return records


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--french-source-tsv", type=Path)
    parser.add_argument("--english-source-tsv", type=Path)
    args = parser.parse_args()
    all_records: list[dict[str, str]] = []
    for language, source in (("french", args.french_source_tsv), ("english", args.english_source_tsv)):
        code = "fr" if language == "french" else "en"
        records = build_metadata(args.root, language, code, source)
        write_csv(args.root / language / "metadata.csv", records)
        all_records.extend(records)
    write_csv(args.root / "metadata.csv", all_records)
    print(f"Generated {len(all_records)} metadata records.")


if __name__ == "__main__":
    main()
