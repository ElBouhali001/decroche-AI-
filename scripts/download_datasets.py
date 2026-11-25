"""Download small, opt-in dataset samples with recorded provenance.

The local French archive can be imported with --french-archive. Hugging Face
downloads are intentionally opt-in and limited with --max-per-language.
"""

import argparse
import csv
import shutil
import subprocess
import tarfile
from pathlib import Path


def import_french_archive(archive: Path, root: Path, limit: int) -> Path:
    target = root / "french" / "audio"
    original = root / "french" / "original"
    target.mkdir(parents=True, exist_ok=True)
    original.mkdir(parents=True, exist_ok=True)
    source_tsv = root / "french" / "source.tsv"
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        tsv_name = next(name for name in members if name.endswith("ss-corpus-fr.tsv"))
        with bundle.extractfile(members[tsv_name]) as stream:
            assert stream is not None
            rows = list(csv.DictReader((line.decode("utf-8") for line in stream), delimiter="\t"))
        rows = [row for row in rows if row.get("transcription", "").strip()][:limit]
        for row in rows:
            row.update({
                "dataset_name": "Mozilla Common Voice Spontaneous Speech French",
                "dataset_source": "https://commonvoice.mozilla.org/",
                "dataset_license": "See archive README and Common Voice dataset terms",
            })
        for row in rows:
            member = next(m for name, m in members.items() if name.endswith("/audios/" + row["audio_file"]))
            original_path = original / row["audio_file"]
            with bundle.extractfile(member) as source, original_path.open("wb") as output:
                assert source is not None
                output.write(source.read())
            wav_path = target / (Path(row["audio_file"]).stem + ".wav")
            converter = shutil.which("afconvert") or shutil.which("ffmpeg")
            if not converter:
                raise RuntimeError("No audio converter found. Install ffmpeg or use macOS afconvert.")
            if Path(converter).name == "afconvert":
                subprocess.run([converter, "-f", "WAVE", "-d", "LEI16@16000", str(original_path), str(wav_path)], check=True)
            else:
                subprocess.run([converter, "-y", "-i", str(original_path), "-ar", "16000", "-ac", "1", str(wav_path)], check=True)
            row["audio_file"] = wav_path.name
        with source_tsv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]) + ["dataset_name", "dataset_source", "dataset_license"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return source_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--french-archive", type=Path)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--max-per-language", type=int, default=10)
    args = parser.parse_args()
    if not args.french_archive:
        print("No dataset selected. Use --french-archive PATH to import a local, licensed archive.")
        return
    source = import_french_archive(args.french_archive, args.root, args.max_per_language)
    print(f"Imported French source metadata: {source}")


if __name__ == "__main__":
    main()
