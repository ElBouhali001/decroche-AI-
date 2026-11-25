"""Score ASR models on a test CSV: WER per language plus entity accuracy.

This is the baseline every fine-tuning run is compared against.

    python -m evaluation.run_wer --metadata dataset/test.csv --models base small large-v3-turbo
    python -m evaluation.run_wer --models ./models/decroche-whisper-ct2 --telephone
    python -m evaluation.run_wer --models small --prompt-file evaluation/prompts.yaml

Outputs (in --output-dir, default experiments/results/):
    hypotheses_<model>.csv   one row per clip: reference, hypothesis, S/D/I, entity hits
    confusions_<model>.csv   most frequent word substitutions, deletions, insertions
    summary.csv              one row per model x language (appended, never overwritten)

WER here is corpus-level (total errors / total reference words), not the mean
of per-clip WER, so long clips are not drowned out by short ones. French and
English are always reported separately; an "all" row is added for convenience
but should not be quoted alone.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.text_numbers import extract_mobile_number, strip_accents  # noqa: E402
from evaluation.text_normalizer import align, normalize  # noqa: E402

# A transcriber takes (audio_path, language or None, prompt or None) and returns
# (text, seconds_spent, audio_duration_or_None). Injectable so tests need no model.
Transcriber = Callable[[Path, str | None, str | None], tuple[str, float, float | None]]

SUMMARY_FIELDS = [
    "timestamp", "model", "language", "clips", "reference_words", "wer",
    "substitutions", "deletions", "insertions",
    "phone_accuracy", "phone_clips", "name_accuracy", "name_clips", "city_accuracy", "city_clips",
    "real_time_factor", "telephone", "language_mode", "prompt", "device", "compute_type",
]


def faster_whisper_transcriber(model: str, device: str = "auto", compute_type: str = "default",
                               vad: bool = True, vad_threshold: float | None = None) -> Transcriber:
    from backend.stt import SpeechToText
    stt = SpeechToText(model, device=device, compute_type=compute_type)

    def run(path: Path, language: str | None, prompt: str | None):
        result = stt.transcribe(path, language=language, initial_prompt=prompt, vad_filter=vad,
                                   vad_threshold=vad_threshold)
        return result.text, result.processing_seconds, result.duration_seconds

    return run


def _load_prompts(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    import yaml
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return {str(k): str(v) for k, v in data.items()}


def _entity_hits(row: dict[str, str], hypothesis: str, language: str) -> dict[str, str]:
    hits = {}
    phone = (row.get("entity_phone") or "").strip()
    if phone:
        hits["phone_ok"] = str(extract_mobile_number(hypothesis, language) == extract_mobile_number(phone, language))
    for field in ("name", "city"):
        value = (row.get(f"entity_{field}") or "").strip()
        if value:
            # Accents are ignored here: "Ines" for "Inès" still finds the right customer.
            wanted = strip_accents(normalize(value, language))
            hits[f"{field}_ok"] = str(wanted in strip_accents(normalize(hypothesis, language)))
    return hits


def score_rows(
    rows: list[dict[str, str]],
    root: Path,
    transcriber: Transcriber,
    language_mode: str = "forced",
    prompts: dict[str, str] | None = None,
    telephone: bool = False,
) -> list[dict[str, object]]:
    prompts = prompts or {}
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            language = row.get("language", "fr")
            path = Path(row["audio_file"])
            if not path.is_absolute() and not (root / path).exists() and path.exists():
                pass  # path is relative to the current folder (synthetic metadata)
            elif not path.is_absolute():
                path = root / path
            if telephone:
                from scripts.augment_telephony import augment_file
                phone_path = Path(tmp) / f"{row['audio_id']}.wav"
                augment_file(path, phone_path, snr_db=None, noise=None, seed=0)
                path = phone_path
            try:
                text, seconds, duration = transcriber(
                    path, language if language_mode == "forced" else None, prompts.get(language)
                )
                error = ""
            except RuntimeError as exc:   # SpeechToText raises when it hears nothing
                text, seconds, duration, error = "", 0.0, None, str(exc)
            ref = normalize(row.get("transcription", ""), language)
            hyp = normalize(text, language)
            alignment = align(ref, hyp)
            results.append({
                "audio_id": row["audio_id"],
                "language": language,
                "recording_type": row.get("recording_type", ""),
                "reference": row.get("transcription", ""),
                "hypothesis": text,
                "reference_normalized": ref,
                "hypothesis_normalized": hyp,
                "wer": round(alignment.wer, 4),
                "substitutions": alignment.substitutions,
                "deletions": alignment.deletions,
                "insertions": alignment.insertions,
                "reference_words": alignment.reference_words,
                "processing_seconds": round(seconds, 3),
                "duration_seconds": duration if duration is not None else "",
                "error": error,
                **_entity_hits(row, text, language),
                "_pairs": alignment.pairs,
            })
    return results


def summarize(results: list[dict[str, object]], languages: list[str]) -> list[dict[str, object]]:
    rows = []
    for language in languages + ["all"]:
        subset = [r for r in results if language == "all" or r["language"] == language]
        if not subset:
            continue
        words = sum(int(r["reference_words"]) for r in subset)
        s = sum(int(r["substitutions"]) for r in subset)
        d = sum(int(r["deletions"]) for r in subset)
        i = sum(int(r["insertions"]) for r in subset)
        audio = sum(float(r["duration_seconds"]) for r in subset if r["duration_seconds"] != "")
        spent = sum(float(r["processing_seconds"]) for r in subset)
        summary = {
            "language": language, "clips": len(subset), "reference_words": words,
            "wer": round((s + d + i) / words, 4) if words else "",
            "substitutions": s, "deletions": d, "insertions": i,
            "real_time_factor": round(spent / audio, 3) if audio else "",
        }
        for field in ("phone", "name", "city"):
            marks = [r[f"{field}_ok"] == "True" for r in subset if f"{field}_ok" in r]
            summary[f"{field}_accuracy"] = round(sum(marks) / len(marks), 4) if marks else ""
            summary[f"{field}_clips"] = len(marks)
        rows.append(summary)
    return rows


def confusions(results: list[dict[str, object]], top: int = 50) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for r in results:
        for ref, hyp in r["_pairs"]:
            if ref == hyp:
                continue
            kind = "deletion" if not hyp else "insertion" if not ref else "substitution"
            counts[(str(r["language"]), kind, ref, hyp)] += 1
    return [
        {"language": lang, "type": kind, "reference": ref, "hypothesis": hyp, "count": n}
        for (lang, kind, ref, hyp), n in counts.most_common(top)
    ]


def _write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            fields += [k for k in row if not k.startswith("_") and k not in fields]
    exists = append and path.exists()
    with path.open("a" if append else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _safe_name(model: str) -> str:
    return Path(model).name.replace("/", "_") or "model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--models", nargs="+", default=["base"],
                        help="faster-whisper sizes (tiny, base, small, large-v3-turbo) or a CTranslate2 folder")
    parser.add_argument("--language-mode", choices=["forced", "auto"], default="forced",
                        help="forced: pass the clip's language to Whisper; auto: let Whisper detect it")
    parser.add_argument("--prompt-file", type=Path, help="YAML with fr:/en: initial prompts (domain words)")
    parser.add_argument("--telephone", action="store_true", help="pass each clip through the 8 kHz phone channel first")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--vad-threshold", type=float,
                        help="speech probability needed to keep audio (faster-whisper default 0.5); lower keeps more")
    parser.add_argument("--no-vad", action="store_true",
                        help="turn off faster-whisper's voice activity filter (checks whether it cuts speech)")
    parser.add_argument("--compute-type", default="default",
                        help="int8 is usually fastest on a Mac CPU; default keeps the model's own type")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("transcription", "").strip()]
    if not rows:
        raise SystemExit(f"No transcribed rows in {args.metadata}")
    root = args.metadata.parent.parent
    prompts = _load_prompts(args.prompt_file)
    languages = sorted({r.get("language", "fr") for r in rows})
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    suffix = ("_tel" if args.telephone else "") + ("_novad" if args.no_vad else "") + (f"_vad{args.vad_threshold}" if args.vad_threshold is not None else "") + ("" if args.compute_type == "default" else f"_{args.compute_type}")

    for model in args.models:
        print(f"== {model} on {len(rows)} clips ({', '.join(languages)})")
        started = time.perf_counter()
        results = score_rows(rows, root, faster_whisper_transcriber(model, args.device, args.compute_type, not args.no_vad, args.vad_threshold), args.language_mode, prompts, args.telephone)
        prefix = "" if args.metadata.stem in ("test", "metadata") and args.metadata.parent.name == "dataset" else f"{args.metadata.parent.name}_"
        name = prefix + _safe_name(model) + suffix
        _write(args.output_dir / f"hypotheses_{name}.csv", results)
        _write(args.output_dir / f"confusions_{name}.csv", confusions(results))
        summary = summarize(results, languages)
        for row in summary:
            row.update({"timestamp": stamp, "model": model, "telephone": args.telephone,
                        "language_mode": args.language_mode, "prompt": bool(prompts),
                        "device": args.device, "compute_type": args.compute_type})
            print(f"   {row['language']:>3}  WER {row['wer']}  phone {'-' if row['phone_accuracy'] == '' else row['phone_accuracy']}  "
                  f"RTF {row['real_time_factor']}")
        _write(args.output_dir / "summary.csv", summary, SUMMARY_FIELDS, append=True)
        print(f"   done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
