"""Make clean recordings sound like a phone call.

Common Voice and FLEURS are recorded on laptops and phones at 16-48 kHz.
Real calls arrive at 8 kHz, cut to the 300-3400 Hz band and squeezed through
the G.711 mu-law codec, usually with some background noise. Training and
testing only on clean audio hides that gap, so this script recreates it.

Usage:
    python -m scripts.augment_telephony --metadata dataset/train.csv \
        --output-dir dataset/telephone --copies 2 --seed 42

The output CSV has the same columns as the input, with audio_file pointing
at the new 16 kHz wav (Whisper's input rate) and recording_type suffixed
with "+telephone". Nothing is invented: transcripts are copied as they are.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

TARGET_RATE = 16000
PHONE_RATE = 8000


def load_audio(path: str | Path, sample_rate: int = TARGET_RATE) -> np.ndarray:
    """Decode any audio file to mono float32 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode audio.")
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = TARGET_RATE) -> None:
    from scipy.io import wavfile
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(str(path), sample_rate, (clipped * 32767).astype(np.int16))


def _mu_law(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """Encode and decode with 8-bit mu-law, the G.711 codec used on phone lines."""
    compressed = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    quantised = np.round((compressed + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(quantised) * ((1 + mu) ** np.abs(quantised) - 1) / mu


def telephone_channel(
    audio: np.ndarray,
    sample_rate: int = TARGET_RATE,
    snr_db: float | None = None,
    noise: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return audio at sample_rate that went through a simulated phone line."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    rng = rng or np.random.default_rng(0)
    peak = float(np.max(np.abs(audio))) or 1.0
    signal = audio / peak * 0.9

    if snr_db is not None:
        if noise is None or len(noise) == 0:
            noise = rng.normal(0, 1, len(signal)).astype(np.float32)
        else:
            reps = int(np.ceil(len(signal) / len(noise)))
            offset = int(rng.integers(0, len(noise)))
            noise = np.roll(np.tile(noise, reps), -offset)[: len(signal)]
        sig_power = float(np.mean(signal ** 2)) or 1e-10
        noise_power = float(np.mean(noise ** 2)) or 1e-10
        signal = signal + noise * np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))

    narrow = resample_poly(signal, PHONE_RATE, sample_rate)
    sos = butter(4, [300, 3400], btype="bandpass", fs=PHONE_RATE, output="sos")
    narrow = sosfiltfilt(sos, narrow)
    narrow = narrow / (np.max(np.abs(narrow)) or 1.0) * 0.9
    narrow = _mu_law(narrow)
    return resample_poly(narrow, sample_rate, PHONE_RATE).astype(np.float32)


def augment_file(source: Path, target: Path, snr_db: float | None, noise: np.ndarray | None, seed: int) -> None:
    audio = load_audio(source)
    processed = telephone_channel(audio, TARGET_RATE, snr_db, noise, np.random.default_rng(seed))
    target.parent.mkdir(parents=True, exist_ok=True)
    save_wav(target, processed)


def _snr(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1, help="augmented versions per clip")
    parser.add_argument("--snr", type=_snr, nargs="+", default=[None, 20, 10],
                        help="SNR values (dB) to sample from; 'None' means no added noise")
    parser.add_argument("--noise-dir", type=Path, help="folder of background noise files (wav/mp3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    noises = [load_audio(p) for p in sorted(args.noise_dir.glob("*")) if p.is_file()] if args.noise_dir else []
    root = args.metadata.parent.parent
    with args.metadata.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    out_rows = []
    for row in rows:
        source = root / row["audio_file"]
        for copy in range(args.copies):
            snr = rng.choice(args.snr)
            noise = rng.choice(noises) if noises and snr is not None else None
            name = f"{Path(row['audio_file']).stem}_tel{copy}.wav"
            target = args.output_dir / row.get("language", "xx") / name
            augment_file(source, target, snr, noise, rng.randrange(1 << 30))
            new = dict(row)
            new["audio_id"] = f"{row['audio_id']}_tel{copy}"
            new["audio_file"] = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
            new["recording_type"] = f"{row.get('recording_type') or 'human'}+telephone"
            new["sample_rate"] = str(TARGET_RATE)
            new["scenario"] = f"telephone snr={snr}"
            out_rows.append(new)

    out_csv = args.output_dir / f"{args.metadata.stem}_telephone.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} augmented clips and {out_csv}")


if __name__ == "__main__":
    main()
