"""Generate synthetic customer-support utterances, optionally voiced by TTS.

Public corpora almost never contain someone dictating a phone number, a SIM
problem or a name and city in one breath, so we write those sentences
ourselves and have several TTS voices read them. Every row is tagged
recording_type=synthetic and carries entity_phone / entity_name / entity_city,
so run_wer can score entity accuracy on them.

Synthetic audio is for TRAINING (and a separate synthetic test row in the
report). It must never be mixed into the human test set.

    # text only (review the sentences first)
    python -m scripts.generate_synthetic_calls --count 400 --tts none

    # voiced with the macOS voices installed on this Mac
    python -m scripts.generate_synthetic_calls --count 400 --tts say
"""

from __future__ import annotations

import argparse
import csv
import platform
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from .prepare_metadata import SCHEMA
except ImportError:
    from prepare_metadata import SCHEMA

ISSUES = {
    "fr": [
        "Ma carte SIM n'a plus de réseau depuis ce matin",
        "Je n'arrive plus à passer d'appels",
        "Mon internet mobile est très lent, la 4G ne marche pas",
        "Mon téléphone affiche code PUK, je suis bloqué",
        "Ma recharge n'apparaît pas sur mon solde",
        "Je ne reçois plus les appels entrants",
        "J'ai un problème avec ma facture, le forfait est trop cher",
        "La 5G ne fonctionne pas chez moi",
        "Euh, bonjour, en fait j'ai plus du tout de connexion",
        "Mes SMS ne partent plus depuis hier",
    ],
    "en": [
        "My SIM card has no network since this morning",
        "I can't make any outgoing calls",
        "My mobile data is really slow, the 4G is not working",
        "My phone is asking for a PUK code and it's locked",
        "My recharge is not showing on my balance",
        "I'm not receiving incoming calls anymore",
        "I have a problem with my bill, the plan is too expensive",
        "The 5G network is not working at my place",
        "Um, hi, so basically I have no signal at all",
        "My text messages are not going through since yesterday",
    ],
}
NUMBER_TEMPLATES = {
    "fr": ["Mon numéro est le {n}", "C'est le {n}", "Oui, alors, {n}", "Mon numéro de mobile, c'est {n}", "{n}"],
    "en": ["My number is {n}", "It's {n}", "Sure, it's {n}", "My mobile number is {n}", "{n}"],
}
NAME_TEMPLATES = {
    "fr": ["Je m'appelle {x}", "Mon nom est {x}", "C'est {x}", "{x}"],
    "en": ["My name is {x}", "It's {x}", "The account is under {x}", "{x}"],
}
CITY_TEMPLATES = {
    "fr": ["Je suis à {x}", "J'habite à {x}", "Je me trouve à {x}", "À {x}"],
    "en": ["I am in {x}", "I live in {x}", "I'm calling from {x}", "{x}"],
}
NAMES = {
    "fr": ["Pierre Martin", "Camille Dubois", "Karim Benali", "Sophie Laurent", "Yasmine El Idrissi",
           "Thomas Moreau", "Fatou Diallo", "Julien Lefèvre", "Inès Garcia", "Nicolas Petit"],
    "en": ["Rahul Sharma", "Emily Johnson", "James Smith", "Priya Patel", "Oliver Brown",
           "Sarah Wilson", "Mohammed Khan", "Grace Taylor", "Daniel Lee", "Chloe Evans"],
}
CITIES = {
    "fr": ["Paris", "Lyon", "Marseille", "Toulouse", "Lille", "Bordeaux", "Nantes", "Strasbourg",
           "Montpellier", "Rennes", "Courbevoie", "Nanterre"],
    "en": ["London", "Manchester", "Delhi", "Mumbai", "Bangalore", "Toronto", "Dublin", "Leeds",
           "Birmingham", "Glasgow"],
}


def random_phone(language: str, rng: random.Random) -> tuple[str, str]:
    """Return (spoken form, canonical digits)."""
    if language == "fr":
        digits = rng.choice("67") + "".join(rng.choice("0123456789") for _ in range(8))
        canonical = "0" + digits
        pairs = [canonical[i:i + 2] for i in range(0, 10, 2)]
        return " ".join(pairs), canonical
    canonical = rng.choice("6789") + "".join(rng.choice("0123456789") for _ in range(9))
    style = rng.random()
    if style < 0.4:
        spoken = f"{canonical[:5]} {canonical[5:]}"
    elif style < 0.7:
        spoken = " ".join(canonical)
    else:
        spoken = f"{canonical[:3]} {canonical[3:6]} {canonical[6:]}"
    return spoken, canonical


def make_utterances(language: str, count: int, rng: random.Random) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        kind = rng.choice(["issue", "number", "name", "city", "combined", "combined"])
        phone_spoken, phone = random_phone(language, rng)
        name, city = rng.choice(NAMES[language]), rng.choice(CITIES[language])
        entity = {"entity_phone": "", "entity_name": "", "entity_city": ""}
        if kind == "issue":
            text = rng.choice(ISSUES[language])
        elif kind == "number":
            text = rng.choice(NUMBER_TEMPLATES[language]).format(n=phone_spoken)
            entity["entity_phone"] = phone
        elif kind == "name":
            text = rng.choice(NAME_TEMPLATES[language]).format(x=name)
            entity["entity_name"] = name
        elif kind == "city":
            text = rng.choice(CITY_TEMPLATES[language]).format(x=city)
            entity["entity_city"] = city
        else:
            parts = [rng.choice(ISSUES[language]),
                     NUMBER_TEMPLATES[language][0].format(n=phone_spoken),
                     NAME_TEMPLATES[language][0].format(x=name),
                     CITY_TEMPLATES[language][0].format(x=city)]
            text = ". ".join(parts) + "."
            entity = {"entity_phone": phone, "entity_name": name, "entity_city": city}
        rows.append({"audio_id": f"syn_{language}_{index + 1:05d}", "transcription": text,
                     "language": language, "scenario": kind, **entity})
    return rows


# macOS novelty voices sing, whisper or play sound effects. They are useless as
# speech data and make the English results look worse than they are.
NOVELTY_VOICES = {
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Deranged", "Good News",
    "Hysterical", "Jester", "Junior", "Organ", "Pipe Organ", "Ralph", "Superstar", "Trinoids",
    "Whisper", "Wobble", "Zarvox", "Fred", "Kathy",
}


def say_voices(language: str) -> list[str]:
    if platform.system() != "Darwin" or not shutil.which("say"):
        return []
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    prefix = "fr_" if language == "fr" else "en_"
    voices = []
    for line in listing.splitlines():
        match = re.match(r"^(.+?)\s{2,}(\w\w_\w\w)", line)
        if match and match.group(2).startswith(prefix) and match.group(1).strip() not in NOVELTY_VOICES:
            voices.append(match.group(1).strip())
    return voices


def voice_with_say(text: str, voice: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        # afconvert ships with macOS, so no ffmpeg is needed here.
        if shutil.which("afconvert"):
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(target)], check=True)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(target)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=200, help="utterances per language")
    parser.add_argument("--languages", nargs="+", default=["fr", "en"])
    parser.add_argument("--tts", choices=["none", "say"], default="none")
    parser.add_argument("--root", type=Path, default=Path("dataset/synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_rows = []
    for language in args.languages:
        rows = make_utterances(language, args.count, rng)
        voices = say_voices(language) if args.tts == "say" else []
        if args.tts == "say" and not voices:
            raise SystemExit(f"No macOS 'say' voices found for {language}. Install some in System Settings > Accessibility > Spoken Content.")
        for row in rows:
            row.update({"language_name": "French" if language == "fr" else "English",
                        "dataset_name": "Décroche synthetic calls", "dataset_source": "generated",
                        "dataset_license": "project-internal", "recording_type": "synthetic",
                        "quality_status": "synthetic", "audio_format": "wav", "sample_rate": "16000",
                        "channels": "1", "audio_file": ""})
            if voices:
                voice = rng.choice(voices)
                target = args.root / ("french" if language == "fr" else "english") / "audio" / f"{row['audio_id']}.wav"
                voice_with_say(row["transcription"], voice, target)
                row["audio_file"] = str(target)
                row["speaker_id"] = f"say:{voice}"
        out = args.root / ("french" if language == "fr" else "english") / "metadata.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCHEMA, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        all_rows += rows
        print(f"{language}: {len(rows)} utterances -> {out} ({len(voices)} voices)")
    with (args.root / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
