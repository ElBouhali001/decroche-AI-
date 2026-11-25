from pathlib import Path

import numpy as np

from evaluation.calculate_wer import corpus_wer, word_error_rate
from evaluation.run_wer import confusions, score_rows, summarize
from evaluation.text_normalizer import align, normalize
from scripts.augment_telephony import telephone_channel
from scripts.split_dataset import split


def test_normalize_ignores_case_punctuation_fillers_and_number_style():
    assert normalize("Euh, je voudrais DEUX tables !", "fr") == normalize("je voudrais 2 tables", "fr")
    assert normalize("Um, it's twenty-one.", "en") == "it s 21"
    assert normalize("c'est", "fr") == normalize("c est", "fr")
    assert normalize("06 12 34 56 78", "fr") == normalize("zéro six douze trente-quatre cinquante-six soixante-dix-huit", "fr")


def test_alignment_counts():
    result = align("pour deux ce soir", "pour douze ce soir là")
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 1)
    assert result.wer == 0.5


def test_corpus_wer_weights_by_words():
    rows = [{"reference": "a b c d", "hypothesis": "a b c d"}, {"reference": "x", "hypothesis": "y"}]
    assert corpus_wer(rows) == 0.2
    assert word_error_rate("bonjour", "bonjour") == 0.0


def test_score_rows_with_fake_model():
    rows = [
        {"audio_id": "fr1", "audio_file": "a.wav", "language": "fr",
         "transcription": "Mon numéro est le 06 12 34 56 78", "entity_phone": "0612345678"},
        {"audio_id": "en1", "audio_file": "b.wav", "language": "en",
         "transcription": "My name is Rahul", "entity_name": "Rahul"},
    ]
    outputs = {"a.wav": "mon numéro est le zéro six douze trente quatre cinquante six soixante dix huit",
               "b.wav": "my name is raul"}

    def fake(path: Path, language, prompt):
        return outputs[path.name], 0.5, 2.0

    results = score_rows(rows, Path("."), fake)
    summary = {row["language"]: row for row in summarize(results, ["en", "fr"])}
    assert summary["fr"]["wer"] == 0.0 and summary["fr"]["phone_accuracy"] == 1.0
    assert summary["en"]["wer"] == 0.25 and summary["en"]["name_accuracy"] == 0.0
    assert confusions(results)[0]["reference"] == "rahul"


def test_split_is_speaker_disjoint_and_balanced():
    records = [{"audio_id": str(i), "speaker_id": f"s{i % 50}", "language": "fr" if i % 2 else "en"}
               for i in range(400)]
    train, val, test = split(records, seed=1)
    assert len(train) > 5 * len(test)
    for a, b in ((train, val), (train, test), (val, test)):
        assert not {r["speaker_id"] + r["language"] for r in a} & {r["speaker_id"] + r["language"] for r in b}
    assert {r["language"] for r in test} == {"fr", "en"}


def test_telephone_channel_removes_high_frequencies():
    rate = 16000
    t = np.arange(rate) / rate
    audio = (np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 6000 * t)).astype(np.float32) * 0.4
    out = telephone_channel(audio, rate)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / rate)
    assert spectrum[np.argmin(abs(freqs - 6000))] < 0.01 * spectrum[np.argmin(abs(freqs - 1000))]
