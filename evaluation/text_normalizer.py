"""Text normalisation and word alignment for ASR scoring.

Raw WER punishes things a caller would never notice: capital letters,
commas, "euh", or "deux" written as "2". Both the reference and the
hypothesis go through the same normalize() before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from backend.text_numbers import words_to_digits

FILLERS = {
    "fr": {"euh", "euhh", "euuh", "heu", "hum", "hmm", "mmh", "bah", "ben", "hein"},
    "en": {"uh", "um", "umm", "uhm", "er", "erm", "hmm", "mm", "mhm"},
}


def normalize(text: str, language: str = "fr", numbers_to_digits: bool = True) -> str:
    text = text or ""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)       # [rire], (bruit)
    text = text.replace("’", "'").replace("`", "'").replace("‘", "'")
    text = text.replace("œ", "oe").replace("Œ", "oe")
    text = text.lower()
    if numbers_to_digits:
        text = words_to_digits(text, language)
        # "06 12 34" and "061234" should count the same
        text = re.sub(r"(?<=\d)[\s.,\-]+(?=\d)", "", text)
    # Punctuation and apostrophes become spaces: "c'est" and "c est" score the same.
    text = re.sub(r"[^\w\s]", " ", text)
    fillers = FILLERS.get(language[:2], set())
    words = [w for w in text.split() if w not in fillers]
    return " ".join(words)


@dataclass(frozen=True)
class Alignment:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    pairs: tuple[tuple[str, str], ...]   # (ref, hyp); "" marks a deletion/insertion

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.reference_words


def align(reference: str, hypothesis: str) -> Alignment:
    """Levenshtein alignment on words, returning S/D/I counts and aligned pairs."""
    ref = reference.split()
    hyp = hypothesis.split()
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            cost[i][j] = min(sub, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    s = d = ins = 0
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            d += 1
            pairs.append((ref[i - 1], ""))
            i -= 1
        else:
            ins += 1
            pairs.append(("", hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return Alignment(s, d, ins, n, tuple(pairs))
