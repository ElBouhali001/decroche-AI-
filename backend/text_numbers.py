"""Spoken-number handling shared by the support agent and the ASR evaluation.

Whisper can write the same phone number as "0612345678", "06 12 34 56 78",
"06.12.34.56.78" or "zéro six douze trente-quatre cinquante-six soixante-dix-huit".
This module turns number words into digits and pulls out mobile numbers, so the
agent and the WER scorer both see one canonical form.
"""

from __future__ import annotations

import re
import unicodedata

FR_UNITS = {
    "zero": 0, "zéro": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
}
FR_TEENS = {
    "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14,
    "quinze": 15, "seize": 16,
}
FR_TENS = {"vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60}

EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
EN_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
EN_REPEAT = {"double": 2, "triple": 3}

_TOKEN = re.compile(r"[\w'’]+|[^\w\s]", re.UNICODE)


def _parse_french(words: list[str], start: int) -> tuple[str | None, int]:
    """Read one French number (0-99) starting at words[start]."""
    word = words[start]
    nxt = words[start + 1] if start + 1 < len(words) else None
    used = 1
    if word == "quatre" and nxt in ("vingt", "vingts"):
        value, used = 80, 2
    elif word in FR_TENS:
        value = FR_TENS[word]
    elif word in FR_TEENS:
        value = FR_TEENS[word]
        if word == "dix" and nxt in ("sept", "huit", "neuf"):
            return str(10 + FR_UNITS[nxt]), 2
        return str(value), 1
    elif word in FR_UNITS:
        return str(FR_UNITS[word]), 1
    else:
        return None, 0

    # value is a tens word (20..80). Look for the second half: "et un", "deux", "dix-sept"...
    index = start + used
    if index < len(words) and words[index] == "et" and index + 1 < len(words) and words[index + 1] in ("un", "onze"):
        index += 1
    if index < len(words):
        follow = words[index]
        if value in (60, 80) and follow in FR_TEENS:
            if follow == "dix" and index + 1 < len(words) and words[index + 1] in ("sept", "huit", "neuf"):
                return str(value + 10 + FR_UNITS[words[index + 1]]), index + 2 - start
            return str(value + FR_TEENS[follow]), index + 1 - start
        if follow in FR_UNITS and FR_UNITS[follow] > 0:
            return str(value + FR_UNITS[follow]), index + 1 - start
    return str(value), index - start


def _parse_english(words: list[str], start: int) -> tuple[str | None, int]:
    word = words[start]
    nxt = words[start + 1] if start + 1 < len(words) else None
    if word in EN_REPEAT and nxt in EN_UNITS:
        return str(EN_UNITS[nxt]) * EN_REPEAT[word], 2
    if word in EN_TENS:
        value = EN_TENS[word]
        if nxt in EN_UNITS and EN_UNITS[nxt] > 0:
            return str(value + EN_UNITS[nxt]), 2
        return str(value), 1
    if word in EN_TEENS:
        return str(EN_TEENS[word]), 1
    if word in EN_UNITS:
        return str(EN_UNITS[word]), 1
    return None, 0


def words_to_digits(text: str, language: str = "fr") -> str:
    """Replace number words with digits, keeping every other word as it was.

    Hyphenated compounds ("soixante-dix-huit", "twenty-one") are split first.
    "un"/"one" become "1" too; that is harmless for scoring because the same
    rule runs on both the reference and the hypothesis.
    """
    parser = _parse_french if language.startswith("fr") else _parse_english
    raw = text.replace("-", " ")
    tokens = _TOKEN.findall(raw)
    lowered = [token.lower() for token in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        digits, used = parser(lowered, i)
        if digits is None:
            out.append(tokens[i])
            i += 1
        else:
            out.append(digits)
            i += used
    text_out = " ".join(out)
    return re.sub(r"\s+([,.!?;:])", r"\1", text_out)


def _collapse_digit_runs(text: str) -> list[str]:
    """Join digit groups separated by spaces, dots or dashes into single strings."""
    runs = re.findall(r"\+?\d[\d\s.\-]*\d|\+?\d", text)
    return [re.sub(r"[\s.\-]", "", run) for run in runs]


def extract_mobile_number(text: str, language: str = "fr") -> str | None:
    """Return a mobile number in canonical form, or None.

    French: 06/07 numbers, also +33 6..., 0033 6... -> "0612345678".
    Indian (kept for the existing demo): 10 digits starting 6-9, optional +91.
    """
    candidates = _collapse_digit_runs(text)
    if language.startswith("fr") or not candidates:
        candidates += _collapse_digit_runs(words_to_digits(text, "fr"))
    candidates += _collapse_digit_runs(words_to_digits(text, "en"))
    for run in candidates:
        number = run.lstrip("+")
        if re.fullmatch(r"(?:33|0033)[67]\d{8}", number):
            return "0" + number[-9:]
        if re.fullmatch(r"0[67]\d{8}", number):
            return number
        if re.fullmatch(r"(?:91|0)?[6-9]\d{9}", number):
            return number[-10:]
    return None


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
