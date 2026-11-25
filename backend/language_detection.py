"""Language classification helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageResult:
    code: str
    name: str
    probability: float


def normalize_language(code: str | None, probability: float = 0.0) -> LanguageResult:
    """Map model language codes to the application's supported labels."""
    normalized = (code or "other").lower().replace("-", "_")
    if normalized.startswith("fr"):
        return LanguageResult("fr", "French", probability)
    if normalized.startswith("en"):
        return LanguageResult("en", "English", probability)
    return LanguageResult("other", "Other", probability)
