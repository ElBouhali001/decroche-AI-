from backend.language_detection import normalize_language


def test_language_normalization():
    assert normalize_language("fr", 0.9).code == "fr"
    assert normalize_language("en-US").name == "English"
    assert normalize_language("de").code == "other"
