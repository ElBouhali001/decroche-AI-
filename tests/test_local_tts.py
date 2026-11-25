import platform

import pytest

from backend.local_tts import synthesize_local


def test_local_tts_rejects_unsupported_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="macOS"):
        synthesize_local("Hello", "en", tmp_path / "response.wav")
