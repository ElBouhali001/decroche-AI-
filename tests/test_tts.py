import pytest

from backend.tts import TextToSpeech


def test_tts_requires_api_key():
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        TextToSpeech()
