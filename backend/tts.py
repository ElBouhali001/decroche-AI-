"""ElevenLabs text-to-speech adapter."""

from pathlib import Path

from .config import settings


class TextToSpeech:
    def __init__(self, client=None):
        if client is None:
            if not settings.elevenlabs_api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is required for speech generation.")
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        self.client = client

    def synthesize(self, text: str, language: str, output_path: str | Path) -> Path:
        voice_id = settings.elevenlabs_fr_voice_id if language == "fr" else settings.elevenlabs_en_voice_id
        if not voice_id:
            raise RuntimeError(f"No ElevenLabs voice ID configured for language {language!r}.")
        audio = self.client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_multilingual_v2",
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            for chunk in audio:
                stream.write(chunk)
        return destination
