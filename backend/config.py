"""Environment-backed configuration for the voice agent."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    elevenlabs_api_key: str | None = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_fr_voice_id: str = os.getenv("ELEVENLABS_FR_VOICE_ID", "")
    elevenlabs_en_voice_id: str = os.getenv("ELEVENLABS_EN_VOICE_ID", "")


settings = Settings()
