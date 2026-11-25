"""Conversation state and voice-agent orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
import time

from .llm import TelecomSupportAssistant
from .stt import SpeechToText
from .tts import TextToSpeech


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


class VoiceAgent:
    def __init__(self, stt: SpeechToText, llm: TelecomSupportAssistant, tts: TextToSpeech):
        self.stt, self.llm, self.tts = stt, llm, tts
        self.conversation = Conversation()

    def process(self, audio_path: str | Path, output_path: str | Path) -> dict[str, object]:
        started = time.perf_counter()
        transcription = self.stt.transcribe(audio_path)
        self.conversation.add("user", transcription.text)
        response = self.llm.respond(self.conversation.messages, transcription.language.code)
        self.conversation.add("assistant", response.text)
        speech_path = self.tts.synthesize(response.text, response.language, output_path)
        return {
            "transcription": transcription.text,
            "detected_language": transcription.language.code,
            "language_probability": transcription.language.probability,
            "response": response.text,
            "speech_path": str(speech_path),
            "processing_seconds": time.perf_counter() - started,
        }
