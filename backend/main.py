"""File-based prototype entry point."""

import argparse
import json

from .config import settings
from .conversation import VoiceAgent
from .llm import RestaurantAssistant
from .stt import SpeechToText
from .tts import TextToSpeech


def main() -> None:
    parser = argparse.ArgumentParser(description="Process one WAV recording.")
    parser.add_argument("audio", help="Input WAV audio path")
    parser.add_argument("--output", default="output/response.mp3")
    parser.add_argument("--model", default=settings.whisper_model, choices=("tiny", "base", "small"))
    args = parser.parse_args()
    agent = VoiceAgent(SpeechToText(args.model), RestaurantAssistant(), TextToSpeech())
    print(json.dumps(agent.process(args.audio, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
