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
    parser.add_argument("--model", default=settings.whisper_model,
                        help="tiny, base, small, large-v3-turbo, or a CTranslate2 model folder")
    args = parser.parse_args()
    agent = VoiceAgent(SpeechToText(args.model, settings.whisper_device, settings.whisper_compute_type), RestaurantAssistant(), TextToSpeech())
    print(json.dumps(agent.process(args.audio, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
