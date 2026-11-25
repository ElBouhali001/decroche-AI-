"""Faster-Whisper speech-to-text adapter."""

from dataclasses import dataclass
from pathlib import Path
import time
import tempfile
from functools import lru_cache

from .language_detection import LanguageResult, normalize_language


@dataclass(frozen=True)
class Transcription:
    text: str
    language: LanguageResult
    processing_seconds: float
    duration_seconds: float | None = None


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "default"):
        self.model = self._get_model(model_size, device, compute_type)

    @staticmethod
    @lru_cache(maxsize=2)
    def _get_model(model_size: str, device: str, compute_type: str):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to use speech recognition.") from exc
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> Transcription:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        if path.stat().st_size < 100:
            raise RuntimeError("No usable audio was recorded. Please record again and speak for at least two seconds.")
        started = time.perf_counter()
        converted_path: Path | None = None
        try:
            import av
            import numpy as np
            transcription_source = str(path)

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted:
                    converted_path = Path(converted.name)

                source = av.open(str(path))
                stream = source.streams.audio[0]
                audio_chunks = []
                sample_rate = 16000

                for frame in source.decode(stream):
                    frame = frame.reformat(format="s16", layout="mono", sample_rate=sample_rate)
                    array = frame.to_ndarray()
                    audio_chunks.append(array.reshape(-1))

                source.close()
                if not audio_chunks:
                    raise RuntimeError("No audio samples were found")

                audio = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
                if len(audio) < sample_rate:
                    raise RuntimeError("The recording is too short")
                import soundfile as sf
                sf.write(str(converted_path), audio, sample_rate, subtype="PCM_16")
                transcription_source = str(converted_path)
            except Exception:
                transcription_source = str(path)

            original_av_open = av.open

            def compatible_av_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return original_av_open(*args, **kwargs)

            av.open = compatible_av_open
            try:
                segments, info = self.model.transcribe(
                    transcription_source,
                    task="transcribe",
                    language=language,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                    vad_filter=True,
                    beam_size=5,
                )
            finally:
                av.open = original_av_open
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                try:
                    segments, info = self.model.transcribe(
                        transcription_source,
                        task="transcribe",
                        language=language,
                        condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                        vad_filter=False,
                        beam_size=5,
                        best_of=5,
                    )
                    text = " ".join(segment.text.strip() for segment in segments).strip()
                except Exception:
                    text = ""
        except Exception as exc:
            raise RuntimeError(
                f"Recording transcription failed ({type(exc).__name__}: {exc}). "
                "Please allow microphone access and record again."
            ) from exc
        finally:
            if converted_path is not None:
                converted_path.unlink(missing_ok=True)
        if not text:
            raise RuntimeError("No speech was detected in the recording. Please move closer to the microphone and try again.")
        elapsed = time.perf_counter() - started
        language = normalize_language(info.language, float(info.language_probability))
        duration = getattr(info, "duration", None)
        return Transcription(text, language, elapsed, duration)
