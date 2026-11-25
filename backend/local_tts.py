"""Local macOS speech fallback for dashboard demonstrations."""

from pathlib import Path
import platform
import subprocess
import tempfile


def synthesize_local(text: str, language: str, output_path: str | Path) -> Path:
    """Create a playable WAV response without requiring an API key."""
    if platform.system() != "Darwin":
        raise RuntimeError("Local voice output requires macOS; configure ElevenLabs for other systems.")
    voice = "Thomas" if language == "fr" else "Samantha"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        aiff = Path(temporary) / "response.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(destination)],
            check=True,
        )
    return destination
