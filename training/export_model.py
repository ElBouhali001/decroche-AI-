"""Merge a LoRA adapter into Whisper and convert it for faster-whisper.

    python -m training.export_model \
        --base-model openai/whisper-small \
        --adapter models/whisper-small-decroche-lora/best_adapter \
        --output models/decroche-whisper-small-ct2

Then point the app at it:  WHISPER_MODEL=models/decroche-whisper-small-ct2
and score it with:         python -m evaluation.run_wer --models models/decroche-whisper-small-ct2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="CTranslate2 folder for faster-whisper")
    parser.add_argument("--quantization", default="int8", help="int8 for CPU, float16 for GPU")
    parser.add_argument("--keep-merged", action="store_true", help="keep the merged Hugging Face copy")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    merged_dir = args.output.with_name(args.output.name + "-hf")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.save_pretrained(str(merged_dir))
    WhisperProcessor.from_pretrained(args.base_model).save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")

    converter = shutil.which("ct2-transformers-converter")
    if not converter:
        raise SystemExit("ct2-transformers-converter not found: pip install ctranslate2")
    subprocess.run([
        converter, "--model", str(merged_dir), "--output_dir", str(args.output),
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", args.quantization, "--force",
    ], check=True)
    print(f"CTranslate2 model written to {args.output}")
    if not args.keep_merged:
        shutil.rmtree(merged_dir)


if __name__ == "__main__":
    main()
