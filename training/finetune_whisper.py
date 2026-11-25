"""LoRA fine-tuning of Whisper on Décroche metadata CSVs.

Run this only after evaluation/run_wer.py has shown that the base model misses
the targets (see training/README.md). The script:

  * reads one or more metadata CSVs (same columns as dataset/train.csv)
  * passes a share of the training clips through the 8 kHz phone channel on
    the fly (--telephone-prob), so no augmented copies are needed on disk
  * trains LoRA adapters on the attention and MLP layers, base weights frozen
  * evaluates corpus WER on the validation CSV every --eval-steps with the
    same normaliser as run_wer, and keeps the best checkpoint

    python -m training.finetune_whisper \
        --base-model openai/whisper-small \
        --train dataset/train.csv dataset/synthetic/train.csv \
        --validation dataset/validation.csv \
        --output-dir models/whisper-small-decroche-lora

Afterwards: python -m training.export_model (merges the adapter and converts
it to CTranslate2 so backend/stt.py can load it).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANG_NAMES = {"fr": "french", "en": "english"}


def read_rows(paths: list[Path], audio_root: Path, max_per_language: int | None, seed: int) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("transcription", "").strip() or not row.get("audio_file"):
                    continue
                audio = Path(row["audio_file"])
                row["_path"] = str(audio if audio.is_absolute() else audio_root / audio)
                rows.append(row)
    if max_per_language:
        rng = random.Random(seed)
        rng.shuffle(rows)
        kept, counts = [], {}
        for row in rows:
            language = row.get("language", "fr")
            if counts.get(language, 0) < max_per_language:
                kept.append(row)
                counts[language] = counts.get(language, 0) + 1
        rows = kept
    return rows


class SpeechDataset:
    """Loads audio lazily, applies the phone channel with some probability."""

    def __init__(self, rows, processor, telephone_prob: float = 0.0, seed: int = 0, max_seconds: float = 30.0):
        self.rows = rows
        self.processor = processor
        self.telephone_prob = telephone_prob
        self.rng = random.Random(seed)
        self.max_samples = int(max_seconds * 16000)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        import numpy as np
        from scripts.augment_telephony import load_audio, telephone_channel

        row = self.rows[index]
        audio = load_audio(row["_path"])[: self.max_samples]
        if self.telephone_prob and self.rng.random() < self.telephone_prob:
            snr = self.rng.choice([None, 25, 15, 10])
            audio = telephone_channel(audio, 16000, snr, None, np.random.default_rng(self.rng.randrange(1 << 30)))
        features = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]

        language = LANG_NAMES.get(row.get("language", "fr"), "french")
        tokenizer = self.processor.tokenizer
        tokenizer.set_prefix_tokens(language=language, task="transcribe")
        labels = tokenizer(row["transcription"].strip()).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = [{"input_ids": f["labels"]} for f in features]
        padded = self.processor.tokenizer.pad(labels, return_tensors="pt")
        label_ids = padded["input_ids"].masked_fill(padded.attention_mask.ne(1), -100)
        # The model adds the start token itself; drop it if the tokenizer already did.
        if (label_ids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            label_ids = label_ids[:, 1:]
        batch["labels"] = label_ids
        return batch


def build_compute_metrics(processor, validation_rows):
    from evaluation.text_normalizer import align, normalize

    languages = [row.get("language", "fr") for row in validation_rows]

    def compute(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        totals = {}
        for ref, hyp, lang in zip(refs, hyps, languages):
            a = align(normalize(ref, lang), normalize(hyp, lang))
            errors, words = totals.get(lang, (0, 0))
            totals[lang] = (errors + a.errors, words + a.reference_words)
        metrics = {f"wer_{lang}": e / w for lang, (e, w) in totals.items() if w}
        all_e = sum(e for e, _ in totals.values())
        all_w = sum(w for _, w in totals.values())
        metrics["wer"] = all_e / all_w if all_w else 0.0
        return metrics

    return compute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="openai/whisper-small",
                        help="e.g. openai/whisper-small, openai/whisper-large-v3-turbo")
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=Path("."), help="audio_file paths are relative to this")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-language", type=int, help="cap training clips per language to keep FR/EN balanced")
    parser.add_argument("--telephone-prob", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--load-in-8bit", action="store_true", help="needs bitsandbytes and a CUDA GPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    train_rows = read_rows(args.train, args.audio_root, args.max_per_language, args.seed)
    val_rows = read_rows([args.validation], args.audio_root, None, args.seed)
    if not train_rows or not val_rows:
        raise SystemExit("Training or validation set is empty. Check the CSV paths and audio_file columns.")
    counts = {}
    for row in train_rows:
        counts[row.get("language", "?")] = counts.get(row.get("language", "?"), 0) + 1
    print(f"train clips per language: {counts}; validation clips: {len(val_rows)}")

    processor = WhisperProcessor.from_pretrained(args.base_model)
    kwargs = {}
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"}
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, **kwargs)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"
    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=cuda,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2 if cuda else 0,
        report_to=[],
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=SpeechDataset(train_rows, processor, args.telephone_prob, args.seed),
        eval_dataset=SpeechDataset(val_rows, processor, 0.0, args.seed),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=build_compute_metrics(processor, val_rows),
    )
    trainer.train()
    best = args.output_dir / "best_adapter"
    trainer.model.save_pretrained(str(best))
    processor.save_pretrained(str(best))
    print(f"Best adapter saved to {best} (checkpoint {trainer.state.best_model_checkpoint}, "
          f"validation WER {trainer.state.best_metric})")


if __name__ == "__main__":
    main()
