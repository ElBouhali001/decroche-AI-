# Fine-tuning the speech recogniser

Owner: Houssam ElMouedden

The app transcribes callers with faster-whisper (`backend/stt.py`, model set by
`WHISPER_MODEL`, `base` by default). This folder holds what we need to adapt
Whisper to telephone support calls in French and English, and the rule for
deciding whether it is worth doing.

## The rule: measure first

A fine-tuned model only counts if it beats a stronger off-the-shelf model on a
test set it never saw. So the order is fixed:

1. Score the stock models (`tiny`, `base`, `small`, `large-v3-turbo`) with
   `evaluation/run_wer.py`, on clean audio and with `--telephone`.
2. Try the cheap fixes: forced language, `--prompt-file evaluation/prompts.yaml`,
   number normalisation (already in `backend/text_numbers.py`).
3. Fine-tune only if the best result still misses the targets:
   under 12% WER on clean audio, under 18% on the phone channel, and phone
   numbers captured correctly in at least 95% of the clips that contain one.

French and English are always reported separately.

## Data

| Set | Source | Used for |
|---|---|---|
| Human test set | Calls recorded by the team over real phones, transcribed by hand, with `entity_phone`, `entity_name`, `entity_city` filled in | Final test only. Never trained on. |
| Public speech | Common Voice Scripted + Spontaneous Speech (FR, EN) from Mozilla Data Collective, FLEURS, VoxPopuli | Training and a public test split |
| Synthetic calls | `scripts/generate_synthetic_calls.py`, voiced with several TTS voices | Training, plus a separate synthetic report row |

Common Voice is no longer on Hugging Face. It is downloaded from Mozilla Data
Collective after accepting its terms; read the licence of each release before
training on it and record it in `dataset_license`.

Splits are speaker-disjoint and made per language (`scripts/split_dataset.py`,
80/10/10). Synthetic voices count as speakers, so the voices in the test split
are not heard during training.

## Commands

```bash
# 1. baseline
python -m evaluation.run_wer --metadata dataset/test.csv --models base small large-v3-turbo
python -m evaluation.run_wer --metadata dataset/test.csv --models base small large-v3-turbo --telephone
python -m evaluation.run_wer --metadata dataset/test.csv --models small --prompt-file evaluation/prompts.yaml

# 2. synthetic training data (macOS voices)
python -m scripts.generate_synthetic_calls --count 400 --tts say

# 3. LoRA fine-tuning (GPU machine: Colab, Kaggle or the school cluster)
pip install -r requirements.txt -r requirements-train.txt
python -m training.finetune_whisper --base-model openai/whisper-small \
    --train dataset/train.csv dataset/synthetic/metadata.csv \
    --validation dataset/validation.csv \
    --output-dir models/whisper-small-decroche-lora --max-per-language 3000

# 4. export for faster-whisper and score it like the others
python -m training.export_model --base-model openai/whisper-small \
    --adapter models/whisper-small-decroche-lora/best_adapter \
    --output models/decroche-whisper-small-ct2
python -m evaluation.run_wer --models models/decroche-whisper-small-ct2 --telephone
```

`experiments/results/summary.csv` keeps one line per model, language and
setting, so every run stays comparable.

## Things that go wrong

- Hallucination after about one epoch: the decoder starts repeating itself.
  Evaluation runs every 200 steps and the best checkpoint is kept. If the
  validation WER rises, stop early, lower the learning rate or the LoRA rank.
- Forgetting: if only telecom sentences are used, the model gets worse on
  everything else. Keep general speech in the training mix.
- Unbalanced languages: use `--max-per-language` so French does not crowd out English.
- Synthetic speech is too clean and too regular. It helps with vocabulary
  and number formats, but it is not evidence that the model works on people.
