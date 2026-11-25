# Décroche AI

Décroche AI is a bilingual French/English telecom customer-support telephone
prototype. A customer can describe a mobile, SIM, network, data, recharge, or
calling problem. The agent asks for the account number, name, and location,
provides the next diagnostic step, and creates a demo support case reference.

The project uses Faster-Whisper for speech-to-text, deterministic Python
support logic without API credentials, optional OpenAI answers, and ElevenLabs
or macOS local text-to-speech.

## Setup locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add API keys and ElevenLabs voice IDs to `.env` only when optional external
features are needed. Local support assistance works without API keys.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. Choose French or English in the sidebar. You can
record audio or use the text test field.

Example French call:

```text
Ma carte SIM n'a plus de réseau
Mon numéro est 9876543210
Je m'appelle Pierre Martin
Je suis à Paris
```

Example English call:

```text
I have no mobile network
My number is 9876543210
My name is Rahul Sharma
I am in Delhi
```

## Voice prototype

```bash
python -m backend.main path/to/input.wav --output output/response.mp3
```

The local support flow does not claim to repair a real network or modify a real
SIM account. It presents a simulated diagnostic and support case.

## Run tests

```bash
python -m pytest -q
```

## Speech recognition evaluation and fine-tuning

```bash
# WER per language + phone/name/city accuracy, on clean audio and on a simulated phone line
python -m evaluation.run_wer --metadata dataset/test.csv --models base small large-v3-turbo
python -m evaluation.run_wer --metadata dataset/test.csv --models small --telephone
```

Results go to `experiments/results/` (`summary.csv` keeps every run). The
fine-tuning workflow, data rules and the decision gate are in
`training/README.md`. `notebooks/finetune_check.ipynb` runs quick checks on
the same code.
