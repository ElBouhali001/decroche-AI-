# Dataset

Audio is not committed to git. Use `scripts/download_datasets.py` to import a
small sample from a local archive or an approved Hugging Face dataset, then run
`scripts/prepare_metadata.py`, `scripts/split_dataset.py`, and
`scripts/generate_statistics.py`.

Metadata preserves source name, URL, license note, speaker identifier (when
provided), transcript, format, and measured audio properties. Empty fields mean
the source did not provide the value; values are never invented.
