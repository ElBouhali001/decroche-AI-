from scripts.prepare_metadata import SCHEMA
from scripts.split_dataset import split


def test_metadata_schema_and_disjoint_splits():
    assert "dataset_license" in SCHEMA
    records = [
        {"audio_id": "1", "speaker_id": "a"},
        {"audio_id": "2", "speaker_id": "b"},
        {"audio_id": "3", "speaker_id": "c"},
        {"audio_id": "4", "speaker_id": "d"},
    ]
    partitions = split(records, seed=42)
    speakers = [set(row["speaker_id"] for row in partition) for partition in partitions]
    assert not speakers[0] & speakers[1]
    assert not speakers[0] & speakers[2]
    assert not speakers[1] & speakers[2]
