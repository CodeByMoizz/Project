import numpy as np

from src.audio_metadata import audio_fingerprint, extract_metadata, file_hash


def test_metadata_has_every_required_field(make_wav, noise_samples):
    metadata = extract_metadata(make_wav("meta.wav", noise_samples), "original.wav")

    for field in [
        "filename",
        "audio_format",
        "duration_sec",
        "sample_rate",
        "channels",
        "file_size",
        "uploaded_at",
        "file_hash",
    ]:
        assert field in metadata, field


def test_original_filename_is_kept(make_wav, noise_samples):
    metadata = extract_metadata(make_wav("stored.wav", noise_samples), "user_upload.wav")
    assert metadata["filename"] == "user_upload.wav"


def test_duration_and_rate_are_correct(make_wav, noise_samples):
    metadata = extract_metadata(make_wav("dur.wav", noise_samples))
    assert abs(metadata["duration_sec"] - 2.0) < 0.05
    assert metadata["sample_rate"] == 16000
    assert metadata["channels"] == 1


def test_identical_files_share_a_hash(make_wav, noise_samples):
    first = make_wav("one.wav", noise_samples)
    second = make_wav("two.wav", noise_samples)
    assert file_hash(first) == file_hash(second)


def test_different_files_differ_in_hash(make_wav, noise_samples):
    rng = np.random.default_rng(11)
    other = rng.normal(0, 0.2, 32000).astype("float32")
    assert file_hash(make_wav("a.wav", noise_samples)) != file_hash(make_wav("b.wav", other))


def test_fingerprint_survives_a_volume_change(noise_samples):
    original = audio_fingerprint(noise_samples, 16000)
    quieter = audio_fingerprint(noise_samples * 0.4, 16000)
    assert original == quieter


def test_fingerprint_of_silence_is_handled():
    assert audio_fingerprint(np.zeros(1000, dtype="float32"), 16000) is not None


def test_fingerprint_of_empty_audio_is_none():
    assert audio_fingerprint(np.array([], dtype="float32"), 16000) is None
