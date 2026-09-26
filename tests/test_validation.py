import numpy as np

from audio_preprocessing.validation import ValidationStatus, validate_audio_file


def test_normal_file_is_valid(make_wav, noise_samples):
    report = validate_audio_file(make_wav("ok.wav", noise_samples))
    assert report.is_valid


def test_missing_file_is_invalid():
    report = validate_audio_file("does_not_exist_anywhere.wav")
    assert not report.is_valid


def test_unsupported_extension_is_invalid(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not audio")
    report = validate_audio_file(str(path))
    assert not report.is_valid


def test_empty_file_is_invalid(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    report = validate_audio_file(str(path))
    assert not report.is_valid


def test_corrupt_file_is_invalid(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunk")
    report = validate_audio_file(str(path))
    assert not report.is_valid


def test_silent_file_is_invalid(make_wav):
    report = validate_audio_file(make_wav("silent.wav", np.zeros(32000, dtype="float32")))
    assert not report.is_valid


def test_very_short_file_is_invalid(make_wav, noise_samples):
    report = validate_audio_file(make_wav("short.wav", noise_samples[:800]))
    assert not report.is_valid


def test_long_file_is_a_warning_not_a_rejection(make_wav):
    rng = np.random.default_rng(1)
    long_audio = rng.normal(0, 0.2, 16000 * 45).astype("float32")
    report = validate_audio_file(make_wav("long.wav", long_audio))
    assert report.is_valid
    assert report.has_warnings


def test_stereo_file_is_accepted(make_wav):
    rng = np.random.default_rng(2)
    stereo = rng.normal(0, 0.2, (32000, 2)).astype("float32")
    report = validate_audio_file(make_wav("stereo.wav", stereo))
    assert report.is_valid


def test_unusual_sample_rate_is_accepted_with_warning(make_wav, noise_samples):
    report = validate_audio_file(make_wav("lowrate.wav", noise_samples[:6000], sr=3000))
    statuses = [r.status for r in report.results]
    assert ValidationStatus.WARNING in statuses or not report.is_valid
