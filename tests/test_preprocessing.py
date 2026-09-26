import numpy as np

from audio_preprocessing.preprocessing import (
    convert_to_mono,
    normalize_audio,
    pad_or_truncate,
    prepare_segments,
    prepare_segments_from_samples,
    resample_audio,
    segment_audio,
)
from config.config import SEGMENT_DURATION_SEC, TARGET_SAMPLE_RATE

EXPECTED_LENGTH = int(SEGMENT_DURATION_SEC * TARGET_SAMPLE_RATE)


def test_stereo_becomes_mono():
    stereo = np.zeros((2, 100), dtype="float32")
    assert convert_to_mono(stereo).ndim == 1


def test_mono_stays_mono():
    mono = np.zeros(100, dtype="float32")
    assert convert_to_mono(mono).shape == (100,)


def test_resample_changes_length():
    samples = np.zeros(44100, dtype="float32")
    resampled, sr = resample_audio(samples, 44100, TARGET_SAMPLE_RATE)
    assert sr == TARGET_SAMPLE_RATE
    assert abs(len(resampled) - TARGET_SAMPLE_RATE) < 100


def test_normalize_keeps_silence_safe():
    silence = np.zeros(100, dtype="float32")
    assert np.max(np.abs(normalize_audio(silence))) == 0


def test_normalize_scales_to_one():
    samples = np.array([0.2, -0.4, 0.1], dtype="float32")
    assert abs(np.max(np.abs(normalize_audio(samples))) - 1.0) < 1e-6


def test_pad_short_segment():
    padded = pad_or_truncate(np.ones(100, dtype="float32"), TARGET_SAMPLE_RATE)
    assert len(padded) == EXPECTED_LENGTH


def test_truncate_long_segment():
    long_audio = np.ones(EXPECTED_LENGTH * 3, dtype="float32")
    assert len(pad_or_truncate(long_audio, TARGET_SAMPLE_RATE)) == EXPECTED_LENGTH


def test_pad_empty_segment():
    assert len(pad_or_truncate(np.array([], dtype="float32"), TARGET_SAMPLE_RATE)) == EXPECTED_LENGTH


def test_segmentation_splits_long_audio():
    samples = np.ones(TARGET_SAMPLE_RATE * 7, dtype="float32")
    segments, info = segment_audio(samples, TARGET_SAMPLE_RATE)
    assert len(segments) > 1
    assert len(segments) == len(info)


def test_segment_timestamps_increase():
    samples = np.ones(TARGET_SAMPLE_RATE * 7, dtype="float32")
    _, info = segment_audio(samples, TARGET_SAMPLE_RATE)
    starts = [start for start, _ in info]
    assert starts == sorted(starts)


def test_every_prepared_segment_has_the_fixed_length(noise_samples):
    segments, sr, _ = prepare_segments_from_samples(noise_samples, TARGET_SAMPLE_RATE)
    assert sr == TARGET_SAMPLE_RATE
    assert all(len(segment) == EXPECTED_LENGTH for segment in segments)


def test_prepare_segments_handles_stereo_and_resampling(make_wav):
    rng = np.random.default_rng(4)
    stereo = rng.normal(0, 0.2, (44100, 2)).astype("float32")
    segments, sr, _ = prepare_segments(make_wav("s.wav", stereo, sr=44100))
    assert sr == TARGET_SAMPLE_RATE
    assert all(len(segment) == EXPECTED_LENGTH for segment in segments)


def test_prepare_segments_caps_the_segment_count():
    rng = np.random.default_rng(5)
    very_long = rng.normal(0, 0.2, TARGET_SAMPLE_RATE * 700).astype("float32")
    segments, _, _ = prepare_segments_from_samples(very_long, TARGET_SAMPLE_RATE)
    assert len(segments) <= 300
