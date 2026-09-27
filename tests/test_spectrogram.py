import numpy as np

from config.config import TARGET_SAMPLE_RATE
from feature_extraction.spectrogram import (
    FRAME_COUNT,
    INPUT_SHAPE,
    N_MELS,
    build_spectrogram,
    fit_frames,
    spectrogram_from_file,
)


def test_shape_is_the_declared_input_shape(noise_samples):
    assert build_spectrogram(noise_samples, TARGET_SAMPLE_RATE).shape == INPUT_SHAPE


def test_input_shape_matches_its_parts():
    assert INPUT_SHAPE == (N_MELS, FRAME_COUNT, 1)


def test_values_are_scaled_between_zero_and_one(noise_samples):
    spectrogram = build_spectrogram(noise_samples, TARGET_SAMPLE_RATE)
    assert spectrogram.min() >= 0.0
    assert spectrogram.max() <= 1.0


def test_values_are_finite(noise_samples):
    assert np.isfinite(build_spectrogram(noise_samples, TARGET_SAMPLE_RATE)).all()


def test_silence_does_not_raise():
    silence = np.zeros(TARGET_SAMPLE_RATE, dtype="float32")
    assert build_spectrogram(silence, TARGET_SAMPLE_RATE).shape == INPUT_SHAPE


def test_very_short_audio_is_padded():
    rng = np.random.default_rng(3)
    short = rng.normal(0, 0.2, 800).astype("float32")
    assert build_spectrogram(short, TARGET_SAMPLE_RATE).shape == INPUT_SHAPE


def test_long_audio_is_trimmed():
    rng = np.random.default_rng(4)
    long_audio = rng.normal(0, 0.2, TARGET_SAMPLE_RATE * 10).astype("float32")
    assert build_spectrogram(long_audio, TARGET_SAMPLE_RATE).shape == INPUT_SHAPE


def test_stereo_is_reduced_to_mono():
    rng = np.random.default_rng(5)
    stereo = rng.normal(0, 0.2, (2, TARGET_SAMPLE_RATE * 3)).astype("float32")
    assert build_spectrogram(stereo, TARGET_SAMPLE_RATE).shape == INPUT_SHAPE


def test_same_audio_gives_the_same_spectrogram(noise_samples):
    first = build_spectrogram(noise_samples, TARGET_SAMPLE_RATE)
    second = build_spectrogram(noise_samples, TARGET_SAMPLE_RATE)
    assert np.array_equal(first, second)


def test_fit_frames_pads_and_trims():
    assert fit_frames(np.zeros((N_MELS, 5))).shape == (N_MELS, FRAME_COUNT)
    assert fit_frames(np.zeros((N_MELS, FRAME_COUNT * 2))).shape == (N_MELS, FRAME_COUNT)


def test_from_file_matches_from_samples(make_wav, noise_samples):
    from audio_preprocessing.preprocessing import prepare_segments

    path = make_wav("spec.wav", noise_samples)
    segments, sr, _ = prepare_segments(path)

    assert np.array_equal(spectrogram_from_file(path), build_spectrogram(segments[0], sr))
