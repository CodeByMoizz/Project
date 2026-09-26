import numpy as np

from config.config import TARGET_SAMPLE_RATE
from feature_extraction.features import (
    FEATURE_NAMES,
    FEATURE_VECTOR_LENGTH,
    extract_all_features,
    extract_features_from_samples,
    features_to_vector,
)


def test_vector_length_is_fixed(noise_samples):
    features = extract_features_from_samples(noise_samples, TARGET_SAMPLE_RATE)
    assert len(features_to_vector(features)) == FEATURE_VECTOR_LENGTH


def test_feature_names_are_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_column_order_is_stable(noise_samples):
    first = extract_features_from_samples(noise_samples, TARGET_SAMPLE_RATE)
    second = extract_features_from_samples(noise_samples, TARGET_SAMPLE_RATE)
    assert list(first.keys()) == list(second.keys()) == FEATURE_NAMES


def test_same_audio_gives_the_same_vector(noise_samples):
    first = features_to_vector(extract_features_from_samples(noise_samples, TARGET_SAMPLE_RATE))
    second = features_to_vector(extract_features_from_samples(noise_samples, TARGET_SAMPLE_RATE))
    assert first == second


def test_silence_does_not_raise():
    vector = features_to_vector(
        extract_features_from_samples(np.zeros(TARGET_SAMPLE_RATE, dtype="float32"))
    )
    assert len(vector) == FEATURE_VECTOR_LENGTH


def test_values_are_finite(noise_samples):
    vector = features_to_vector(extract_features_from_samples(noise_samples, TARGET_SAMPLE_RATE))
    assert np.isfinite(vector).all()


def test_stereo_input_is_reduced_to_mono():
    rng = np.random.default_rng(8)
    stereo = rng.normal(0, 0.2, (2, TARGET_SAMPLE_RATE)).astype("float32")
    vector = features_to_vector(extract_features_from_samples(stereo, TARGET_SAMPLE_RATE))
    assert len(vector) == FEATURE_VECTOR_LENGTH


def test_file_path_and_samples_agree(make_wav, noise_samples):
    # A file and its samples must produce the same vector, because training uses
    # files and live monitoring uses samples.
    path = make_wav("same.wav", noise_samples)
    from audio_preprocessing.preprocessing import prepare_segments

    segments, sr, _ = prepare_segments(path)

    from_file = features_to_vector(extract_all_features(path))
    from_samples = features_to_vector(extract_features_from_samples(segments[0], sr))

    assert from_file == from_samples
