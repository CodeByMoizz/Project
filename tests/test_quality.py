import numpy as np

from config.config import TARGET_SAMPLE_RATE
from src.quality import analyse_quality, detect_clipping, detect_silence


def test_silence_is_unusable():
    result = analyse_quality(np.zeros(TARGET_SAMPLE_RATE, dtype="float32"), TARGET_SAMPLE_RATE)
    assert result["quality"] == "Unusable"
    assert result["is_silent"]


def test_clean_tone_is_good():
    time = np.linspace(0, 2, TARGET_SAMPLE_RATE * 2, dtype="float32")
    tone = (0.5 * np.sin(2 * np.pi * 440 * time)).astype("float32")
    result = analyse_quality(tone, TARGET_SAMPLE_RATE)
    assert result["quality"] in ("Good", "Acceptable")


def test_clipped_audio_is_poor():
    rng = np.random.default_rng(9)
    clipped = np.clip(rng.normal(0, 5, TARGET_SAMPLE_RATE), -1, 1).astype("float32")
    result = analyse_quality(clipped, TARGET_SAMPLE_RATE)
    assert result["is_clipped"]
    assert result["quality"] == "Poor"


def test_very_short_audio_is_poor():
    rng = np.random.default_rng(10)
    result = analyse_quality(rng.normal(0, 0.2, 2000).astype("float32"), TARGET_SAMPLE_RATE)
    assert result["quality"] in ("Poor", "Unusable")


def test_empty_audio_does_not_raise():
    result = analyse_quality(np.array([], dtype="float32"), TARGET_SAMPLE_RATE)
    assert result["quality"] == "Unusable"


def test_noise_level_is_between_zero_and_one(noise_samples):
    result = analyse_quality(noise_samples, TARGET_SAMPLE_RATE)
    assert 0.0 <= result["noise_level"] <= 1.0


def test_detect_silence_on_quiet_audio():
    is_silent, rms = detect_silence(np.full(1000, 1e-6, dtype="float32"))
    assert is_silent
    assert rms < 0.001


def test_detect_clipping_on_clean_audio(noise_samples):
    is_clipped, ratio = detect_clipping(noise_samples * 0.3)
    assert not is_clipped
    assert ratio == 0.0


def test_quality_always_reports_a_note(noise_samples):
    assert analyse_quality(noise_samples, TARGET_SAMPLE_RATE)["notes"]


def test_a_steady_tone_is_not_counted_as_noise():
    # An alarm or a horn is steady but not spectrally flat, so it must not be
    # treated as background interference.
    time = np.linspace(0, 2, TARGET_SAMPLE_RATE * 2, dtype="float32")
    tone = (0.5 * np.sin(2 * np.pi * 440 * time)).astype("float32")
    assert analyse_quality(tone, TARGET_SAMPLE_RATE)["noise_level"] < 0.2


def test_a_transient_burst_is_not_counted_as_noise():
    # A gunshot is broadband but not steady, so it must not be treated as noise
    # either, otherwise its critical alert would be blocked.
    rng = np.random.default_rng(13)
    burst = np.concatenate([
        np.zeros(TARGET_SAMPLE_RATE // 2, dtype="float32"),
        (rng.normal(0, 0.8, TARGET_SAMPLE_RATE // 8) *
         np.exp(-np.linspace(0, 10, TARGET_SAMPLE_RATE // 8))).astype("float32"),
        np.zeros(TARGET_SAMPLE_RATE, dtype="float32"),
    ])
    result = analyse_quality(burst, TARGET_SAMPLE_RATE)
    assert result["noise_level"] < 0.2
    assert result["quality"] in ("Good", "Acceptable")


def test_steady_broadband_noise_is_counted_as_noise():
    rng = np.random.default_rng(14)
    hiss = rng.normal(0, 0.2, TARGET_SAMPLE_RATE * 2).astype("float32")
    assert analyse_quality(hiss, TARGET_SAMPLE_RATE)["noise_level"] > 0.3
