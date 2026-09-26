import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import TARGET_SAMPLE_RATE


@pytest.fixture
def audio_dir(tmp_path):
    return tmp_path


@pytest.fixture
def make_wav(tmp_path):
    def build(name, samples, sr=TARGET_SAMPLE_RATE):
        path = tmp_path / name
        sf.write(path, samples, sr)
        return str(path)

    return build


@pytest.fixture
def noise_samples():
    rng = np.random.default_rng(7)
    return rng.normal(0, 0.2, TARGET_SAMPLE_RATE * 2).astype("float32")


@pytest.fixture
def settings():
    return {
        "min_confidence": 0.6,
        "top_two_margin": 0.15,
        "max_confidence_difference": 0.3,
        "overlap_confidence": 0.25,
        "repeated_detection_window_sec": 30,
    }


@pytest.fixture
def good_quality():
    return {"quality": "Good", "notes": "fine", "noise_level": 0.1}
