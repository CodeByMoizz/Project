import numpy as np

from config.config import CLASSES, TARGET_SAMPLE_RATE
from src import python_model


def test_status_reports_missing_model_clearly():
    status = python_model.model_status()
    assert "available" in status
    assert status["message"]

    if not status["available"]:
        assert "not trained yet" in status["message"]


def test_prediction_returns_none_without_a_model(noise_samples):
    if python_model.model_status()["available"]:
        return

    assert python_model.get_prediction(noise_samples, TARGET_SAMPLE_RATE) is None


def test_empty_scores_cover_every_class():
    scores = python_model.empty_scores()
    assert sorted(scores.keys()) == sorted(CLASSES)
    assert set(scores.values()) == {0.0}


def test_prediction_on_silence_does_not_raise():
    silence = np.zeros(TARGET_SAMPLE_RATE, dtype="float32")
    python_model.get_prediction(silence, TARGET_SAMPLE_RATE)


def test_model_paths_follow_the_config():
    from config.config import ACTIVE_PYTHON_MODEL, PYTHON_MODELS

    paths = python_model.model_paths()
    assert PYTHON_MODELS[ACTIVE_PYTHON_MODEL]["model_file"] in str(paths["model"])
