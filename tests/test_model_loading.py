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


def test_every_configured_model_declares_a_kind():
    from config.config import PYTHON_MODELS

    for name, settings in PYTHON_MODELS.items():
        assert settings["kind"] in (
            "features", "spectrogram", "embedding", "embedding_sklearn",
        ), name


def test_spectrogram_models_declare_no_scaler():
    from config.config import PYTHON_MODELS

    for name, settings in PYTHON_MODELS.items():
        if settings["kind"] in ("spectrogram", "embedding"):
            assert settings.get("scaler_file") is None, name


def test_model_paths_handle_a_missing_scaler(monkeypatch):
    import config.config as config
    from src import python_model

    monkeypatch.setattr(config, "ACTIVE_PYTHON_MODEL", "cnn")
    monkeypatch.setattr(python_model, "ACTIVE_PYTHON_MODEL", "cnn")

    paths = python_model.model_paths()
    assert paths["scaler"] is None
    assert "cnn_model.keras" in str(paths["model"])


def test_status_of_an_untrained_spectrogram_model(monkeypatch):
    import config.config as config
    from src import python_model

    monkeypatch.setattr(config, "ACTIVE_PYTHON_MODEL", "cnn")
    monkeypatch.setattr(python_model, "ACTIVE_PYTHON_MODEL", "cnn")

    status = python_model.model_status()
    # No CNN is trained yet, so this must report cleanly rather than raise.
    assert status["available"] is False
    assert "cnn" in status["message"]


def test_yamnet_is_configured_as_an_embedding_model():
    from config.config import PYTHON_MODELS

    settings = PYTHON_MODELS["yamnet"]
    assert settings["kind"] == "embedding"
    assert settings["model_file"].endswith(".keras")
    assert settings.get("scaler_file") is None


def test_yamnet_paths_follow_the_config(monkeypatch):
    import config.config as config
    from src import python_model

    monkeypatch.setattr(config, "ACTIVE_PYTHON_MODEL", "yamnet")
    monkeypatch.setattr(python_model, "ACTIVE_PYTHON_MODEL", "yamnet")

    paths = python_model.model_paths()
    assert paths["scaler"] is None
    assert "yamnet_model.keras" in str(paths["model"])
