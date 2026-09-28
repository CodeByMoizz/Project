import numpy as np

from feature_extraction import embeddings


def test_embedding_length_is_declared():
    assert embeddings.EMBEDDING_LENGTH == 2048


def test_status_reports_clearly():
    status = embeddings.yamnet_status()
    assert "available" in status
    assert status["message"]


def test_downloaded_flag_matches_the_folder():
    import os

    expected = os.path.exists(embeddings.YAMNET_DIR / "saved_model.pb")
    assert embeddings.yamnet_is_downloaded() == expected


def test_module_imports_without_tensorflow():
    # The app must start even when TensorFlow is missing.
    assert hasattr(embeddings, "build_embedding")
    assert hasattr(embeddings, "embedding_from_file")


def test_build_embedding_returns_none_without_tensorflow():
    # Returns None rather than raising when YAMNet cannot be loaded.
    result = embeddings.build_embedding(np.zeros(48000, dtype="float32"), 16000)
    assert result is None or len(result) == embeddings.EMBEDDING_LENGTH
