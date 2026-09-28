import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import CLASSES
from src import yamnet_transfer


def three_clean_splits():
    count = len(CLASSES)
    labels = np.array(CLASSES * 3)
    splits = np.array(["train"] * count + ["validation"] * count + ["test"] * count)
    ids = np.array(
        [f"t{i}" for i in range(count)]
        + [f"v{i}" for i in range(count)]
        + [f"s{i}" for i in range(count)]
    )
    return labels, splits, ids


def test_clean_splits_are_accepted():
    labels, splits, ids = three_clean_splits()
    masks = yamnet_transfer.check_splits(labels, splits, ids)

    assert set(masks) == {"train", "validation", "test"}
    assert all(mask.sum() == len(CLASSES) for mask in masks.values())


@pytest.mark.parametrize("position, expected", [
    (len(CLASSES), "validation"),
    (2 * len(CLASSES), "test"),
])
def test_a_source_in_two_splits_is_rejected(position, expected):
    labels, splits, ids = three_clean_splits()
    ids[position] = "t0"

    with pytest.raises(ValueError, match="more than one split|both the train"):
        yamnet_transfer.check_splits(labels, splits, ids)


def test_a_missing_class_is_rejected():
    labels, splits, ids = three_clean_splits()
    labels[3] = labels[0]

    with pytest.raises(ValueError, match="missing"):
        yamnet_transfer.check_splits(labels, splits, ids)


def test_an_empty_split_is_rejected():
    labels, splits, ids = three_clean_splits()
    splits[splits == "test"] = "train"

    with pytest.raises(ValueError, match="no segments"):
        yamnet_transfer.check_splits(labels, splits, ids)


def write_features(path, version, width):
    np.savez_compressed(
        path,
        x=np.zeros((4, width), dtype=np.float32),
        y=np.array(CLASSES[:1] * 4),
        split=np.array(["train"] * 4),
        audio_id=np.array(["a"] * 4),
        feature_version=np.array(version),
    )


def test_a_stale_cache_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "features.npz"
    write_features(path, "yamnet1-something-older", yamnet_transfer.EMBEDDING_LENGTH)
    monkeypatch.setattr(yamnet_transfer, "FEATURES_FILE", path)

    with pytest.raises(ValueError, match="feature version"):
        yamnet_transfer.load_features()


def test_the_wrong_embedding_width_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "features.npz"
    write_features(path, yamnet_transfer.FEATURE_VERSION, 99)
    monkeypatch.setattr(yamnet_transfer, "FEATURES_FILE", path)

    with pytest.raises(ValueError, match="shape"):
        yamnet_transfer.load_features()


def test_a_missing_cache_asks_for_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(yamnet_transfer, "FEATURES_FILE", tmp_path / "nothing.npz")

    with pytest.raises(FileNotFoundError, match="--extract"):
        yamnet_transfer.load_features()


def test_the_mean_view_takes_only_the_mean_half():
    x = np.arange(2 * yamnet_transfer.EMBEDDING_LENGTH, dtype=np.float32)
    x = x.reshape(2, yamnet_transfer.EMBEDDING_LENGTH)

    assert yamnet_transfer.view_of(x, "mean").shape[1] == yamnet_transfer.EMBEDDING_MEAN_LENGTH
    assert yamnet_transfer.view_of(x, "combined").shape[1] == yamnet_transfer.EMBEDDING_LENGTH


def test_the_cache_key_follows_the_feature_version(monkeypatch):
    first = yamnet_transfer.cache_key("abc", 0)

    assert first != yamnet_transfer.cache_key("abc", 1)
    assert first != yamnet_transfer.cache_key("abd", 0)

    monkeypatch.setattr(yamnet_transfer, "FEATURE_VERSION", "something-else")
    assert first != yamnet_transfer.cache_key("abc", 0)


def test_every_candidate_is_named_once_and_has_a_known_view():
    names = []

    for name, view, model in yamnet_transfer.candidates():
        names.append(name)
        assert view in ("mean", "combined")
        assert hasattr(model, "fit") and hasattr(model, "predict_proba")

    assert len(names) == len(set(names))
