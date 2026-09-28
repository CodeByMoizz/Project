import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as application


def write_meta(labels):
    path = os.path.join("gtm_model", "metadata.json")
    os.makedirs("gtm_model", exist_ok=True)

    before = None
    if os.path.exists(path):
        before = open(path).read()

    with open(path, "w") as f:
        json.dump({"labels": labels, "version": "test"}, f)

    return path, before


def restore(path, before):
    if before is None:
        os.remove(path)
    else:
        with open(path, "w") as f:
            f.write(before)


def test_matching_labels_pass():
    names = application.encoder_classes()
    path, before = write_meta(names)
    try:
        assert application.gtm_label_check() is None
    finally:
        restore(path, before)


def test_wrong_order_fails():
    names = application.encoder_classes()
    swapped = list(names)
    swapped[0], swapped[1] = swapped[1], swapped[0]

    path, before = write_meta(swapped)
    try:
        assert application.gtm_label_check() is not None
    finally:
        restore(path, before)


def test_missing_label_fails():
    names = application.encoder_classes()[:-1]
    path, before = write_meta(names)
    try:
        assert application.gtm_label_check() is not None
    finally:
        restore(path, before)


def test_encoder_order_is_the_ten_classes():
    from config.config import CLASSES

    names = application.encoder_classes()
    assert len(names) == len(CLASSES)
    assert sorted(names) == sorted(CLASSES)
