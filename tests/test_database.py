import json

import pytest

from database import database


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "unit.db")
    database.init_db()
    return database


def test_settings_have_defaults(db):
    settings = db.get_settings()
    assert "min_confidence" in settings


def test_setting_round_trip(db):
    db.save_setting("min_confidence", 0.81)
    assert db.get_settings()["min_confidence"] == 0.81


def test_user_round_trip(db):
    user_id = db.add_user("someone", "hash", "reviewer", "A Name", "a@b.com")
    user = db.get_user(user_id)
    assert user["username"] == "someone"
    assert user["role"] == "reviewer"


def test_duplicate_username_is_rejected(db):
    db.add_user("twice", "hash", "user")
    with pytest.raises(Exception):
        db.add_user("twice", "hash", "user")


def test_invalid_role_is_rejected(db):
    with pytest.raises(Exception):
        db.add_user("badrole", "hash", "wizard")


def test_detection_round_trip(db):
    audio_id = db.add_audio_file({
        "filename": "a.wav", "stored_path": "/tmp/a.wav", "file_hash": "abc",
        "duration_sec": 2.0, "sample_rate": 16000, "channels": 1,
    })

    detection_id = db.add_detection({
        "audio_id": audio_id, "segment_index": 0,
        "python_class": "gunshot", "python_scores": json.dumps({"gunshot": 0.9}),
        "python_confidence": 0.9, "python_model_version": "rf-v1",
        "final_class": "gunshot", "severity": "Critical",
        "alert_status": "Alert Generated", "quality": "Good", "status": "Alert Generated",
    })

    row = db.get_detection(detection_id)
    assert row["python_class"] == "gunshot"
    assert row["filename"] == "a.wav"


def test_hash_lookup_finds_a_duplicate(db):
    db.add_audio_file({"filename": "a.wav", "stored_path": "/tmp/a.wav", "file_hash": "dup1"})
    assert db.find_audio_by_hash("dup1") is not None
    assert db.find_audio_by_hash("nothing") is None


def test_fingerprint_lookup(db):
    db.add_audio_file({
        "filename": "a.wav", "stored_path": "/tmp/a.wav",
        "file_hash": "h", "fingerprint": "1010",
    })
    assert db.find_audio_by_fingerprint("1010") is not None


def test_alert_lifecycle(db):
    audio_id = db.add_audio_file({"filename": "a.wav", "stored_path": "/tmp/a.wav", "file_hash": "h2"})
    detection_id = db.add_detection({"audio_id": audio_id, "final_class": "gunshot", "severity": "Critical"})

    alert_id = db.add_alert(detection_id, "Critical", "gunshot detected")
    assert len(db.get_alerts("active")) == 1

    user_id = db.add_user("op", "hash", "operator")
    db.update_alert_status(alert_id, "acknowledged", user_id)
    assert len(db.get_alerts("active")) == 0


def test_review_keeps_the_model_output(db):
    audio_id = db.add_audio_file({"filename": "a.wav", "stored_path": "/tmp/a.wav", "file_hash": "h3"})
    detection_id = db.add_detection({
        "audio_id": audio_id, "python_class": "gunshot",
        "python_scores": json.dumps({"gunshot": 0.9}), "final_class": "gunshot",
        "manual_review_required": 1,
    })

    reviewer_id = db.add_user("rev", "hash", "reviewer")
    db.add_review(detection_id, reviewer_id, "corrected", "vehicle_horn", "a horn", "none")

    row = db.get_detection(detection_id)
    assert row["python_class"] == "gunshot"
    assert row["reviewer_decision"] == "corrected"
    assert row["manual_review_required"] == 0


def test_review_queue_only_lists_open_items(db):
    audio_id = db.add_audio_file({"filename": "a.wav", "stored_path": "/tmp/a.wav", "file_hash": "h4"})
    db.add_detection({"audio_id": audio_id, "final_class": "unknown", "manual_review_required": 1})
    db.add_detection({"audio_id": audio_id, "final_class": "gunshot", "manual_review_required": 0})
    assert len(db.get_review_queue()) == 1


def test_audit_log_records_actions(db):
    user_id = db.add_user("aud", "hash", "admin")
    db.add_audit(user_id, "login", "ok")
    assert len(db.get_audit_log()) == 1
    assert db.count_recent_audit("login", 3600) == 1


def test_statistics_on_an_empty_database(db):
    stats = db.get_statistics()
    assert stats["total_detections"] == 0
    assert stats["average_confidence"] == 0.0


def test_search_filters_by_class(db):
    audio_id = db.add_audio_file({"filename": "a.wav", "stored_path": "/tmp/a.wav", "file_hash": "h5"})
    db.add_detection({"audio_id": audio_id, "final_class": "gunshot", "python_confidence": 0.9})
    db.add_detection({"audio_id": audio_id, "final_class": "vehicle_horn", "python_confidence": 0.7})

    assert len(db.search_detections({"sound_class": "gunshot"})) == 1
    assert len(db.search_detections({"min_confidence": 0.8})) == 1
    assert len(db.search_detections({})) == 2
