import io
import os

import numpy as np
import pytest
import soundfile as sf

import app as application
from database import database


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Each run gets its own database file so the tests never touch real events.
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    monkeypatch.setattr(application.database, "DB_PATH", test_db)
    database.init_db()

    application.app.config["TESTING"] = True
    test_client = application.app.test_client()

    test_client.post(
        "/register",
        data={"username": "tester", "password": "secret123", "role": "admin"},
        follow_redirects=True,
    )

    return test_client


@pytest.fixture
def wav_bytes():
    rng = np.random.default_rng(12)
    buffer = io.BytesIO()
    sf.write(buffer, rng.normal(0, 0.2, 32000).astype("float32"), 16000, format="WAV")
    buffer.seek(0)
    return buffer.read()


def test_pages_load(client):
    for path in ["/", "/live", "/event-history", "/manual-review", "/alerts", "/reports", "/admin"]:
        response = client.get(path)
        assert response.status_code == 200, path


def test_signed_out_user_gets_the_landing_page():
    application.app.config["TESTING"] = True
    anonymous = application.app.test_client()
    response = anonymous.get("/")
    assert response.status_code == 200
    assert b"lp-hero" in response.data


def test_signed_out_upload_is_redirected():
    application.app.config["TESTING"] = True
    anonymous = application.app.test_client()
    assert anonymous.post("/").status_code == 302


def test_unknown_page_gives_a_friendly_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert b"Traceback" not in response.data


def test_upload_without_a_file_is_reported(client):
    response = client.post("/", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Choose an audio file" in response.data


def test_upload_of_a_text_file_is_rejected(client):
    response = client.post(
        "/",
        data={"audio": (io.BytesIO(b"not audio"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"not a supported format" in response.data
    assert b"Traceback" not in response.data


def test_upload_of_a_corrupt_wav_is_rejected(client):
    response = client.post(
        "/",
        data={"audio": (io.BytesIO(b"RIFF\x00\x00\x00\x00WAVEjunk"), "broken.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Traceback" not in response.data


def test_upload_of_silence_is_rejected(client):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(32000, dtype="float32"), 16000, format="WAV")
    buffer.seek(0)

    response = client.post(
        "/",
        data={"audio": (buffer, "silent.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Traceback" not in response.data


def test_valid_upload_is_stored(client, wav_bytes):
    response = client.post(
        "/",
        data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Traceback" not in response.data
    assert database.get_statistics()["total_detections"] >= 1


def test_gtm_endpoint_rejects_empty_scores(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
                content_type="multipart/form-data")
    detection_id = database.get_recent_detections(1)[0]["detection_id"]

    response = client.post(f"/api/gtm/{detection_id}", json={"scores": {}})
    assert response.status_code == 400


def test_gtm_endpoint_rejects_a_missing_detection(client):
    response = client.post("/api/gtm/999999", json={"scores": {"Gunshot": 1.0}})
    assert response.status_code == 404


def test_gtm_scores_are_stored_and_compared(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
                content_type="multipart/form-data")
    detection_id = database.get_recent_detections(1)[0]["detection_id"]

    response = client.post(
        f"/api/gtm/{detection_id}",
        json={"scores": {"Background Noise": 0.8, "Gunshot": 0.2}},
    )
    assert response.status_code == 200

    row = database.get_detection(detection_id)
    assert row["gtm_class"] == "background_noise"
    assert row["gtm_model_version"]
    assert row["agreement_status"]


def test_gtm_endpoint_needs_a_signed_in_user():
    application.app.config["TESTING"] = True
    anonymous = application.app.test_client()
    assert anonymous.post("/api/gtm/1", json={"scores": {"Gunshot": 1.0}}).status_code == 401


def test_live_window_endpoint_needs_a_signed_in_user():
    application.app.config["TESTING"] = True
    anonymous = application.app.test_client()
    assert anonymous.post("/api/analyse-window").status_code == 401


def test_review_requires_a_valid_decision(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
                content_type="multipart/form-data")
    detection_id = database.get_recent_detections(1)[0]["detection_id"]

    response = client.post(
        "/manual-review",
        data={"detection_id": detection_id, "decision": "nonsense"},
    )
    assert b"valid decision" in response.data


def test_reviewer_decision_is_recorded(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
                content_type="multipart/form-data")
    detection_id = database.get_recent_detections(1)[0]["detection_id"]

    client.post(
        "/manual-review",
        data={
            "detection_id": detection_id,
            "decision": "corrected",
            "corrected_class": "vehicle_horn",
            "comments": "sounded like a horn",
        },
    )

    row = database.get_detection(detection_id)
    assert row["reviewer_decision"] == "corrected"
    assert row["status"] == "Reviewed"
    # The model outputs must survive a review.
    assert row["python_scores"]


def test_admin_can_change_a_threshold(client):
    client.post("/admin", data={"action": "settings", "min_confidence": "0.75"})
    assert database.get_settings()["min_confidence"] == 0.75


def test_admin_rejects_an_out_of_range_threshold(client):
    response = client.post("/admin", data={"action": "settings", "min_confidence": "5"})
    assert b"between 0 and 1" in response.data


def test_csv_export_works(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
                content_type="multipart/form-data")
    response = client.get("/export/detections.csv")
    assert response.status_code == 200
    assert b"detection_id" in response.data


def test_duplicate_upload_is_detected(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "first.wav")},
                content_type="multipart/form-data")
    response = client.post("/", data={"audio": (io.BytesIO(wav_bytes), "second.wav")},
                           content_type="multipart/form-data")
    assert b"duplicate" in response.data.lower()


def test_event_history_filter_does_not_break(client, wav_bytes):
    client.post("/", data={"audio": (io.BytesIO(wav_bytes), "event.wav")},
                content_type="multipart/form-data")
    response = client.get("/event-history?sound_class=gunshot&severity=Critical&min_confidence=0.5")
    assert response.status_code == 200
