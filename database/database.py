import json
import os
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import DATABASE_FILE, DEFAULT_SETTINGS

DB_PATH = DATABASE_FILE


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
            CHECK (role IN ('user', 'reviewer', 'operator', 'maintenance', 'admin')),
        full_name TEXT,
        email TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audio_files (
        audio_id INTEGER PRIMARY KEY,
        uploaded_by INTEGER,
        filename TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        fingerprint TEXT,
        audio_format TEXT,
        duration_sec REAL,
        sample_rate INTEGER,
        channels INTEGER,
        bit_depth INTEGER,
        file_size INTEGER,
        source_kind TEXT NOT NULL DEFAULT 'upload',
        augmentation_type TEXT,
        uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (uploaded_by) REFERENCES users(user_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        detection_id INTEGER PRIMARY KEY,
        audio_id INTEGER,
        segment_index INTEGER NOT NULL DEFAULT 0,
        start_sec REAL,
        end_sec REAL,

        python_class TEXT,
        python_scores TEXT,
        python_confidence REAL,
        python_model_version TEXT,

        gtm_class TEXT,
        gtm_scores TEXT,
        gtm_confidence REAL,
        gtm_model_version TEXT,

        agreement_status TEXT,
        confidence_difference REAL,
        top_two_margin REAL,
        overlapping_classes TEXT,

        quality TEXT,
        quality_notes TEXT,
        noise_level REAL,

        final_class TEXT,
        confidence_level TEXT,
        severity TEXT,
        alert_status TEXT,
        recommended_action TEXT,
        manual_review_required INTEGER NOT NULL DEFAULT 0,
        manual_review_reason TEXT,
        status TEXT NOT NULL DEFAULT 'Classified',
        reviewer_decision TEXT,

        waveform_path TEXT,
        spectrogram_path TEXT,

        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id INTEGER PRIMARY KEY,
        detection_id INTEGER NOT NULL,
        severity TEXT NOT NULL,
        message TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        handled_by INTEGER,
        handled_at TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (detection_id) REFERENCES detections(detection_id),
        FOREIGN KEY (handled_by) REFERENCES users(user_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id INTEGER PRIMARY KEY,
        detection_id INTEGER NOT NULL,
        reviewer_id INTEGER,
        decision TEXT NOT NULL,
        corrected_class TEXT,
        comments TEXT,
        recommended_action TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (detection_id) REFERENCES detections(detection_id),
        FOREIGN KEY (reviewer_id) REFERENCES users(user_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        audit_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        action_type TEXT NOT NULL,
        details TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        name TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    # Indexes matter here: the SRS asks for 20,000 records with filtering.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(final_class)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_review ON detections(manual_review_required)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_hash ON audio_files(file_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")

    for name, value in DEFAULT_SETTINGS.items():
        cursor.execute(
            "INSERT OR IGNORE INTO settings (name, value) VALUES (?, ?)",
            (name, json.dumps(value)),
        )

    connection.commit()
    connection.close()


def get_settings():
    connection = get_connection()
    rows = connection.execute("SELECT name, value FROM settings").fetchall()
    connection.close()

    settings = dict(DEFAULT_SETTINGS)
    for row in rows:
        try:
            settings[row["name"]] = json.loads(row["value"])
        except ValueError:
            pass

    return settings


def save_setting(name, value):
    connection = get_connection()
    connection.execute(
        "INSERT INTO settings (name, value) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET value = excluded.value",
        (name, json.dumps(value)),
    )
    connection.commit()
    connection.close()


def add_user(username, password_hash, role, full_name=None, email=None):
    connection = get_connection()
    cursor = connection.execute(
        "INSERT INTO users (username, password_hash, role, full_name, email) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, password_hash, role, full_name, email),
    )
    user_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return user_id


def get_user_by_username(username):
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    connection.close()
    return row


def get_user(user_id):
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    connection.close()
    return row


def update_user_profile(user_id, full_name, email):
    connection = get_connection()
    connection.execute(
        "UPDATE users SET full_name = ?, email = ? WHERE user_id = ?",
        (full_name, email, user_id),
    )
    connection.commit()
    connection.close()


def count_users():
    connection = get_connection()
    count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    connection.close()
    return count


def add_audio_file(values):
    connection = get_connection()
    cursor = connection.execute(
        "INSERT INTO audio_files ("
        "uploaded_by, filename, stored_path, file_hash, fingerprint, audio_format, "
        "duration_sec, sample_rate, channels, bit_depth, file_size, source_kind"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            values.get("uploaded_by"),
            values.get("filename"),
            values.get("stored_path"),
            values.get("file_hash"),
            values.get("fingerprint"),
            values.get("audio_format"),
            values.get("duration_sec"),
            values.get("sample_rate"),
            values.get("channels"),
            values.get("bit_depth"),
            values.get("file_size"),
            values.get("source_kind", "upload"),
        ),
    )
    audio_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return audio_id


def get_audio_file(audio_id):
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM audio_files WHERE audio_id = ?", (audio_id,)
    ).fetchone()
    connection.close()
    return row


def find_audio_by_hash(file_hash):
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM audio_files WHERE file_hash = ? ORDER BY audio_id LIMIT 1",
        (file_hash,),
    ).fetchone()
    connection.close()
    return row


def find_audio_by_fingerprint(fingerprint):
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM audio_files WHERE fingerprint = ? ORDER BY audio_id LIMIT 1",
        (fingerprint,),
    ).fetchone()
    connection.close()
    return row


DETECTION_FIELDS = [
    "audio_id",
    "segment_index",
    "start_sec",
    "end_sec",
    "python_class",
    "python_scores",
    "python_confidence",
    "python_model_version",
    "gtm_class",
    "gtm_scores",
    "gtm_confidence",
    "gtm_model_version",
    "agreement_status",
    "confidence_difference",
    "top_two_margin",
    "overlapping_classes",
    "quality",
    "quality_notes",
    "noise_level",
    "final_class",
    "confidence_level",
    "severity",
    "alert_status",
    "recommended_action",
    "manual_review_required",
    "manual_review_reason",
    "status",
    "waveform_path",
    "spectrogram_path",
]


DETECTION_DEFAULTS = {
    "segment_index": 0,
    "manual_review_required": 0,
    "status": "Classified",
}


def add_detection(values):
    columns = ", ".join(DETECTION_FIELDS)
    placeholders = ", ".join("?" for _ in DETECTION_FIELDS)

    row_values = []
    for field in DETECTION_FIELDS:
        value = values.get(field)

        if value is None and field in DETECTION_DEFAULTS:
            value = DETECTION_DEFAULTS[field]

        row_values.append(value)

    connection = get_connection()
    cursor = connection.execute(
        f"INSERT INTO detections ({columns}) VALUES ({placeholders})", row_values
    )
    detection_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return detection_id


def update_detection_gtm(detection_id, values):
    connection = get_connection()
    connection.execute(
        "UPDATE detections SET "
        "gtm_class = ?, gtm_scores = ?, gtm_confidence = ?, gtm_model_version = ?, "
        "agreement_status = ?, confidence_difference = ?, top_two_margin = ?, "
        "final_class = ?, confidence_level = ?, severity = ?, alert_status = ?, "
        "recommended_action = ?, manual_review_required = ?, manual_review_reason = ?, "
        "status = ? WHERE detection_id = ?",
        (
            values.get("gtm_class"),
            values.get("gtm_scores"),
            values.get("gtm_confidence"),
            values.get("gtm_model_version"),
            values.get("agreement_status"),
            values.get("confidence_difference"),
            values.get("top_two_margin"),
            values.get("final_class"),
            values.get("confidence_level"),
            values.get("severity"),
            values.get("alert_status"),
            values.get("recommended_action"),
            values.get("manual_review_required"),
            values.get("manual_review_reason"),
            values.get("status"),
            detection_id,
        ),
    )
    connection.commit()
    connection.close()


def get_detection(detection_id):
    connection = get_connection()
    row = connection.execute(
        "SELECT d.*, a.filename, a.duration_sec, a.sample_rate, a.channels, "
        "a.audio_format, a.file_size, a.stored_path, a.uploaded_at, a.source_kind "
        "FROM detections d LEFT JOIN audio_files a ON a.audio_id = d.audio_id "
        "WHERE d.detection_id = ?",
        (detection_id,),
    ).fetchone()
    connection.close()
    return row


def get_recent_detections(limit=10):
    connection = get_connection()
    rows = connection.execute(
        "SELECT d.*, a.filename FROM detections d "
        "LEFT JOIN audio_files a ON a.audio_id = d.audio_id "
        "ORDER BY d.detection_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    connection.close()
    return rows


def get_recent_detections_for_class(final_class, seconds):
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM detections WHERE final_class = ? "
        "AND created_at >= datetime('now', ?) ORDER BY detection_id DESC",
        (final_class, f"-{int(seconds)} seconds"),
    ).fetchall()
    connection.close()
    return rows


def search_detections(filters, limit=200):
    where = []
    params = []

    if filters.get("audio_id"):
        where.append("d.audio_id = ?")
        params.append(filters["audio_id"])

    if filters.get("filename"):
        where.append("a.filename LIKE ?")
        params.append("%" + filters["filename"] + "%")

    if filters.get("sound_class"):
        where.append("d.final_class = ?")
        params.append(filters["sound_class"])

    if filters.get("date_from"):
        where.append("date(d.created_at) >= date(?)")
        params.append(filters["date_from"])

    if filters.get("date_to"):
        where.append("date(d.created_at) <= date(?)")
        params.append(filters["date_to"])

    if filters.get("min_confidence") is not None:
        where.append("d.python_confidence >= ?")
        params.append(filters["min_confidence"])

    if filters.get("max_confidence") is not None:
        where.append("d.python_confidence <= ?")
        params.append(filters["max_confidence"])

    if filters.get("severity"):
        where.append("d.severity = ?")
        params.append(filters["severity"])

    if filters.get("quality"):
        where.append("d.quality = ?")
        params.append(filters["quality"])

    if filters.get("status"):
        where.append("d.status = ?")
        params.append(filters["status"])

    if filters.get("user_id"):
        where.append("a.uploaded_by = ?")
        params.append(filters["user_id"])

    query = (
        "SELECT d.*, a.filename, a.uploaded_by FROM detections d "
        "LEFT JOIN audio_files a ON a.audio_id = d.audio_id"
    )

    if where:
        query += " WHERE " + " AND ".join(where)

    query += " ORDER BY d.detection_id DESC LIMIT ?"
    params.append(limit)

    connection = get_connection()
    rows = connection.execute(query, params).fetchall()
    connection.close()
    return rows


def get_review_queue(limit=100):
    connection = get_connection()
    rows = connection.execute(
        "SELECT d.*, a.filename, a.stored_path FROM detections d "
        "LEFT JOIN audio_files a ON a.audio_id = d.audio_id "
        "WHERE d.manual_review_required = 1 AND d.status != 'Closed' "
        "ORDER BY d.detection_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    connection.close()
    return rows


def add_review(detection_id, reviewer_id, decision, corrected_class, comments, recommended_action):
    connection = get_connection()
    connection.execute(
        "INSERT INTO reviews (detection_id, reviewer_id, decision, corrected_class, "
        "comments, recommended_action) VALUES (?, ?, ?, ?, ?, ?)",
        (detection_id, reviewer_id, decision, corrected_class, comments, recommended_action),
    )

    # The model outputs are never overwritten, only the reviewed outcome.
    connection.execute(
        "UPDATE detections SET reviewer_decision = ?, status = 'Reviewed', "
        "manual_review_required = 0 WHERE detection_id = ?",
        (decision, detection_id),
    )
    connection.commit()
    connection.close()


def get_reviews(detection_id):
    connection = get_connection()
    rows = connection.execute(
        "SELECT r.*, u.username FROM reviews r "
        "LEFT JOIN users u ON u.user_id = r.reviewer_id "
        "WHERE r.detection_id = ? ORDER BY r.review_id",
        (detection_id,),
    ).fetchall()
    connection.close()
    return rows


def add_alert(detection_id, severity, message):
    connection = get_connection()
    cursor = connection.execute(
        "INSERT INTO alerts (detection_id, severity, message) VALUES (?, ?, ?)",
        (detection_id, severity, message),
    )
    alert_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return alert_id


def get_alerts(status=None, limit=100):
    connection = get_connection()

    if status:
        rows = connection.execute(
            "SELECT al.*, d.final_class, d.audio_id FROM alerts al "
            "LEFT JOIN detections d ON d.detection_id = al.detection_id "
            "WHERE al.status = ? ORDER BY al.alert_id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT al.*, d.final_class, d.audio_id FROM alerts al "
            "LEFT JOIN detections d ON d.detection_id = al.detection_id "
            "ORDER BY al.alert_id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    connection.close()
    return rows


def update_alert_status(alert_id, status, user_id):
    connection = get_connection()
    connection.execute(
        "UPDATE alerts SET status = ?, handled_by = ?, "
        "handled_at = CURRENT_TIMESTAMP WHERE alert_id = ?",
        (status, user_id, alert_id),
    )
    connection.commit()
    connection.close()


def add_audit(user_id, action_type, details=None):
    connection = get_connection()
    connection.execute(
        "INSERT INTO audit_log (user_id, action_type, details) VALUES (?, ?, ?)",
        (user_id, action_type, details),
    )
    connection.commit()
    connection.close()


def get_audit_log(limit=100):
    connection = get_connection()
    rows = connection.execute(
        "SELECT al.*, u.username FROM audit_log al "
        "LEFT JOIN users u ON u.user_id = al.user_id "
        "ORDER BY al.audit_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    connection.close()
    return rows


def count_recent_audit(action_type, seconds):
    connection = get_connection()
    count = connection.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action_type = ? "
        "AND created_at >= datetime('now', ?)",
        (action_type, f"-{int(seconds)} seconds"),
    ).fetchone()[0]
    connection.close()
    return count


def get_statistics():
    connection = get_connection()

    stats = {}
    stats["total_detections"] = connection.execute(
        "SELECT COUNT(*) FROM detections"
    ).fetchone()[0]

    stats["total_audio"] = connection.execute(
        "SELECT COUNT(*) FROM audio_files"
    ).fetchone()[0]

    stats["critical_alerts"] = connection.execute(
        "SELECT COUNT(*) FROM alerts WHERE severity = 'Critical'"
    ).fetchone()[0]

    stats["active_alerts"] = connection.execute(
        "SELECT COUNT(*) FROM alerts WHERE status = 'active'"
    ).fetchone()[0]

    average = connection.execute(
        "SELECT AVG(python_confidence) FROM detections"
    ).fetchone()[0]
    stats["average_confidence"] = round(average, 3) if average else 0.0

    stats["disagreements"] = connection.execute(
        "SELECT COUNT(*) FROM detections WHERE agreement_status = 'Model Disagreement'"
    ).fetchone()[0]

    stats["poor_quality"] = connection.execute(
        "SELECT COUNT(*) FROM detections WHERE quality IN ('Poor', 'Unusable')"
    ).fetchone()[0]

    stats["pending_review"] = connection.execute(
        "SELECT COUNT(*) FROM detections WHERE manual_review_required = 1"
    ).fetchone()[0]

    stats["by_class"] = connection.execute(
        "SELECT final_class, COUNT(*) AS total FROM detections "
        "GROUP BY final_class ORDER BY total DESC"
    ).fetchall()

    stats["by_severity"] = connection.execute(
        "SELECT severity, COUNT(*) AS total FROM detections "
        "GROUP BY severity ORDER BY total DESC"
    ).fetchall()

    stats["by_quality"] = connection.execute(
        "SELECT quality, COUNT(*) AS total FROM detections "
        "GROUP BY quality ORDER BY total DESC"
    ).fetchall()

    stats["trend"] = connection.execute(
        "SELECT date(created_at) AS day, COUNT(*) AS total FROM detections "
        "GROUP BY day ORDER BY day DESC LIMIT 14"
    ).fetchall()

    stats["confidence_buckets"] = connection.execute(
        "SELECT CASE "
        "WHEN python_confidence >= 0.9 THEN '0.9 - 1.0' "
        "WHEN python_confidence >= 0.7 THEN '0.7 - 0.9' "
        "WHEN python_confidence >= 0.5 THEN '0.5 - 0.7' "
        "ELSE 'below 0.5' END AS bucket, COUNT(*) AS total "
        "FROM detections WHERE python_confidence IS NOT NULL "
        "GROUP BY bucket ORDER BY bucket DESC"
    ).fetchall()

    stats["reviewer_outcomes"] = connection.execute(
        "SELECT decision, COUNT(*) AS total FROM reviews "
        "GROUP BY decision ORDER BY total DESC"
    ).fetchall()

    connection.close()
    return stats


def delete_records_older_than(days):
    connection = get_connection()
    cursor = connection.execute(
        "DELETE FROM detections WHERE created_at < datetime('now', ?)",
        (f"-{int(days)} days",),
    )
    removed = cursor.rowcount
    connection.execute(
        "DELETE FROM audio_files WHERE uploaded_at < datetime('now', ?)",
        (f"-{int(days)} days",),
    )
    connection.commit()
    connection.close()
    return removed


if __name__ == "__main__":
    init_db()
    print("Database ready at", DB_PATH)
