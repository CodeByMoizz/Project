import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sonicsentinel.db"

connection = sqlite3.connect(DB_PATH)

cursor = connection.cursor()

cursor.execute("PRAGMA foreign_keys = ON")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL
        CHECK (role IN (
            'user',
            'reviewer',
            'operator',
            'maintenance',
            'admin'
        ))
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS audio_files (
    audio_id INTEGER PRIMARY KEY,
    uploaded_by INTEGER NOT NULL,
    stored_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    augmentation_type TEXT,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (uploaded_by)
        REFERENCES users(user_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY,
    audio_id INTEGER,
    python_class TEXT NOT NULL,
    python_scores TEXT NOT NULL,
    python_model_version TEXT NOT NULL,
    gtm_class TEXT,
    gtm_scores TEXT,
    gtm_model_version TEXT,
    final_class TEXT NOT NULL,
    severity TEXT NOT NULL,
    alert_status TEXT NOT NULL,
    reviewer_decision TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (audio_id)
        REFERENCES audio_files(audio_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    action_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(user_id)
)
""")

connection.commit()

connection.close()

print("SQLite database created successfully!")
print("All 4 tables created successfully!")