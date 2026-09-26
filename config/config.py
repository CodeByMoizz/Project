import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change this one line to switch the active Python model after training.
# It must match a key in PYTHON_MODELS below.
ACTIVE_PYTHON_MODEL = "random_forest"

PYTHON_MODELS = {
    "random_forest": {
        "model_file": "random_forest_model.pkl",
        "scaler_file": "random_forest_scaler.pkl",
        "metrics_file": "random_forest_metrics.json",
        "version": "rf-v1",
    },
    "svm": {
        "model_file": "svm_model.pkl",
        "scaler_file": "svm_scaler.pkl",
        "metrics_file": "svm_metrics.json",
        "version": "svm-v1",
    },
    "xgboost": {
        "model_file": "xgboost_model.pkl",
        "scaler_file": "xgboost_scaler.pkl",
        "metrics_file": "xgboost_metrics.json",
        "version": "xgb-v1",
    },
    "logistic_regression": {
        "model_file": "logistic_regression_model.pkl",
        "scaler_file": "logistic_regression_scaler.pkl",
        "metrics_file": "logistic_regression_metrics.json",
        "version": "lr-v1",
    },
}

PYTHON_MODEL_DIR = PROJECT_ROOT / "python_models"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# The GTM model is exported from Teachable Machine as TensorFlow.js and runs in
# the browser. Put model.json, metadata.json and the .bin weights in gtm_model/,
# or set GTM_MODEL_URL to the hosted Teachable Machine URL instead.
GTM_MODEL_DIR = PROJECT_ROOT / "gtm_model"
GTM_MODEL_URL = os.environ.get("GTM_MODEL_URL", "")
GTM_MODEL_VERSION = "gtm-v1"

ALERT_RULES_FILE = PROJECT_ROOT / "alert_rules" / "alert_rules.json"

DATABASE_FILE = PROJECT_ROOT / "database" / "sonicsentinel.db"

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
SEGMENT_DIR = PROJECT_ROOT / "data" / "segments"
VISUAL_DIR = PROJECT_ROOT / "static" / "visuals"
METRICS_DIR = PROJECT_ROOT / "reports"
DATASET_DIR = PROJECT_ROOT / "audio_dataset"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SECRET_KEY = os.environ.get("SONICSENTINEL_SECRET_KEY", "sonicsentinel-dev-key")

# Audio settings. These must stay the same for training and for inference,
# because the feature vector changes if they do.
TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION_SEC = 2.0
SILENCE_TOP_DB = 30

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_BATCH_FILES = 20
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# Defaults for the admin-configurable thresholds. The live values are kept in
# the settings table so an administrator can change them without a restart.
DEFAULT_SETTINGS = {
    "min_confidence": 0.60,
    "top_two_margin": 0.15,
    "max_confidence_difference": 0.30,
    "repeated_detection_window_sec": 30,
    "noise_alert_level": 0.50,
    "overlap_confidence": 0.25,
    "retention_days": 90,
    "failed_login_alert_count": 5,
}

# Canonical class ids. Used as folder names, model labels and database values.
CLASSES = [
    "machinery_fault",
    "glass_breaking",
    "alarm_siren",
    "vehicle_horn",
    "animal_sound",
    "gunshot",
    "panic_scream",
    "aggression",
    "person_asking_help",
    "background_noise",
]

CLASS_NAMES = {
    "machinery_fault": "Machinery Fault",
    "glass_breaking": "Glass Breaking",
    "alarm_siren": "Alarm or Siren",
    "vehicle_horn": "Vehicle Horn",
    "animal_sound": "Animal Sound",
    "gunshot": "Gunshot",
    "panic_scream": "Panic Scream",
    "aggression": "Aggression",
    "person_asking_help": "Person Asking for Help",
    "background_noise": "Background Noise",
    "unknown": "Unknown Sound",
}

CRITICAL_CLASSES = [
    "gunshot",
    "panic_scream",
    "person_asking_help",
    "aggression",
    "glass_breaking",
]

# Classes whose recordings contain speech. The dataset split has to group these
# by speaker as well as by recording, otherwise recall is inflated.
SPEECH_CLASSES = [
    "panic_scream",
    "aggression",
    "person_asking_help",
]

ROLES = ["user", "reviewer", "operator", "maintenance", "admin"]

EVENT_STATUSES = [
    "Uploaded",
    "Classified",
    "Uncertain",
    "Alert Generated",
    "Manual Review",
    "Reviewed",
    "Closed",
]


def display_name(class_id):
    return CLASS_NAMES.get(class_id, class_id)


def active_model_config():
    return PYTHON_MODELS[ACTIVE_PYTHON_MODEL]
